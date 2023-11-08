# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from PIL import Image
from pathlib import Path
from multiprocessing import Process, Queue

import collections
import hashlib
import io
import json
import multiprocessing
import os
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image, ImageOps

import tensorflow as tf
from . import tfrecord_util
from ml_tools import tools
from ml_tools.imageprocessing import normalize, rotate
from track.cliptracker import get_diff_back_filtered
import cv2
import random
import math


def process_job(queue, labels, base_dir, save_data, extra_args):
    import gc

    pid = os.getpid()

    writer_i = 1
    name = f"{writer_i}-{pid}.tfrecord"

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    writer = tf.io.TFRecordWriter(str(base_dir / name), options=options)
    i = 0
    saved = 0
    files = 0
    num_frames = extra_args.get("num_frames", 25)
    while True:
        i += 1
        samples = queue.get()
        try:
            if samples == "DONE":
                writer.close()
                break
            else:
                if len(samples) == 0:
                    continue
                saved += save_data(samples, writer, labels, extra_args)
                files += 1
                del samples
                if saved > 250000 / num_frames:
                    logging.info("Closing old writer")
                    writer.close()
                    writer_i += 1
                    name = f"{writer_i}-{pid}.tfrecord"
                    logging.info("Opening %s", name)
                    saved = 0
                    writer = tf.io.TFRecordWriter(str(base_dir / name), options=options)
                if i % int(25000 / num_frames) == 0:
                    logging.info("Saved %s ", files)
                    gc.collect()
                    writer.flush()
        except:
            logging.error("Process_job error %s", samples[0].source_file, exc_info=True)


def create_tf_records(
    dataset,
    output_path,
    labels,
    save_data,
    num_shards=1,
    augment=False,
    **extra_args,
):
    output_path = Path(output_path)
    if output_path.is_dir():
        logging.info("Clearing dir %s", output_path)
        for child in output_path.glob("*"):
            if child.is_file():
                child.unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    samples_by_source = dataset.get_samples_by_source()
    source_files = list(samples_by_source.keys())
    np.random.shuffle(source_files)

    num_labels = len(dataset.labels)
    logging.info(
        "writing to output path: %s for %s samples", output_path, len(samples_by_source)
    )
    num_processes = 8
    try:
        job_queue = Queue()
        processes = []
        for i in range(num_processes):
            p = Process(
                target=process_job,
                args=(job_queue, labels, output_path, save_data, extra_args),
            )
            processes.append(p)
            p.start()
            added = 0
        for source_file in source_files:
            job_queue.put((samples_by_source[source_file]))
            added += 1
            while job_queue.qsize() > num_processes * 10:
                logging.info("Sleeping for %s", 10)
                # give it a change to catch up
                time.sleep(10)

        logging.info("Processing %d", job_queue.qsize())
        for i in range(len(processes)):
            job_queue.put(("DONE"))
        for process in processes:
            try:
                process.join()
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt, terminating.")
                for process in processes:
                    process.terminate()
                exit()
        logging.info("Saved %s", len(dataset.samples_by_id))

    except:
        logging.error("Error saving track info", exc_info=True)
