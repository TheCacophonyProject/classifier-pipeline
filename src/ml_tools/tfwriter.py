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
from pathlib import Path
import multiprocessing
_spawn_ctx = multiprocessing.get_context("spawn")
import os
from absl import logging
import numpy as np
import psutil


def process_job(queue, labels, base_dir, save_data_name, writer_i,excluded_tags, extra_args):
    import gc
    import tensorflow as tf
    from ml_tools.logs import init_logging
    init_logging()
    
    if save_data_name == "thermal":
        from ml_tools.thermalwriter import save_data
    elif save_data_name == "ir":
        from ml_tools.irwriter import save_data
    else:
        raise ValueError(f"Unknown save_data_name: {save_data_name}")

    pid = os.getpid()

    name = f"{writer_i}-{pid}.tfrecord"
    mem_mb = psutil.Process().memory_info().rss / 1024**2
    logging.info("Writing to %s mem usage %s", name,mem_mb)
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    writer = tf.io.TFRecordWriter(str(base_dir / name), options=options)
    i = 0
    saved = 0
    files = 0
    num_frames = extra_args.get("num_frames", 25)
    while True:
        i += 1
        source_file = queue.get()
        try:
            if isinstance(source_file, str):
                if source_file == "DONE":

                    mem_mb = psutil.Process().memory_info().rss / 1024**2
                    logging.info("Worker %s done: received %s samples, processed %s files memory %s", name,i, files,mem_mb)
                    writer.close()
                    break
                else:
                    logging.error("Unknown string %s",source_file)
            else:
                saved += save_data(source_file, excluded_tags,writer, labels, extra_args)
                files += 1

                if i % int(2500 / num_frames) == 0:
                    mem_mb = psutil.Process().memory_info().rss / 1024**2
                    logging.info("Saved %s to %s  mem %.1f MB", files, name, mem_mb)
                    gc.collect()
                    writer.flush()
        except:
            logging.error("Process_job error %s", source_file, exc_info=True)


def create_tf_records(
    dataset,
    output_path,
    labels,
    save_data_name,
    excluded_tags,
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
    source_files =list(dataset.source_files)
    dataset.clear()

    np.random.shuffle(source_files)
    logging.info(
        "writing to output path: %s for %s recordings", output_path, len(source_files))
    
    num_processes = 8
    writer_i = 0
    index = 0
    jobs_per_process = 100 * num_processes
    logging.info("Writing samples")
    try:
        while index < len(source_files):
            job_queue = _spawn_ctx.Queue()
            processes = []
            for i in range(num_processes):
                p = _spawn_ctx.Process(
                    target=process_job,
                    args=(
                        job_queue,
                        labels,
                        output_path,
                        save_data_name,
                        writer_i,
                        excluded_tags,
                        extra_args,
                    ),
                )
                processes.append(p)
                p.start()
                added = 0
            writer_i += 1
            for source_file in source_files[index : index + jobs_per_process]:
                job_queue.put(source_file)
                added += 1

            index += jobs_per_process
            mem_mb = psutil.Process().memory_info().rss / 1024**2
            logging.info("Processing %d mem %.1f MB", job_queue.qsize(), mem_mb)
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
            mem_mb = psutil.Process().memory_info().rss / 1024**2
            logging.info("Saved %s mem %.1f MB", len(dataset.samples_by_id), mem_mb)

    except:
        logging.error("Error saving track info", exc_info=True)
