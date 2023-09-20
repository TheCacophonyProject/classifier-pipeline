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

crop_rectangle = tools.Rectangle(0, 0, 640 - 1, 480 - 1)


def create_tf_example(sample, thermal, filtered, labels):
    """Converts image and annotations to a tf.Example proto.

    Args:
      image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
        u'width', u'date_captured', u'flickr_url', u'id']
      image_dir: directory containing the image files.
      bbox_annotations:
        list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
          u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
          coordinates in the official COCO dataset are given as [x, y, width,
          height] tuples using absolute coordinates where x, y represent the
          top-left (0-indexed) corner.  This function converts to the format
          expected by the Tensorflow Object Detection API (which is which is
          [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
          size).
      category_index: a dict containing COCO category information keyed by the
        'id' field of each category.  See the label_map_util.create_category_index
        function.
      caption_annotations:
        list of dict with keys: [u'id', u'image_id', u'str'].
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.

    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height, image_width = thermal.shape

    image = Image.fromarray(np.uint8(thermal))

    image_id = sample.unique_id

    encoded_jpg_io = io.BytesIO()
    image.save(encoded_jpg_io, format="PNG", quality=100, subsampling=0)

    encoded_thermal = encoded_jpg_io.getvalue()
    thermal_key = hashlib.sha256(encoded_thermal).hexdigest()

    image = Image.fromarray(np.uint8(filtered))

    encoded_jpg_io = io.BytesIO()
    image.save(encoded_jpg_io, format="PNG", quality=100, subsampling=0)
    encoded_filtered = encoded_jpg_io.getvalue()
    filtered_key = hashlib.sha256(encoded_filtered).hexdigest()
    # if frame.mask:
    # image = Image.fromarray(frame.mask)
    # image = ImageOps.grayscale(image)
    # image_id = sample.id
    #
    # encoded_jpg_io = io.BytesIO()
    # image.save(encoded_jpg_io, format="JPEG")
    # encoded_mask = encoded_jpg_io.getvalue()
    # mask_key = hashlib.sha256(encoded_mask).hexdigest()
    feature_dict = {
        "image/augmented": tfrecord_util.int64_feature(sample.augment),
        "image/height": tfrecord_util.int64_feature(image_height),
        "image/width": tfrecord_util.int64_feature(image_width),
        "image/filename": tfrecord_util.bytes_feature(
            str(sample.source_file).encode("utf8")
        ),
        "image/source_id": tfrecord_util.bytes_feature(str(image_id).encode("utf8")),
        "image/thermalkey/sha256": tfrecord_util.bytes_feature(
            thermal_key.encode("utf8")
        ),
        "image/thermalencoded": tfrecord_util.bytes_feature(encoded_thermal),
        "image/filteredkey/sha256": tfrecord_util.bytes_feature(
            filtered_key.encode("utf8")
        ),
        "image/clip_id": tfrecord_util.int64_feature(sample.clip_id),
        "image/track_id": tfrecord_util.int64_feature(sample.track_id),
        "image/filteredencoded": tfrecord_util.bytes_feature(encoded_filtered),
        # "image/maskkey/sha256": tfrecord_util.bytes_feature(mask_key.encode("utf8")),
        # "image/maskencoded": tfrecord_util.bytes_feature(encoded_mask),
        "image/format": tfrecord_util.bytes_feature("jpeg".encode("utf8")),
        "image/class/text": tfrecord_util.bytes_feature(sample.label.encode("utf8")),
        "image/class/label": tfrecord_util.int64_feature(labels.index(sample.label)),
        # "image/object/bbox/xmin": tfrecord_util.float_list_feature([0]),
        # "image/object/bbox/xmax": tfrecord_util.float_list_feature([1]),
        # "image/object/bbox/ymin": tfrecord_util.float_list_feature([0]),
        # "image/object/bbox/ymax": tfrecord_util.float_list_feature([1]),
    }
    #
    # num_annotations_skipped = 0
    # if len(sample.regions) > 0:
    #     xmin = []
    #     xmax = []
    #     ymin = []
    #     ymax = []
    #     is_crowd = []
    #     category_names = []
    #     category_ids = []
    #     area = []
    #     encoded_mask_png = []
    #     for r, label in zip(sample.regions, sample.labels):
    #         (x, y, width, height) = r.x, r.y, r.width, r.height
    #         if width <= 0 or height <= 0:
    #             num_annotations_skipped += 1
    #             continue
    #         if x + width > image_width or y + height > image_height:
    #             num_annotations_skipped += 1
    #             continue
    #         xmin.append(float(x) / image_width)
    #         xmax.append(float(x + width) / image_width)
    #         ymin.append(float(y) / image_height)
    #         ymax.append(float(y + height) / image_height)
    #         category_id = int(labels.index(label) + 1)
    #         category_ids.append(category_id)
    #         category_names.append(label.encode("utf8"))
    #         area.append(r.area)
    #     feature_dict.update(
    #         {
    #             "image/object/bbox/xmin": tfrecord_util.float_list_feature(xmin),
    #             "image/object/bbox/xmax": tfrecord_util.float_list_feature(xmax),
    #             "image/object/bbox/ymin": tfrecord_util.float_list_feature(ymin),
    #             "image/object/bbox/ymax": tfrecord_util.float_list_feature(ymax),
    #             "image/object/class/text": tfrecord_util.bytes_list_feature(
    #                 category_names
    #             ),
    #             "image/object/class/label": tfrecord_util.int64_list_feature(
    #                 category_ids
    #             ),
    #             "image/object/area": tfrecord_util.float_list_feature(area),
    #         }
    #     )
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def process_job(queue, labels, base_dir, back_thresh):
    import gc

    pid = os.getpid()

    writer_i = 1
    name = f"{writer_i}-{pid}.tfrecord"

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    writer = tf.io.TFRecordWriter(str(base_dir / name), options=options)
    i = 0
    saved = 0
    files = 0
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
                saved += save_data(samples, writer, labels, back_thresh)
                files += 1
                del samples
                if saved > 10000:
                    logging.info("Closing old writer")
                    writer.close()
                    writer_i += 1
                    name = f"{writer_i}-{pid}.tfrecord"
                    logging.info("Opening %s", name)
                    saved = 0
                    writer = tf.io.TFRecordWriter(str(base_dir / name), options=options)
                if i % 100 == 0:
                    logging.info("Saved %s ", files)
                    gc.collect()
                    writer.flush()
        except:
            logging.error("Process_job error %s", samples[0].source_file, exc_info=True)


def get_data(samples, back_thresh):
    vidcap = cv2.VideoCapture(str(samples[0].source_file))
    frames = {}
    backgorund = None
    frame_num = 0
    print("Loading ", str(samples[0].source_file))
    while True:
        success, image = vidcap.read()
        is_background_frame = False
        if not success:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if backgorund is None:
            is_background_frame = np.all(image[:, :, 0] == image[:, :, 1]) and np.all(
                image[:, :, 1] == image[:, :, 2]
            )
            background = np.uint8(gray)
        if not is_background_frame:
            frames[frame_num] = gray
            frame_num += 1
            # append(gray)
    data = []
    for sample in samples:
        frame = frames[sample.region.frame_number]
        gray_sub = sample.region.subimage(frame)
        back_sub = sample.region.subimage(background)
        filtered = get_diff_back_filtered(back_sub, gray_sub, back_thresh)
        gray_sub, stats = normalize(gray_sub, new_max=255)
        if not stats[0]:
            continue
        filtered, stats = normalize(filtered, new_max=255)
        cv2.imwrite(f"{sample.id}-{sample.track_id}-{sample.label}.png", gray_sub)
        if not stats[0]:
            continue
        data.append((sample, gray_sub, filtered))

    return data


def save_data(samples, writer, labels, back_thresh):
    sample_data = get_data(samples, back_thresh)
    if sample_data is None:
        return 0
    saved = 0
    try:
        for sample, thermal, filtered in sample_data:
            tf_example = create_tf_example(
                sample,
                thermal,
                filtered,
                labels,
            )
            writer.write(tf_example.SerializeToString())
            saved += 1
    except:
        logging.error(
            "Could not save data for %s", samples[0].source_file, exc_info=True
        )
    return saved


def create_tf_records(
    dataset, output_path, labels, back_thresh, num_shards=1, cropped=True, augment=False
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
    num_processes = 1
    try:
        job_queue = Queue()
        processes = []
        for i in range(num_processes):
            p = Process(
                target=process_job,
                args=(job_queue, labels, output_path, back_thresh),
            )
            processes.append(p)
            p.start()
            added = 0
        for source_file in source_files:
            job_queue.put((samples_by_source[source_file]))
            added += 1
            while job_queue.qsize() > num_processes * 3:
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
