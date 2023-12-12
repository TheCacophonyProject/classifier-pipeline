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
        "image/format": tfrecord_util.bytes_feature("jpeg".encode("utf8")),
        "image/class/text": tfrecord_util.bytes_feature(sample.label.encode("utf8")),
        "image/class/label": tfrecord_util.int64_feature(labels.index(sample.label)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def get_data(samples, back_thresh):
    vidcap = cv2.VideoCapture(str(samples[0].source_file))
    frames = {}
    backgorund = None
    frame_num = 0
    frames_needed = [s.region.frame_number for s in samples]
    frames_needed.sort()
    if len(frames_needed) == 0:
        return []
    while True:
        for _ in range(2):
            # try read first frame twice
            success, image = vidcap.read()
            is_background_frame = False
            if success or frame_num > 0:
                break
        if not success:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if backgorund is None:
            is_background_frame = np.all(image[:, :, 0] == image[:, :, 1]) and np.all(
                image[:, :, 1] == image[:, :, 2]
            )
            background = np.uint8(gray)
        if not is_background_frame and frame_num in frames_needed:
            frames[frame_num] = gray
            # append(gray)
        frame_num += 1
        if frame_num > frames_needed[-1]:
            break
    data = []
    failed = []
    for sample in samples:
        if sample.region.frame_number not in frames:
            failed.append(sample.region.frame_number)
            continue
        frame = frames[sample.region.frame_number]
        gray_sub = sample.region.subimage(frame)
        back_sub = sample.region.subimage(background)
        filtered = get_diff_back_filtered(back_sub, gray_sub, back_thresh)
        gray_sub, stats = normalize(gray_sub, new_max=255)
        if not stats[0]:
            continue
        filtered, stats = normalize(filtered, new_max=255)
        if not stats[0]:
            continue
        data.append((sample, gray_sub, filtered))
    if len(failed) > 0:
        logging.warning("Could not get %s for %s", failed, str(samples[0].source_file))

    return data


def save_data(samples, writer, labels, extra_args):
    sample_data = get_data(samples, extra_args["back_thresh"])
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
