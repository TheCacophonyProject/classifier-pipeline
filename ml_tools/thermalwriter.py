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
r"""Convert raw COCO 2017 dataset to TFRecord.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --caption_annotations_file="${CAPTION_ANNOTATIONS_FILE}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""
from PIL import Image
from pathlib import Path

import collections
import hashlib
import io
import json
import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image, ImageOps

from pycocotools import mask
import tensorflow as tf
from . import tfrecord_util
from ml_tools import tools
from ml_tools.imageprocessing import normalize

crop_rectangle = tools.Rectangle(0, 0, 640, 480)


def create_tf_example(data, image_dir, sample, labels, filename):
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
    average_dim = [r.area for r in sample.regions]
    average_dim = int(round(np.mean(average_dim) ** 0.5))
    thermal = data[0] * 255
    filtered = data[1] * 255
    image_height, image_width = thermal.shape
    image = Image.fromarray(thermal)
    image = ImageOps.grayscale(image)

    image_id = sample.id

    encoded_jpg_io = io.BytesIO()
    image.save(encoded_jpg_io, format="JPEG")

    encoded_thermal = encoded_jpg_io.getvalue()
    thermal_key = hashlib.sha256(encoded_thermal).hexdigest()

    image = Image.fromarray(filtered)
    image = ImageOps.grayscale(image)

    encoded_jpg_io = io.BytesIO()
    image.save(encoded_jpg_io, format="JPEG")
    encoded_filtered = encoded_jpg_io.getvalue()
    filtered_key = hashlib.sha256(encoded_filtered).hexdigest()

    feature_dict = {
        "image/avg_dim": tfrecord_util.int64_feature(average_dim),
        "image/height": tfrecord_util.int64_feature(image_height),
        "image/width": tfrecord_util.int64_feature(image_width),
        "image/clip_id": tfrecord_util.int64_feature(sample.clip_id),
        "image/track_iod": tfrecord_util.int64_feature(sample.track_id),
        "image/filename": tfrecord_util.bytes_feature(filename.encode("utf8")),
        "image/source_id": tfrecord_util.bytes_feature(str(image_id).encode("utf8")),
        "image/thermalencoded": tfrecord_util.bytes_feature(encoded_thermal),
        "image/filteredencoded": tfrecord_util.bytes_feature(encoded_filtered),
        "image/filteredkey/sha256": tfrecord_util.bytes_feature(
            filtered_key.encode("utf8")
        ),
        "image/thermalkey/sha256": tfrecord_util.bytes_feature(
            thermal_key.encode("utf8")
        ),
        "image/format": tfrecord_util.bytes_feature("jpeg".encode("utf8")),
        "image/class/text": tfrecord_util.bytes_feature(sample.label.encode("utf8")),
        "image/class/label": tfrecord_util.int64_feature(labels.index(sample.label)),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


def create_tf_records(dataset, output_path, labels, num_shards=1, cropped=True):

    output_path = Path(output_path)
    if output_path.is_dir():
        logging.info("Clearing dir %s", output_path)
        for child in output_path.glob("*"):
            if child.is_file():
                child.unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    samples = dataset.samples
    # keys = list(samples.keys())
    np.random.shuffle(samples)

    dataset.load_db()
    db = dataset.db
    total_num_annotations_skipped = 0
    num_labels = len(labels)
    # pool = multiprocessing.Pool(4)
    logging.info("writing to output path: %s for %s samples", output_path, len(samples))
    writers = []
    lbl_counts = [0] * num_labels
    # lbl_counts[l] = 0
    logging.info("labels are %s", labels)

    writers = []
    for label in labels:
        for i in range(num_shards):

            writers.append(
                tf.io.TFRecordWriter(
                    str(
                        output_path
                        / (f"{label}-%05d-of-%05d.tfrecord" % (i, num_shards))
                    )
                )
            )

    load_first = 200
    try:
        count = 0
        while len(samples) > 0:
            local_set = samples[:load_first]
            samples = samples[load_first:]
            loaded = []

            for sample in local_set:
                data = sample.get_data(db)
                if data is None:
                    continue

                loaded.append((data, sample))

            loaded = np.array(loaded)
            np.random.shuffle(loaded)

            for data, sample in loaded:
                try:
                    tf_example, num_annotations_skipped = create_tf_example(
                        data, output_path, sample, labels, ""
                    )
                    total_num_annotations_skipped += num_annotations_skipped
                    l_i = labels.index(sample.label)
                    writers[num_shards * l_i + lbl_counts[l_i] % num_shards].write(
                        tf_example.SerializeToString()
                    )
                    lbl_counts[l_i] += 1
                    # print("saving example", [count % num_shards])
                    count += 1
                    if count % 100 == 0:
                        logging.info("saved %s", count)
                    # count += 1
                except Exception as e:
                    logging.error("Error saving ", exc_info=True)
                    raise e
            # break
    except:
        logging.error("Error saving track info", exc_info=True)
        raise "EX"
    for writer in writers:
        writer.close()

    logging.info(
        "Finished writing, skipped %d annotations.", total_num_annotations_skipped
    )
