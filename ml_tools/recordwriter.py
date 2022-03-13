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


def create_tf_example(frame, image_dir, sample, labels, filename):
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
    image_height, image_width = frame.thermal.shape
    image = Image.fromarray(frame.thermal)
    image = ImageOps.grayscale(image)

    image_id = sample.id

    encoded_jpg_io = io.BytesIO()
    image.save(encoded_jpg_io, format="JPEG")
    encoded_thermal = encoded_jpg_io.getvalue()
    thermal_key = hashlib.sha256(encoded_thermal).hexdigest()

    image = Image.fromarray(frame.filtered)
    image = ImageOps.grayscale(image)
    image_id = sample.id

    encoded_jpg_io = io.BytesIO()
    image.save(encoded_jpg_io, format="JPEG")
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
        "image/height": tfrecord_util.int64_feature(image_height),
        "image/width": tfrecord_util.int64_feature(image_width),
        "image/filename": tfrecord_util.bytes_feature(filename.encode("utf8")),
        "image/source_id": tfrecord_util.bytes_feature(str(image_id).encode("utf8")),
        "image/thermalkey/sha256": tfrecord_util.bytes_feature(
            thermal_key.encode("utf8")
        ),
        "image/thermalencoded": tfrecord_util.bytes_feature(encoded_thermal),
        "image/filteredkey/sha256": tfrecord_util.bytes_feature(
            filtered_key.encode("utf8")
        ),
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
    return example, 0


def create_tf_records(dataset, output_path, num_shards=1):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    samples = dataset.samples
    # keys = list(samples.keys())
    np.random.shuffle(samples)

    dataset.load_db()
    db = dataset.db
    total_num_annotations_skipped = 0
    # pool = multiprocessing.Pool(4)
    logging.info("writing to output path: %s for %s samples", output_path, len(samples))
    writers = [
        tf.io.TFRecordWriter(
            str(output_path / ("%05d-of-%05d.tfrecord" % (i, num_shards)))
        )
        for i in range(num_shards)
    ]
    load_first = 200
    try:
        count = 0
        while len(samples) > 0:
            local_set = samples[:load_first]
            samples = samples[load_first:]
            loaded = []

            for sample in local_set:
                try:
                    background = db.get_clip_background(sample.clip_id)
                    try:
                        f = db.get_clip(
                            sample.clip_id,
                            [sample.frame_number],
                        )[0]
                    except Exception as e:
                        print(sample.clip_id, sample.frame_number, "error", e)
                        continue
                    thresh = np.percentile(f.thermal, 15)
                    region = sample.regions[0].copy()
                    region.enlarge(20, max=crop_rectangle)
                    f.crop_by_region(region, out=f)
                    background = region.subimage(background)
                    # f.thermal -= int(thresh)
                    # f.thermal, _ = normalize(f.thermal, new_max=255)
                    # print("normaled", normaled.dtype)
                    f.mask = f.filtered
                    f.filtered = f.thermal - background
                    f.filtered[f.filtered < 0] = 0
                    # f.filtered[f.filtered > 10] += 30
                    # f.thermal -= int(thresh)
                    # f.filtered, _ = normalize(f.filtered, new_max=255)
                    assert f.thermal.shape == f.filtered.shape
                    loaded.append((f, sample))
                except Exception as e:
                    logging.error("GOt exception", exc_info=True)
                    pass
            loaded = np.array(loaded)
            np.random.shuffle(loaded)
            print("shuffled loaded", len(loaded))
            for data, sample in loaded:
                try:
                    tf_example, num_annotations_skipped = create_tf_example(
                        data, output_path, sample, dataset.labels, sample.filename
                    )
                    total_num_annotations_skipped += num_annotations_skipped
                    writers[count % num_shards].write(tf_example.SerializeToString())
                    # print("saving example", [count % num_shards])
                    count += 1
                    if count % 100 == 0:
                        logging.debug("saved %s", count)
                except Exception as e:
                    logging.error("Error saving ", exc_info=True)
    except:
        raise "EX"
        logging.error("Error saving track info", exc_info=True)

    for writer in writers:
        writer.close()

    logging.info(
        "Finished writing, skipped %d annotations.", total_num_annotations_skipped
    )
