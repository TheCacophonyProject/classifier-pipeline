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

import tensorflow as tf
from . import tfrecord_util
from ml_tools import tools
from ml_tools.imageprocessing import normalize
from ml_tools.forestmodel import forest_features
from ml_tools import imageprocessing
from ml_tools.frame import TrackChannels

crop_rectangle = tools.Rectangle(0, 0, 640, 480)
from functools import lru_cache


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
    features = data[1]
    data = data[0]
    thermals = list(data[0])
    filtereds = list(data[1])
    image_id = sample.unique_track_id
    image_height, image_width = thermals[0].shape
    while len(thermals) < 25:
        # ensure 25 frames even if 0s
        thermals.append(np.zeros((thermals[0].shape)))
        filtereds.append(np.zeros((filtereds[0].shape)))
    thermals = np.array(thermals)
    filtereds = np.array(filtereds)
    thermal_key = hashlib.sha256(thermals).hexdigest()
    filtered_key = hashlib.sha256(filtereds).hexdigest()

    feature_dict = {
        "image/avg_dim": tfrecord_util.int64_feature(average_dim),
        "image/height": tfrecord_util.int64_feature(image_height),
        "image/width": tfrecord_util.int64_feature(image_width),
        "image/clip_id": tfrecord_util.int64_feature(sample.clip_id),
        "image/track_id": tfrecord_util.int64_feature(sample.track_id),
        "image/filename": tfrecord_util.bytes_feature(filename.encode("utf8")),
        "image/source_id": tfrecord_util.bytes_feature(str(image_id).encode("utf8")),
        "image/thermalencoded": tfrecord_util.float_list_feature(thermals.ravel()),
        "image/filteredencoded": tfrecord_util.float_list_feature(filtereds.ravel()),
        "image/features": tfrecord_util.float_list_feature(features),
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


def create_tf_records(
    dataset, output_path, labels, back_thresh, num_shards=1, cropped=True, by_label=True
):

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
    dataset.set_read_only(True)
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
            if by_label:
                safe_l = label.replace("/", "-")
                name = f"{safe_l}-%05d-of-%05d.tfrecord" % (i, num_shards)
            else:
                name = f"%05d-of-%05d.tfrecord" % (i, num_shards)
            writers.append(tf.io.TFRecordWriter(str(output_path / name)))
        if not by_label:
            break
    load_first = 200
    try:
        count = 0
        while len(samples) > 0:
            local_set = samples[:load_first]
            samples = samples[load_first:]
            loaded = []
            start = time.time()
            for sample in local_set:
                data, features = get_data(sample, db)
                # features = forest_features(db, sample.clip_id, sample.track_id)
                if data is None:
                    continue

                loaded.append(((data, features), sample))
            loaded = np.array(loaded, dtype=object)
            np.random.shuffle(loaded)

            for data, sample in loaded:
                try:
                    tf_example, num_annotations_skipped = create_tf_example(
                        data, output_path, sample, labels, ""
                    )
                    total_num_annotations_skipped += num_annotations_skipped
                    l_i = labels.index(sample.label)
                    if by_label:
                        wrtier = writers[
                            num_shards * l_i + lbl_counts[l_i] % num_shards
                        ]
                    else:
                        writer = writers[count % num_shards]
                    writer.write(tf_example.SerializeToString())
                    lbl_counts[l_i] += 1
                    # print("saving example", [count % num_shards])
                    count += 1
                    if count % 100 == 0:
                        logging.info("saved %s", count)
                    # count += 1
                except Exception as e:
                    logging.error("Error saving ", exc_info=True)

    except:
        logging.error("Error saving track info", exc_info=True)
    for writer in writers:
        writer.close()

    logging.info(
        "Finished writing, skipped %d annotations.", total_num_annotations_skipped
    )


@lru_cache(maxsize=10000)
def get_track_data(clip_id, track_id, db):
    background = db.get_clip_background(clip_id)

    track_frames = db.get_track(
        clip_id,
        track_id,
        original=False,
        channels=[TrackChannels.thermal],
    )
    clip_meta = db.get_clip_meta(clip_id)
    frame_temp_median = clip_meta["frame_temp_median"]
    regions = [f.region for f in track_frames]
    features = forest_features(track_frames, background, frame_temp_median, regions)
    return background, track_frames, features, frame_temp_median


def get_data(sample, db):

    # prepare the sample data for saving
    crop_rectangle = tools.Rectangle(2, 2, 160 - 2 * 2, 140 - 2 * 2)
    try:
        background, track_frames, features, frame_temp_median = get_track_data(
            sample.clip_id, sample.track_id, db
        )

        thermals = []  # np.empty(len(frames), dtype=object)
        filtered = []  # np.empty(len(frames), dtype=object)
        for i, frame in enumerate(track_frames):
            if frame.frame_number not in sample.frame_numbers:
                continue
            frame.float_arrays()
            frame.filtered = frame.thermal - frame.region.subimage(background)
            temp = frame_temp_median[i]
            frame.resize_with_aspect((32, 32), crop_rectangle, keep_edge=True)
            frame.thermal -= temp
            np.clip(frame.thermal, a_min=0, a_max=None, out=frame.thermal)

            frame.thermal, stats = imageprocessing.normalize(frame.thermal, new_max=255)
            if not stats[0]:
                frame.thermal = np.zeros((frame.thermal.shape))
                # continue
            frame.filtered, stats = imageprocessing.normalize(
                frame.filtered, new_max=255
            )
            if not stats[0]:
                frame.filtered = np.zeros((frame.filtered.shape))

            filtered.append(frame.filtered)
            thermals.append(frame.thermal)
        thermals = np.array(thermals)
        filtered = np.array(filtered)
    except:
        logging.error("Cant get segment %s", sample, exc_info=True)
        return None
    return (thermals, filtered), features
