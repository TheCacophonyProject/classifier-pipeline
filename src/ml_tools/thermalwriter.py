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
import time
import collections
import hashlib
import io
import json
import os
from multiprocessing import Process, Queue

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
from ml_tools.trackdatabase import TrackDatabase
from ml_tools.rawdb import RawDatabase

crop_rectangle = tools.Rectangle(0, 0, 640, 480)
from functools import lru_cache


def create_tf_example(sample, data, features, labels, num_frames):
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
    average_dim = [r.area for r in sample.track_bounds]
    average_dim = int(round(np.mean(average_dim) ** 0.5))
    thermals = list(data[0])
    filtereds = list(data[1])
    image_id = sample.unique_track_id
    image_height, image_width = thermals[0].shape
    while len(thermals) < num_frames:
        # ensure 25 frames even if 0s
        thermals.append(np.zeros((thermals[0].shape)))
        filtereds.append(np.zeros((filtereds[0].shape)))
    thermals = np.array(thermals)
    filtereds = np.array(filtereds)
    thermal_key = hashlib.sha256(thermals).hexdigest()
    filtered_key = hashlib.sha256(filtereds).hexdigest()
    avg_mass = int(round(sample.mass / len(sample.frame_numbers)))
    feature_dict = {
        "image/filtered": tfrecord_util.int64_feature(1 if sample.filtered else 0),
        "image/avg_mass": tfrecord_util.int64_feature(avg_mass),
        "image/track_median_mass": tfrecord_util.int64_feature(
            int(sample.track_median_mass)
        ),
        "image/avg_dim": tfrecord_util.int64_feature(average_dim),
        "image/height": tfrecord_util.int64_feature(image_height),
        "image/width": tfrecord_util.int64_feature(image_width),
        "image/clip_id": tfrecord_util.int64_feature(sample.clip_id),
        "image/track_id": tfrecord_util.int64_feature(sample.track_id),
        "image/filename": tfrecord_util.bytes_feature(
            str(sample.source_file).encode("utf8")
        ),
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
    return example


# @lru_cache(maxsize=1000)
# only going to get called once now per track
def get_track_data(clip_id, track_id, db):
    background = db.get_clip_background(clip_id)

    track_frames = db.get_track(
        clip_id, track_id, channels=[TrackChannels.thermal], crop=True
    )
    clip_meta = db.get_clip_meta(clip_id)
    frame_temp_median = clip_meta["frame_temp_median"]
    regions = [f.region for f in track_frames]
    features = forest_features(
        track_frames, background, frame_temp_median, regions, normalize=True
    )
    return background, track_frames, features, frame_temp_median


def save_data(samples, writer, labels, extra_args):
    sample_data = get_data(samples, extra_args)
    if sample_data is None:
        return 0
    saved = 0
    try:
        for data in sample_data:
            tf_example = create_tf_example(
                data[0], data[1], data[2], labels, extra_args["num_frames"]
            )
            writer.write(tf_example.SerializeToString())
            saved += 1
    except:
        logging.error(
            "Could not save data for %s", samples[0].source_file, exc_info=True
        )
    return saved


def get_data(clip_samples, extra_args):
    # prepare the sample data for saving
    if len(clip_samples) == 0:
        return None
    data = []
    crop_rectangle = tools.Rectangle(2, 2, 160 - 2 * 2, 140 - 2 * 2)
    if clip_samples[0].source_file.suffix == ".hdf5":
        db = TrackDatabase(clip_samples[0].source_file)
    else:
        db = RawDatabase(clip_samples[0].source_file)
        db.load_frames()
        # going to redo segments to get rid of ffc segments

    clip_id = clip_samples[0].clip_id
    try:
        background = db.get_clip_background()
        if background is None:
            frame_data = db.get_frames()
            background = np.median(frame_data, axis=0)
            del frame_data
        clip_meta = db.get_clip_meta(extra_args.get("tag_precedence"))
        frame_temp_median = clip_meta.frame_temp_median

        # group samples by track_id
        samples_by_track = {}
        for s in clip_samples:
            samples_by_track.setdefault(s.track_id, []).append(s)

        for track_id in samples_by_track.keys():
            samples = samples_by_track[track_id]
            if clip_samples[0].source_file.suffix != ".hdf5":
                track = next(
                    (track for track in clip_meta.tracks if track.track_id == track_id),
                    None,
                )
                if extra_args.get("label_mapping") is not None:
                    track.remapped_label = extra_args["label_mapping"].get(
                        track.original_label, track.original_label
                    )
                if track is None:
                    logging.error(
                        "Cannot find track %s in clip %s", track_id, clip_meta.clip_id
                    )
                    continue
                # GP All assumes we dont have a track over multiple bins (Whcih we probably never want)
                if extra_args.get("use_segments", True):
                    track.get_segments(
                        extra_args.get("segment_frame_spacing", 9),
                        extra_args.get("segment_width", 25),
                        extra_args.get("segment_type"),
                        extra_args.get("segment_min_avg_mass"),
                        max_segments=extra_args.get("max_segments"),
                        dont_filter=extra_args.get("dont_filter_segment", False),
                        skip_ffc=extra_args.get("skip_ffc", True),
                        ffc_frames=clip_meta.ffc_frames,
                    )
                else:
                    filter_by_lq = extra_args.get("filter_by_lq", False)
                    track.calculate_sample_frames(
                        min_mass=(
                            extra_args.get("min_mass")
                            if not filter_by_lq
                            else track.lower_mass
                        ),
                        max_mass=(
                            extra_args.get("max_mass")
                            if not filter_by_lq
                            else track.upper_mass
                        ),
                        ffc_frames=clip_meta.ffc_frames,
                    )
                samples = track.samples
                frame_temp_median = {}
                track_frames = []
                for frame_i in range(
                    track.start_frame, track.start_frame + track.num_frames
                ):
                    f = db.frames[frame_i]
                    region = track.regions_by_frame[frame_i]
                    frame_temp_median[frame_i] = np.median(f.thermal)
                    cropped = f.crop_by_region(region)
                    cropped.float_arrays()
                    track_frames.append(cropped)

            else:
                track_frames = db.get_track(
                    clip_id, track_id, channels=[TrackChannels.thermal], crop=True
                )
            logging.debug("Saving %s samples %s", track_id, len(samples))
            used_frames = []

            features = forest_features(
                track_frames,
                background,
                frame_temp_median,
                [f.region for f in track_frames],
                normalize=True,
                cropped=True,
            )

            by_frame_number = {}
            max_diff = 0
            min_diff = 0
            for f in track_frames:
                if f.region.blank or f.region.width <= 0 or f.region.height <= 0:
                    continue

                by_frame_number[f.frame_number] = (f, frame_temp_median[f.frame_number])
                f.float_arrays()
                diff_frame = f.thermal - f.region.subimage(background)
                new_max = np.amax(diff_frame)
                new_min = np.amin(diff_frame)
                if new_min < min_diff:
                    min_diff = new_min
                if new_max > max_diff:
                    max_diff = new_max

            # normalize by maximum difference between background and tracked region
            # probably only need to use difference on the frames used for this record
            # also min_diff maybe could just be set to 0 and clip values below 0,
            # these represent pixels whcih are cooler than the background
            for sample in samples:
                thermals = []  # np.empty(len(frames), dtype=object)
                filtered = []  # np.empty(len(frames), dtype=object)
                for frame_number in sample.frame_indices:
                    frame, temp_median = by_frame_number[frame_number]
                    # no need to do work twice
                    if frame_number not in used_frames:
                        used_frames.append(frame_number)
                        frame.filtered = frame.thermal - frame.region.subimage(
                            background
                        )
                        frame.resize_with_aspect(
                            (32, 32), crop_rectangle, keep_edge=True
                        )
                        frame.thermal -= temp_median
                        np.clip(frame.thermal, a_min=0, a_max=None, out=frame.thermal)

                        frame.thermal, stats = imageprocessing.normalize(
                            frame.thermal, new_max=255
                        )
                        if not stats[0]:
                            frame.thermal = np.zeros((frame.thermal.shape))
                            # continue
                        # f2 = frame.filtered.copy()
                        # frame.filtered, stats = imageprocessing.normalize(
                        #     frame.filtered, new_max=255
                        # )
                        # np.clip(frame.filtered, a_min=min_diff, a_max=None, out=frame.filtered)

                        frame.filtered, stats = imageprocessing.normalize(
                            frame.filtered, min=min_diff, max=max_diff, new_max=255
                        )

                        if not stats[0]:
                            frame.filtered = np.zeros((frame.filtered.shape))
                        f2 = np.uint8(frame.filtered)
                    filtered.append(frame.filtered)
                    thermals.append(frame.thermal)

                thermals = np.array(thermals)
                filtered = np.array(filtered)
                data.append((sample, (thermals, filtered), features))
    except:
        logging.error(
            "Cant get Samples for %s", clip_samples[0].source_file, exc_info=True
        )
        return None
    return data
