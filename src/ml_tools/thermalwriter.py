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
import cv2
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
from ml_tools.rectangle import Rectangle
from ml_tools.datasetstructures import SegmentType

crop_rectangle = Rectangle(0, 0, 640, 480)


def create_tf_example(sample, data, features, labels, num_frames, country_code):
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
    image_id = sample.unique_id
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
        "image/features": tfrecord_util.float_list_feature(features.ravel()),
        "image/filteredkey/sha256": tfrecord_util.bytes_feature(
            filtered_key.encode("utf8")
        ),
        "image/thermalkey/sha256": tfrecord_util.bytes_feature(
            thermal_key.encode("utf8")
        ),
        "image/format": tfrecord_util.bytes_feature("jpeg".encode("utf8")),
        "image/class/text": tfrecord_util.bytes_feature(sample.label.encode("utf8")),
        "image/class/label": tfrecord_util.int64_feature(labels.index(sample.label)),
        "image/country_id": tfrecord_util.bytes_feature(
            str(country_code).encode("utf8")
        ),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def save_data(samples, writer, labels, extra_args):
    sample_data = get_data(samples, extra_args)
    if sample_data is None:
        return 0
    saved = 0
    try:
        country_code = sample_data[1]
        sample_data = sample_data[0]
        for sample, images, features in sample_data:
            tf_example = create_tf_example(
                sample, images, features, labels, extra_args["num_frames"], country_code
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
    ENLARGE_FOR_AUGMENT = True
    if len(clip_samples) == 0:
        return None
    data = []
    crop_rectangle = tools.Rectangle(1, 1, 160 - 2, 120 - 2)
    resize_dim = 32
    if ENLARGE_FOR_AUGMENT:
        # allow extra pixels for augmentation
        resize_dim = 45
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
            thermal_min = 0
            by_frame_number = {}
            thermal_max_diff = None
            thermal_min_diff = None
            max_diff = None
            min_diff = None
            thermal_diff_norm = extra_args.get("thermal_diff_norm", False)
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
                    segment_types = extra_args.get("segment_types")
                    track.get_segments(
                        segment_width=extra_args.get("segment_width", 25),
                        segment_frame_spacing=extra_args.get(
                            "segment_frame_spacing", 9
                        ),
                        segment_types=segment_types,
                        segment_min_mass=extra_args.get("segment_min_avg_mass"),
                        dont_filter=extra_args.get("dont_filter_segment", False),
                        skip_ffc=extra_args.get("skip_ffc", True),
                        ffc_frames=clip_meta.ffc_frames,
                        max_segments=len(samples),
                        frame_min_mass=extra_args.get("min_mass"),
                        filter_by_fp=extra_args.get("filter_by_fp"),
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
                        max_frames=extra_args.get("max_frames"),
                    )
                samples = track.samples
                frame_temp_median = {}
                track_frames = []

                for frame_i in range(
                    track.start_frame, track.start_frame + track.num_frames
                ):
                    f = db.frames[frame_i]
                    region = track.regions_by_frame[frame_i]

                    if region.blank or region.width <= 0 or region.height <= 0:
                        continue
                    median_temp = np.median(f.thermal)
                    frame_temp_median[frame_i] = median_temp

                    diff_frame = region.subimage(f.filtered)
                    new_max = np.amax(diff_frame)
                    new_min = np.amin(diff_frame)
                    if min_diff is None or new_min < min_diff:
                        min_diff = new_min
                        # min_diff = max(0, new_min)
                    if max_diff is None or new_max > max_diff:
                        max_diff = new_max
                    if thermal_diff_norm:
                        # no benefit in doing for thermal is better
                        diff_frame = region.subimage(f.thermal) - median_temp
                        new_max = np.amax(diff_frame)
                        new_min = np.amin(diff_frame)
                        if thermal_min_diff is None or new_min < thermal_min_diff:
                            thermal_min_diff = new_min
                        if thermal_max_diff is None or new_max > thermal_max_diff:
                            thermal_max_diff = new_max

                    if thermal_min == 0:
                        # check that we have nice values other wise allow negatives when normalizing
                        sub_thermal = region.subimage(f.thermal)
                        sub_thermal = np.float32(sub_thermal) - median_temp
                        if np.median(sub_thermal) <= 0:
                            thermal_min = None

                    enlarged_region = region.copy()
                    if ENLARGE_FOR_AUGMENT:
                        enlarged_region.enlarge_for_rotation(crop_rectangle)
                    cropped = f.crop_by_region(enlarged_region)
                    cropped.float_arrays()
                    track_frames.append(cropped)
                    by_frame_number[f.frame_number] = (cropped, median_temp)

            else:
                raise Exception(
                    "Need to implement min max filtered values for hdf5 track"
                )
                track_frames = db.get_track(
                    clip_id, track_id, channels=[TrackChannels.thermal], crop=True
                )

            logging.debug("Saving %s samples %s", track_id, len(samples))
            used_frames = []

            features, _, _ = forest_features(
                track_frames,
                background,
                frame_temp_median,
                [f.region for f in track_frames],
                normalize=True,
                cropped=True,
            )

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
                        # frame.filtered = frame.thermal - frame.region.subimage(
                        #     background
                        # )
                        # print("Reiszie fram e",frame_number,track_id)
                        region = track.regions_by_frame[frame_number]

                        frame.resize_with_aspect(
                            (resize_dim, resize_dim),
                            crop_rectangle,
                            keep_edge=True,
                            edge_offset=(7, 7, 6, 6),
                            original_region=region,
                        )
                        if (
                            np.amax(frame.thermal) > 50000
                            or np.amin(frame.thermal) < 1000
                        ):
                            logging.error(
                                "Strange values for %s max %s min %s",
                                clip_id,
                                np.amax(frame.thermal),
                                np.amin(frame.thermal),
                            )
                            raise Exception(
                                f"Strange values for {clip_id} - {track_id} #{frame_number}"
                            )
                        frame.thermal -= temp_median
                        if not thermal_diff_norm and thermal_min == 0:
                            np.clip(
                                frame.thermal, a_min=0, a_max=None, out=frame.thermal
                            )

                        frame.thermal, stats = imageprocessing.normalize(
                            frame.thermal,
                            min=thermal_min_diff,
                            max=thermal_max_diff,
                            new_max=255,
                        )
                        if not stats[0]:
                            frame.thermal = np.zeros((frame.thermal.shape))

                        frame.filtered, stats = imageprocessing.normalize(
                            frame.filtered, min=min_diff, max=max_diff, new_max=255
                        )

                        np.clip(frame.filtered, a_min=0, a_max=255, out=frame.filtered)

                        if not stats[0]:
                            frame.filtered = np.zeros((frame.filtered.shape))
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
    return (data, clip_meta.country_code)
