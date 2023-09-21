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


def process_job(queue, labels, base_dir, num_frames):
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
                saved += save_data(
                    samples,
                    writer,
                    labels,
                    num_frames,
                )
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


def create_tf_records(
    dataset,
    output_path,
    labels,
    back_thresh,
    num_shards=1,
    cropped=True,
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
    total_num_annotations_skipped = 0
    num_labels = len(labels)
    # pool = multiprocessing.Pool(4)
    logging.info(
        "writing to output path: %s for %s samples",
        output_path,
        len(dataset.samples_by_id),
    )
    lbl_counts = [0] * num_labels
    # lbl_counts[l] = 0
    logging.info("labels are %s", labels)

    num_processes = 8
    try:
        job_queue = Queue()
        processes = []
        for i in range(num_processes):
            p = Process(
                target=process_job,
                args=(
                    job_queue,
                    labels,
                    output_path,
                    dataset.segment_length,
                ),
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

    logging.info(
        "Finished writing, skipped %d annotations.", total_num_annotations_skipped
    )


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


def save_data(samples, writer, labels, num_frames):
    sample_data = get_data(samples)
    if sample_data is None:
        return 0
    saved = 0
    try:
        for data in sample_data:
            tf_example = create_tf_example(
                data[0], data[1], data[2], labels, num_frames
            )
            writer.write(tf_example.SerializeToString())
            saved += 1
    except:
        logging.error(
            "Could not save data for %s", samples[0].source_file, exc_info=True
        )
    return saved


def get_data(clip_samples):
    # prepare the sample data for saving
    if len(clip_samples) == 0:
        return None
    data = []
    crop_rectangle = tools.Rectangle(2, 2, 160 - 2 * 2, 140 - 2 * 2)
    db = TrackDatabase(clip_samples[0].source_file)
    clip_id = clip_samples[0].clip_id
    try:
        background = db.get_clip_background(clip_id)
        if background is None:
            frame_data = db.get_clip(clip_id)
            background = np.median(frame_data, axis=0)
            del frame_data
        clip_meta = db.get_clip_meta()
        frame_temp_median = clip_meta.frame_temp_median

        # group samples by track_id
        samples_by_track = {}
        for s in clip_samples:
            samples_by_track.setdefault(s.track_id, []).append(s)

        for track_id, samples in samples_by_track.items():
            logging.debug("Saving %s samples %s", track_id, len(samples))
            used_frames = []
            track_frames = db.get_track(
                clip_id, track_id, channels=[TrackChannels.thermal], crop=True
            )
            features = forest_features(
                track_frames,
                background,
                frame_temp_median,
                [f.region for f in track_frames],
                normalize=True,
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
