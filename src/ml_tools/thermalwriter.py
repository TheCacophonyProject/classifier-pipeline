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

import hashlib
from dataclasses import dataclass, field

from absl import logging
import numpy as np

from . import tfrecord_util
from ml_tools.frame import TrackChannels
from ml_tools.imageprocessing import normalize
from ml_tools.rawdb import RawDatabase
from ml_tools.rectangle import Rectangle

THERMAL_MIN_KV = 27315
THERMAL_MAX_KV = 31515  # 42 celcius


@dataclass
class MeanData:
    """Holds per-channel border pixel lists (during collection) or mean values (after aggregation)."""

    thermal: float = 0
    filtered: float = 0
    thermal_norm: float = 0
    frames_used: int = 0

    def add_means(self, other):
        self.thermal = (
            other.thermal * other.frames_used + self.thermal * self.frames_used
        )
        self.filtered = (
            other.filtered * other.frames_used + self.filtered * self.frames_used
        )
        self.thermal_norm = (
            other.thermal_norm * other.frames_used
            + self.thermal_norm * self.frames_used
        )
        self.frames_used += other.frames_used
        if self.frames_used > 0:
            self.thermal /= self.frames_used
            self.filtered /= self.frames_used
            self.thermal_norm /= self.frames_used

    def __mul__(self, scalar):
        return MeanData(
            self.thermal * scalar, self.filtered * scalar, self.thermal_norm * scalar
        )

    def __truediv__(self, scalar):
        return MeanData(
            self.thermal / scalar, self.filtered / scalar, self.thermal_norm / scalar
        )

    def to_dict(self):
        return {
            TrackChannels.thermal.name: self.thermal,
            TrackChannels.filtered.name: self.filtered,
            TrackChannels.thermal_norm.name: self.thermal_norm,
        }

    def __len__(self):
        return self.frames_used


# @dataclass
# class BorderData:
#     """Holds per-channel border pixel lists (during collection) or mean values (after aggregation)."""

#     thermal: list = field(default_factory=list)
#     filtered: list = field(default_factory=list)
#     thermal_norm: list = field(default_factory=list)

#     def __len__(self):
#         return len(self.thermal)

#     def mean(self):
#         n = len(self)
#         return MeanData(
#              float(np.mean(self.thermal)) if n > 0 else 0.0,
#              float(np.mean(self.filtered)) if n > 0 else 0.0,
#              float(np.mean(self.thermal_norm)) if n > 0 else 0.0,
#             n,
#         )

#     # def __iadd__(self, other):
#     #     self.thermal += other.thermal
#     #     self.filtered += other.filtered
#     #     self.thermal_norm += other.thermal_norm
#     #     return self

#     # def __mul__(self, scalar):
#     #     return BorderData(self.thermal * scalar, self.filtered * scalar, self.thermal_norm * scalar)

#     # __rmul__ = __mul__

#     def __truediv__(self, scalar):
#         return BorderData(
#             self.thermal / scalar,
#             self.filtered / scalar,
#             self.thermal_norm / scalar,
#             frames_used=scalar,
#         )

#     def to_dict(self):
#         return {
#             TrackChannels.thermal.name: self.thermal,
#             TrackChannels.filtered.name: self.filtered,
#             TrackChannels.thermal_norm.name: self.thermal_norm,
#         }


crop_rectangle = Rectangle(0, 0, 640, 480)


def create_tf_example(sample, data, features, labels, country_code):
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
    # if we wanted can save writing memory by using alternative tf record writer
    import tensorflow as tf

    average_dim = [r.area for r in sample.track_bounds]
    average_dim = int(round(np.mean(average_dim) ** 0.5))
    thermal_raw, filtered, thermal_norm, frame_indices, roi, means = data
    if len(thermal_raw) == 0:
        return None
    image_id = sample.unique_id
    image_height, image_width = thermal_raw[0].shape
    # while len(thermal_raw) < num_frames:
    #     # ensure 25 frames even if 0s
    #     thermal_raw.append(np.zeros((thermal_raw[0].shape)))
    #     filtereds.append(np.zeros((filtereds[0].shape),dtype = np.uint8))
    #     thermal_norm.append(np.zeros((thermal_norm[0].shape),dtype=np.uint8))

    thermal_raw = np.array(thermal_raw)
    filtered = np.array(filtered)
    thermal_norm = np.array(thermal_norm)
    frame_indices = np.array(frame_indices)
    means = np.float32(means)

    roi = np.uint8(roi)
    thermal_key = hashlib.sha256(thermal_raw).hexdigest()
    filtered_key = hashlib.sha256(filtered).hexdigest()
    mask_key = hashlib.sha256(thermal_norm).hexdigest()

    avg_mass = int(round(sample.mass / len(sample.frame_numbers)))
    feature_dict = {
        "image/upsampled": tfrecord_util.int64_feature(sample.upsampled),
        "image/roi": tfrecord_util.bytes_feature(roi.ravel().tobytes()),
        "image/means": tfrecord_util.float_list_feature(means.ravel()),
        "image/frame_numbers": tfrecord_util.int64_list_feature(frame_indices),
        "image/filtered": tfrecord_util.int64_feature(1 if sample.filtered else 0),
        "image/avg_mass": tfrecord_util.int64_feature(avg_mass),
        "image/track_median_mass": tfrecord_util.int64_feature(
            int(sample.track_median_mass)
        ),
        "image/num_frames": tfrecord_util.int64_feature(len(filtered)),
        "image/avg_dim": tfrecord_util.int64_feature(average_dim),
        "image/height": tfrecord_util.int64_feature(image_height),
        "image/width": tfrecord_util.int64_feature(image_width),
        "image/clip_id": tfrecord_util.int64_feature(sample.clip_id),
        "image/track_id": tfrecord_util.int64_feature(sample.track_id),
        "image/filename": tfrecord_util.bytes_feature(
            str(sample.source_file).encode("utf8")
        ),
        "image/source_id": tfrecord_util.bytes_feature(str(image_id).encode("utf8")),
        "image/thermal_raw_encoded": tfrecord_util.float_list_feature(
            thermal_raw.ravel()
        ),
        "image/filtered_encoded": tfrecord_util.float_list_feature(filtered.ravel()),
        "image/thermal_norm_encoded": tfrecord_util.float_list_feature(
            thermal_norm.ravel()
        ),
        "image/filteredkey/sha256": tfrecord_util.bytes_feature(
            filtered_key.encode("utf8")
        ),
        "image/thermalkey/sha256": tfrecord_util.bytes_feature(
            thermal_key.encode("utf8")
        ),
        "image/maskkey/sha256": tfrecord_util.bytes_feature(mask_key.encode("utf8")),
        "image/format": tfrecord_util.bytes_feature("jpeg".encode("utf8")),
        "image/class/text": tfrecord_util.bytes_feature(sample.label.encode("utf8")),
        "image/class/label": tfrecord_util.int64_feature(labels.index(sample.label)),
        "image/country_id": tfrecord_util.bytes_feature(
            str(country_code).encode("utf8")
        ),
    }
    if features is not None:
        feature_dict["image/features"] = tfrecord_util.float_list_feature(
            features.ravel()
        )

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def save_data(source_file, excluded_tags, writer, labels, extra_args):
    sample_data = get_data(source_file, excluded_tags, extra_args)
    if sample_data is None:
        return 0
    saved = 0
    try:
        sample_data, country_code, border_data = sample_data

        for sample, images, features in sample_data:
            tf_example = create_tf_example(
                sample, images, features, labels, country_code
            )
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
                saved += 1
    except:
        logging.error("Could not save data for %s", source_file, exc_info=True)

    return (saved, border_data)


def get_data(source_file, excluded_tags, extra_args):
    from skimage import exposure
    import cv2
    import math
    from ml_tools.dataset import filter_track
    from ml_tools.preprocess import preprocess_frame_v2

    # prepare the sample data for saving
    ENLARGE_FOR_AUGMENT = True
    ADD_FEATURES = False

    BACKGROUND_THRESH = 150  # lepton3.5
    mosaic_dim = extra_args.get("mosaic_dim")
    border_pixels = MeanData()
    data = []
    crop_rectangle = Rectangle(1, 1, 160 - 2, 120 - 2)
    resize_dim = mosaic_dim
    if ENLARGE_FOR_AUGMENT:
        # allow extra pixels for augmentation
        resize_dim = int(math.floor(mosaic_dim * 1.41))
    if source_file.suffix == ".hdf5":
        from ml_tools.trackdatabase import TrackDatabase

        raise Exception("Need to implement min max filtered values for hdf5 track")

        db = TrackDatabase(clip_samples[0].source_file)
    else:
        db = RawDatabase(source_file)
        db.load_frames()
    # going to redo segments to get rid of ffc segments
    try:
        clip_meta = db.get_clip_meta(extra_args.get("tag_precedence"))
        frame_temp_median = clip_meta.frame_temp_median

        # group samples by track_id
        # samples_by_track = {}
        # for s in clip_samples:
        # samples_by_track.setdefault(s.track_id, []).append(s)

        clip_meta.tracks = [
            track
            for track in clip_meta.tracks
            if not filter_track(track, excluded_tags)
        ]
        for track in clip_meta.tracks:
            thermal_min = 0
            by_frame_number = {}
            thermal_max_diff = None
            thermal_min_diff = None
            max_diff = None
            min_diff = None
            thermal_diff_norm = extra_args.get("thermal_diff_norm", False)

            if extra_args.get("label_mapping") is not None:
                track.remapped_label = extra_args["label_mapping"].get(
                    track.original_label, track.original_label
                )

            # GP All assumes we dont have a track over multiple bins (Whcih we probably never want)
            if extra_args.get("use_segments", True):
                segment_types = extra_args.get("segment_types")
                # loading segments here again as this has access to ffc frames
                track.get_segments(
                    segment_width=extra_args.get("segment_width", 25),
                    segment_frame_spacing=extra_args.get("segment_frame_spacing", 9),
                    segment_types=segment_types,
                    segment_min_mass=extra_args.get("segment_min_avg_mass"),
                    dont_filter=extra_args.get("dont_filter_segment", False),
                    skip_ffc=extra_args.get("skip_ffc", True),
                    ffc_frames=clip_meta.ffc_frames,
                    max_segments=extra_args.get("max_segments"),
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

            # normalize by maximum difference between background and tracked region
            # probably only need to use difference on the frames used for this record
            # also min_diff maybe could just be set to 0 and clip values below 0,
            # these represent pixels whcih are cooler than the background

            by_frame_number = {}
            used_frames = []
            features = None
            for sample in samples:
                assert len(set(sample.frame_indices)) == len(
                    sample.frame_indices
                ), "Frame indices must be unique"
                thermalNorm = []
                thermalRaw = []  # np.empty(len(frames), dtype=object)
                filtered = []  # np.empty(len(frames), dtype=object)
                frame_indices = []
                roi = []
                means = []
                for frame_number in sample.frame_indices:
                    # no need to do work twice
                    if frame_number not in used_frames:
                        frame = db.frames[frame_number]
                        used_frames.append(frame_number)
                        region = track.regions_by_frame[frame_number]

                        result = preprocess_frame_v2(
                            frame, resize_dim, region, crop_rectangle,original_dim = mosaic_dim
                        )
                        if result is None:
                            by_frame_number[frame_number] = None
                            continue

                        cropped_frame, mean_value, data_region = result

                        if mean_value.frames_used == 0:
                            # probably doesn't matter to just ignore these clips
                            logging.error(
                                "%s Empty border for clip: %s track: %s frame %s  original %s",
                                mean_value,
                                clip_meta.clip_id,
                                track.track_id,
                                frame_number,
                                region,
                            )
                        else:
                            border_pixels.add_means(mean_value)

                        by_frame_number[frame_number] = cropped_frame
                    else:
                        cropped_frame, _ = by_frame_number[frame_number]

                    assert cropped_frame.thermal.shape == (
                        resize_dim,
                        resize_dim,
                    ), f"Shape is wrong {cropped_frame.region}"
                    # GP could handle each type separately, may be instances where one is valid
                    if (
                        cropped_frame is not None
                        and cropped_frame.filtered is not None
                        and cropped_frame.thermal is not None
                        and cropped_frame.thermal_norm is not None
                    ):
                        filtered.append(cropped_frame.filtered)
                        thermalRaw.append(cropped_frame.thermal)
                        thermalNorm.append(cropped_frame.thermal_norm)
                        roi.append(data_region)
                        means.append(
                            [
                                mean_value.thermal,
                                mean_value.filtered,
                                mean_value.thermal_norm,
                            ]
                        )
                        frame_indices.append(frame_number)
                thermalRaw = np.array(thermalRaw)
                filtered = np.array(filtered)
                thermalNorm = np.array(thermalNorm)
                roi = np.array(roi)

                data.append(
                    (
                        sample,
                        (thermalRaw, filtered, thermalNorm, frame_indices, roi, means),
                        features,
                    )
                )
    except:
        logging.error("Cant get Samples for %s", source_file, exc_info=True)
        return None
    return (data, clip_meta.country_code, border_pixels)


def pad_frame(frame, new_height, new_width, top, left, pad_values):
    """Pad channels to (new_height, new_width), placing existing content at (top, left)."""
    padded = np.full((new_height, new_width), pad_values[0], dtype=frame.thermal.dtype)
    h, w = frame.thermal.shape[:2]
    padded[top : top + h, left : left + w] = frame.thermal
    frame.thermal = padded

    padded = np.full(
        (new_height, new_width), pad_values[1], dtype=frame.thermal_norm.dtype
    )
    h, w = frame.thermal_norm.shape[:2]
    padded[top : top + h, left : left + w] = frame.thermal_norm
    frame.thermal_norm = padded

    padded = np.full((new_height, new_width), pad_values[2], dtype=frame.filtered.dtype)
    h, w = frame.filtered.shape[:2]
    padded[top : top + h, left : left + w] = frame.filtered
    frame.filtered = padded


def feature_stuff():
    # TODO if we ever want  random forest features needs to be sorted
    frame_temp_median = {}


# track_frames = []

# for frame_i in range(
#     track.start_frame, track.start_frame + track.num_frames
# ):
#     f = db.frames[frame_i]
#     region = track.regions_by_frame[frame_i]

#     if region.blank or region.width <= 0 or region.height <= 0:
#         continue
#     median_temp = np.median(f.thermal)
#     frame_temp_median[frame_i] = median_temp

# old way if we need to calculate min and max diff
# diff_frame = region.subimage(f.filtered)
# new_max = np.amax(diff_frame)
# new_min = np.amin(diff_frame)
# if min_diff is None or new_min < min_diff:
#     min_diff = new_min
#     # min_diff = max(0, new_min)
# if max_diff is None or new_max > max_diff:
#     max_diff = new_max
# if thermal_diff_norm:
#     # no benefit in doing for thermal
#     diff_frame = region.subimage(f.thermal) - median_temp
#     new_max = np.amax(diff_frame)
#     new_min = np.amin(diff_frame)
#     if thermal_min_diff is None or new_min < thermal_min_diff:
#         thermal_min_diff = new_min
#     if thermal_max_diff is None or new_max > thermal_max_diff:
#         thermal_max_diff = new_max

# if thermal_min == 0:
#     # check that we have nice values other wise allow negatives when normalizing
#     sub_thermal = region.subimage(f.thermal)
#     sub_thermal = np.float32(sub_thermal) - median_temp
#     if np.median(sub_thermal) <= 0:
#         thermal_min = None


#     enlarged_region = region.copy()
#     if ENLARGE_FOR_AUGMENT:
#         if region.width > resize_dim or region.height >resize_dim:
#             enlarged_region.enlarge_for_rotation(mosaic_dim, resize_dim - mosaic_dim)
#             logging.info("%s %s Region %s becomes %s",source_file,clip_meta.clip_id,region,enlarged_region)

#         else:
#             enlarged_region.enlarge_to(resize_dim)
#         # logging.info("Enlarging for augment %s %s",region, enlarged_region)


#     if not crop_rectangle.contains_rec(enlarged_region):
#         cropped = f.crop_by_region_with_padding(enlarged_region,crop_rectangle,resize_dim)
#     else:

#         cropped = f.crop_by_region(enlarged_region)
#     cropped.float_arrays()
#     track_frames.append(cropped)
#     by_frame_number[f.frame_number] = (cropped, median_temp)

# logging.debug("Saving %s samples %s", track.track_id, len(samples))
# used_frames = []
# features = None
# if ADD_FEATURES:
#     from ml_tools.forestmodel import forest_features

#     features, _, _ = forest_features(
#         track_frames,
#         db.get_clip_background(),
#         frame_temp_median,
#         [f.region for f in track_frames],
#         normalize=True,
#         cropped=True,
#     )
