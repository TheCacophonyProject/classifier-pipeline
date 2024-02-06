"""
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import datetime
import logging
import numpy as np
import os
import pytz
import cv2

from ml_tools.imageprocessing import normalize, detect_objects
from ml_tools.tools import Rectangle
from track.framebuffer import FrameBuffer
from track.track import Track
from track.region import Region
from piclassifier.cptvmotiondetector import is_affected_by_ffc

RES_X = 160
RES_Y = 120


class Clip:
    PREVIEW = "preview"
    FRAMES_PER_SECOND = 9
    local_tz = pytz.timezone("Pacific/Auckland")
    CLIP_ID = 1
    # used when calculating background, mininimum percentage the difference object
    # and background object must overlap i.e. they are a valid object
    MIN_ORIGIN_OVERLAP = 0.80

    def __init__(
        self,
        trackconfig,
        sourcefile,
        background=None,
        calc_stats=True,
        model=None,
        type="thermal",
        fps=FRAMES_PER_SECOND,
    ):
        self._id = Clip.CLIP_ID
        Clip.CLIP_ID += 1
        Track._track_id = 1
        self.disable_background_subtraction = False
        self.current_frame = -1
        self.ffc_affected = False
        self.crop_rectangle = None
        self.region_history = []
        self.active_tracks = set()
        self.tracks = []
        self.filtered_tracks = []
        self.from_metadata = False
        self.video_start_time = None
        self.location = None
        self.frame_buffer = None
        self.device = None
        self._background = None
        self.background_calculated = False
        self.res_x = None
        self.res_y = None
        self.background_frames = 0
        self.config = trackconfig
        self.frames_per_second = fps
        self.station_id = None
        self.calc_stats = calc_stats
        self.source_file = sourcefile
        self.stats = ClipStats()
        self.camera_model = None
        self.threshold_config = None
        self.track_min_delta = None
        self.track_max_delta = None
        self.background_thresh = None
        self.ffc_frames = []
        self.tags = None
        self.type = type
        # sets defaults
        self.set_model(model)
        if background is not None:
            self._background = background
            self._background_calculated()

        self.rescaled = None

    def get_frame(self, frame_number):
        return self.frame_buffer.get_frame(frame_number)

    def frames_kept(self):
        return self.frame_buffer.max_frames

    def rescaled_background(self, dims):
        if self.rescaled is not None:
            if self.rescaled[0] == self.current_frame:
                logging.info("Loading from cache")
                # 1 / 0
                return self.rescaled[1]
        resized = cv2.resize(
            self.background,
            (dims),
        )
        self.rescaled = (self.current_frame, resized)
        return resized

    @property
    def background(self):
        return self._background

    def set_model(self, camera_model):
        logging.debug("set model %s", camera_model)
        self.camera_model = camera_model
        threshold = self.config.motion.threshold_for_model(camera_model)
        if threshold:
            self.threshold_config = threshold
            self.set_motion_thresholds(threshold)

    def set_motion_thresholds(self, threshold):
        logging.debug("set thresholds %s", threshold)
        self.background_thresh = threshold.background_thresh
        self.temp_thresh = threshold.temp_thresh
        self.stats.threshold = self.background_thresh
        self.track_min_delta = threshold.track_min_delta
        self.track_max_delta = threshold.track_max_delta

    def _background_calculated(self):
        if self.type != "IR" or self.calc_stats:
            self.stats.mean_background_value = np.average(self._background)
            self.set_temp_thresh()
        self.background_calculated = True

    def on_preview(self):
        return not self.background_calculated

    def set_background(self, frame):
        self._background = frame
        self._background_calculated()

    def update_background(self, frame):
        """updates the clip _background"""
        if self._background is None:
            self._background = frame
        else:
            self._background = np.minimum(self._background, frame)
        self.background_frames += 1

    def calculate_initial_diff(self, frame, initial_frames, initial_diff):
        """Compare this frame with the initial frames (to detect movement) and update the initial_diff
        frame (such that it is the maximum difference). This is essentially creating a frame which has
        all the warms moving parts from the initial frame
        """
        if initial_frames is None:
            return np.zeros((frame.shape))
        else:
            diff = initial_frames - frame
            if initial_diff is not None:
                initial_diff = np.maximum(initial_diff, diff)
            else:
                initial_diff = diff
        return initial_diff

    def calculate_background(self, frame_reader):
        """
        Calculate background by reading whole clip and grouping into sets of
        9 frames. Take the average of these 9 frames and use the minimum
        over the sets as the initial background
        Also check for animals in the background by checking for connected components in
        the intital_diff frame - this is the maximum change between first average frame and all other average frames in the clip
        """
        frames = []
        if frame_reader.background_frames > 0:
            for frame in frame_reader:
                if frame.background_frame:
                    frames.append(frame.pix)
                else:
                    break
            frame_average = np.average(frames, axis=0)
            self.update_background(frame_average)
            self._background_calculated()
            return

        initial_frames = None
        initial_diff = None
        first_frame = None
        for frame in frame_reader:
            if first_frame is None:
                first_frame = frame.pix
            ffc_affected = is_affected_by_ffc(frame)
            if ffc_affected:
                continue
            frames.append(frame.pix)
            if len(frames) == 9:
                frame_average = np.average(frames, axis=0)
                self.update_background(frame_average)
                initial_diff = self.calculate_initial_diff(
                    frame_average, initial_frames, initial_diff
                )
                if initial_frames is None:
                    initial_frames = frame_average

                frames = []

        if len(frames) > 0:
            frame_average = np.average(frames, axis=0)
            if initial_frames is None:
                initial_frames = frame_average
            self.update_background(frame_average)
            initial_diff = self.calculate_initial_diff(
                frame_average, initial_frames, initial_diff
            )

            if initial_frames is None:
                initial_frames = frame_average
        frames = []
        if initial_diff is None:
            if first_frame is not None:
                # fall back if whole clip is ffc
                self.update_background(frame.pix)
                self._background_calculated()
            return
        np.clip(initial_diff, 0, None, out=initial_diff)
        initial_frames = self.remove_background_animals(initial_frames, initial_diff)

        self.update_background(initial_frames)
        self._background_calculated()

    def remove_background_animals(self, initial_frame, initial_diff):
        """
        Try and remove animals that are already in the initial frames, by
        checking for connected components in the intital_diff frame
        (this is the maximum change between first frame and all other frames in the clip)
        """
        # remove some noise
        initial_diff[initial_diff < self.background_thresh] = 0
        initial_diff[initial_diff > 255] = 255
        initial_diff = np.uint8(initial_diff)
        initial_diff = cv2.fastNlMeansDenoising(initial_diff, None)

        _, lower_mask, lower_objects, centroids = detect_objects(
            initial_diff, otsus=True
        )

        max_region = Rectangle(0, 0, self.res_x, self.res_y)
        for component, centroid in zip(lower_objects[1:], centroids[1:]):
            region = Region(
                component[0],
                component[1],
                component[2],
                component[3],
                centroid=centroid,
            )
            region.enlarge(2, max=max_region)
            if region.width >= self.res_x or region.height >= self.res_y:
                logging.info(
                    "Background animal bigger than max, probably false positive %s %s",
                    region,
                    component[4],
                )
                continue
            background_region = region.subimage(initial_frame)
            norm_back = background_region.copy()
            norm_back, _ = normalize(norm_back, new_max=255)
            sub_components, sub_connected, sub_stats, centroids = detect_objects(
                norm_back, otsus=True
            )

            if sub_components <= 1:
                continue
            overlap_image = region.subimage(lower_mask) * 255
            overlap_pixels = np.sum(sub_connected[overlap_image > 0])
            overlap_pixels = overlap_pixels / float(component[4])

            # filter out components which are too big, or dont match original causes
            # for filtering
            if (
                overlap_pixels < Clip.MIN_ORIGIN_OVERLAP
                or sub_stats[1][4] == 0
                or sub_stats[1][4] == region.area
            ):
                logging.info(
                    "Invalid components mass: %s, components: %s region area %s overlap %s",
                    sub_stats[1][4],
                    sub_components,
                    region.area,
                    overlap_pixels,
                )
                continue

            sub_connected[sub_connected > 0] = 1
            # remove this component from the background by painting with
            # colours of neighbouring pixels
            background_region[:] = cv2.inpaint(
                np.float32(background_region),
                np.uint8(sub_connected),
                3,
                cv2.INPAINT_TELEA,
            )
        return initial_frame

    def _add_active_track(self, track):
        self.active_tracks.add(track)
        self.tracks.append(track)

    def get_id(self):
        return str(self._id)

    def set_temp_thresh(self):
        if self.config.motion.dynamic_thresh:
            min_temp = self.threshold_config.min_temp_thresh
            max_temp = self.threshold_config.max_temp_thresh
            if max_temp:
                self.temp_thresh = min(max_temp, self.stats.mean_background_value)
            else:
                self.temp_thresh = self.stats.mean_background_value
            if min_temp:
                self.temp_thresh = max(min_temp, self.temp_thresh)
            self.stats.temp_thresh = self.temp_thresh
        else:
            self.temp_thresh = self.config.motion.temp_thresh

    def set_video_stats(self, video_start_time):
        """
        Extracts useful statics from video clip.
        """
        self.video_start_time = video_start_time
        self.stats.date_time = video_start_time.astimezone(Clip.local_tz)
        self.stats.is_night = (
            video_start_time.astimezone(Clip.local_tz).time().hour >= 2
        )

    def load_metadata(self, metadata, tag_precedence=None):
        self._id = metadata.get("id", 0)
        device_meta = metadata.get("Device")
        self.tags = metadata.get("Tags")

        if device_meta:
            self.device = device_meta.get("devicename")
        else:
            self.device = os.path.splitext(os.path.basename(self.source_file))[0].split(
                "-"
            )[-1]
        self.location = metadata.get("location")
        self.station_id = metadata.get("stationId")
        tracks = self.load_tracks_meta(metadata, tag_precedence)
        self.from_metadata = True
        self.tracks = set(tracks)

    def load_tracks_meta(self, metadata, tag_precedence):
        if "Tracks" in metadata:
            tracks_meta = metadata.get("Tracks", [])
        else:
            tracks_meta = metadata.get("tracks", [])
        tracks = []
        # get track data
        for track_meta in tracks_meta:
            track = Track(self.get_id())
            if track.load_track_meta(
                track_meta,
                self.frames_per_second,
                tag_precedence,
                self.config.min_tag_confidence,
            ):
                tracks.append(track)
        return tracks

    def start_and_end_in_secs(self, track):
        if track.end_s is None:
            track.end_s = (track.end_frame + 1) / self.frames_per_second

        return (track.start_s, track.end_s)

    def set_frame_buffer(
        self, high_quality_flow, cache_to_disk, use_flow, keep_frames, max_frames=None
    ):
        self.frame_buffer = FrameBuffer(
            self.source_file,
            high_quality_flow,
            cache_to_disk,
            use_flow,
            keep_frames,
            max_frames,
        )

    def set_res(self, res_x, res_y):
        if res_x == 0 or res_x == None:
            self.res_x = RES_X
        else:
            self.res_x = res_x
        if res_y == 0 or res_y == None:
            self.res_y = RES_Y
        else:
            self.res_y = res_y
        self._set_crop_rectangle()
        for track in self.tracks:
            track.crop_rectangle = self.crop_rectangle

    def _set_crop_rectangle(self):
        edge = self.config.edge_pixels
        self.crop_rectangle = Rectangle(
            edge, edge, self.res_x - 2 * edge, self.res_y - 2 * edge
        )

    def start_and_end_time_absolute(self, start_s=0, end_s=None):
        if not end_s:
            end_s = len(self.frame_buffer.frames) / self.frames_per_second
        return (
            self.video_start_time + datetime.timedelta(seconds=start_s),
            self.video_start_time + datetime.timedelta(seconds=end_s),
        )

    def print_if_verbose(self, info_string):
        if self.config.verbose:
            logging.info(info_string)

    def add_frame(self, thermal, filtered, mask=None, ffc_affected=False):
        self.current_frame += 1
        if ffc_affected:
            self.ffc_frames.append(self.current_frame)

        f = self.frame_buffer.add_frame(
            thermal, filtered, mask, self.current_frame, ffc_affected
        )

        if self.calc_stats:
            self.stats.add_frame(thermal, filtered)
        return f

    def get_metadata(self, predictions_per_model=None):
        meta_data = {}
        if self.camera_model:
            meta_data["camera_model"] = self.camera_model
        meta_data["background_thresh"] = self.background_thresh
        start, end = self.start_and_end_time_absolute()
        meta_data["start_time"] = start.isoformat()
        meta_data["end_time"] = end.isoformat()

        tracks = []
        for track in self.tracks:
            track_info = track.get_metadata(predictions_per_model)
            tracks.append(track_info)
        meta_data["tracks"] = tracks

        return meta_data


class ClipStats:
    """Stores background analysis statistics."""

    def __init__(self):
        self.mean_background_value = 0
        self.max_temp = None
        self.min_temp = None
        self.mean_temp = None
        self.frame_stats_min = []
        self.frame_stats_max = []
        self.frame_stats_median = []
        self.frame_stats_mean = []
        self.filtered_deviation = None
        self.filtered_sum = 0
        self.temp_thresh = 0
        self.threshold = None
        self.average_delta = None
        self.is_static_background = None

    def add_frame(self, thermal, filtered=None):
        f_median = np.median(thermal)
        f_max = np.max(thermal)
        f_min = np.min(thermal)
        f_mean = np.nanmean(thermal)
        self.max_temp = null_safe_compare(self.max_temp, f_max, max)
        self.min_temp = null_safe_compare(self.min_temp, f_min, min)

        self.frame_stats_min.append(f_min)
        self.frame_stats_max.append(f_max)
        self.frame_stats_median.append(f_median)
        self.frame_stats_mean.append(f_mean)
        if filtered is not None:
            self.filtered_sum += np.sum(np.abs(filtered))

    def completed(self):
        if self.filtered_sum is not None:
            self.filtered_deviation = np.mean(np.uint16(self.filtered_sum))
        self.mean_temp = np.mean(np.uint16(self.frame_stats_mean))


def null_safe_compare(a, b, cmp_fn):
    if a is None:
        return b
    elif b is not None:
        return cmp_fn(a, b)
    else:
        return None
