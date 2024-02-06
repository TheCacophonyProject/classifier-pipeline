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

import logging
import numpy as np
import time
import yaml

from cptv import CPTVReader
import cv2

from .clip import Clip
from ml_tools.tools import Rectangle
from track.region import Region
from track.track import Track
from piclassifier.cptvmotiondetector import is_affected_by_ffc
from ml_tools.imageprocessing import detect_objects, normalize
from track.cliptracker import ClipTracker


class ClipTrackExtractor(ClipTracker):
    PREVIEW = "preview"
    VERSION = 10
    TYPE = "thermal"

    @property
    def tracker_version(self):
        return ClipTrackExtractor.VERSION
        # until api takes a string
        # return f"ClipTrackExtractor-{ClipTrackExtractor.VERSION}"

    @property
    def type(self):
        return ClipTrackExtractor.TYPE

    def __init__(
        self,
        config,
        use_opt_flow,
        cache_to_disk=False,
        keep_frames=True,
        calc_stats=True,
        high_quality_optical_flow=False,
        verbose=False,
        do_tracking=True,
    ):
        super().__init__(
            config,
            cache_to_disk,
            keep_frames=keep_frames,
            calc_stats=calc_stats,
            verbose=verbose,
            do_tracking=do_tracking,
        )
        self.use_opt_flow = use_opt_flow
        self.high_quality_optical_flow = high_quality_optical_flow
        # self.cache_to_disk = cache_to_disk
        # self.max_tracks = config.max_tracks
        # # frame_padding < 3 causes problems when we get small areas...
        # self.frame_padding = max(3, self.config.frame_padding)
        # # the dilation effectively also pads the frame so take it into consideration.
        # # self.frame_padding = max(0, self.frame_padding - self.config.dilation_pixels)
        # self.keep_frames = keep_frames
        # self.calc_stats = calc_stats
        # self._tracking_time = None
        # if self.config.dilation_pixels > 0:
        #     size = self.config.dilation_pixels * 2 + 1
        #     self.dilate_kernel = np.ones((size, size), np.uint8)

    def parse_clip(self, clip, process_background=False):
        """
        Loads a cptv file, and prepares for track extraction.
        """
        self._tracking_time = None
        start = time.time()
        clip.set_frame_buffer(
            self.high_quality_optical_flow,
            self.cache_to_disk,
            self.use_opt_flow,
            self.keep_frames,
        )
        clip.type = self.type
        with open(clip.source_file, "rb") as f:
            reader = CPTVReader(f)
            clip.set_res(reader.x_resolution, reader.y_resolution)
            if clip.from_metadata:
                for track in clip.tracks:
                    track.crop_regions()
            camera_model = None
            if reader.model:
                camera_model = reader.model.decode()
            clip.set_model(camera_model)

            # if we have the triggered motion threshold should use that
            # maybe even override dynamic threshold with this value
            if reader.motion_config:
                motion = yaml.safe_load(reader.motion_config)
                temp_thresh = motion.get("triggeredthresh")
                if temp_thresh:
                    clip.temp_thresh = temp_thresh

            video_start_time = reader.timestamp.astimezone(Clip.local_tz)
            clip.set_video_stats(video_start_time)
            clip.calculate_background(reader)

        with open(clip.source_file, "rb") as f:
            reader = CPTVReader(f)
            for frame in reader:
                if not process_background and frame.background_frame:
                    continue
                self.process_frame(clip, frame)

        if not clip.from_metadata and self.do_tracking:
            self.apply_track_filtering(clip)

        if self.calc_stats:
            clip.stats.completed()
        self._tracking_time = time.time() - start
        return True

    @property
    def tracking_time(self):
        return self._tracking_time

    def start_tracking(self, clip, frames, track_frames=True, **args):
        # no need to retrack all of preview
        do_tracking = self.do_tracking
        self.do_tracking = self.do_tracking and track_frames
        for frame in frames:
            self.process_frame(clip, frame)
        self.do_tracking = do_tracking

    def process_frame(self, clip, frame, **args):
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
        If specified background subtraction algorithm will be used.
        """
        ffc_affected = is_affected_by_ffc(frame)
        thermal = frame.pix.copy()
        if ffc_affected:
            self.print_if_verbose("{} ffc_affected".format(clip.current_frame))
        clip.ffc_affected = ffc_affected
        filtered, threshold = self._get_filtered_frame(clip, thermal)
        mask = None
        if self.do_tracking:
            _, mask, component_details, centroids = detect_objects(
                filtered.copy(), otsus=False, threshold=threshold, kernel=(5, 5)
            )
        cur_frame = clip.add_frame(thermal, filtered, mask, ffc_affected)
        if not self.do_tracking:
            return

        if clip.from_metadata:
            for track in clip.tracks:
                if clip.current_frame in track.frame_list:
                    track.add_frame_for_existing_region(
                        cur_frame,
                        threshold,
                        (
                            clip.frame_buffer.prev_frame.filtered
                            if clip.frame_buffer.prev_frame is not None
                            else None
                        ),
                    )
        else:
            regions = []
            if ffc_affected:
                clip.active_tracks = set()
            else:
                regions = self._get_regions_of_interest(
                    clip, component_details[1:], centroids[1:]
                )
                self._apply_region_matchings(clip, regions)
            clip.region_history.append(regions)


def get_background_filtered(background, thermal):
    """
    Calculates filtered frame from thermal
    :param thermal: the thermal frame
    :param background: (optional) used for background subtraction
    :return: uint8 filtered frame and adjusted clip threshold for normalized frame
    """

    filtered = np.float32(thermal.copy())

    avg_change = 0
    filtered = filtered - background
    filtered[filtered < 0] = 0
    filtered, stats = normalize(filtered, new_max=255)
    return filtered, 0
