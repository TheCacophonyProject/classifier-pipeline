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

import numpy as np
import time
import yaml
from datetime import datetime

from .clip import Clip
from piclassifier.cptvmotiondetector import is_affected_by_ffc
from ml_tools.imageprocessing import detect_objects, normalize
from track.cliptracker import ClipTracker
import logging
from cptv_rs_python_bindings import CptvReader
from piclassifier.motiondetector import WeightedBackground


class ClipTrackExtractor(ClipTracker):
    PREVIEW = "preview"
    VERSION = 11
    TYPE = "thermal"

    @property
    def tracker_version(self):
        return self.version

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
        update_background=True,
        calculate_filtered=False,
        calculate_thumbnail_info=False,
        from_pi=False,
    ):
        super().__init__(
            config,
            cache_to_disk,
            keep_frames=keep_frames,
            calc_stats=calc_stats,
            verbose=verbose,
            do_tracking=do_tracking,
            calculate_thumbnail_info=calculate_thumbnail_info,
        )

        if from_pi:
            self.version = f"PI-{ClipTrackExtractor.VERSION}"
        else:
            self.version = ClipTrackExtractor.VERSION

        self.use_opt_flow = use_opt_flow
        self.high_quality_optical_flow = high_quality_optical_flow
        self.background_alg = None
        self.update_background = update_background
        self.calculate_filtered = calculate_filtered
        self.weighting_percent = 1
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

    def init_clip(self, clip):

        clip.set_frame_buffer(
            self.high_quality_optical_flow,
            self.cache_to_disk,
            self.use_opt_flow,
            self.keep_frames,
        )
        clip.type = self.type
        reader = CptvReader(str(clip.source_file))
        header = reader.get_header()

        clip.set_res(header.x_resolution, header.y_resolution)
        if clip.from_metadata:
            for track in clip.tracks:
                track.crop_regions()
        camera_model = None
        if header.model:
            camera_model = header.model
        clip.set_model(camera_model)

        # if we have the triggered motion threshold should use that
        # maybe even override dynamic threshold with this value
        if header.motion_config:
            motion = yaml.safe_load(header.motion_config)
            temp_thresh = motion.get("triggeredthresh")
            if temp_thresh:
                clip.temp_thresh = temp_thresh
        video_start_time = datetime.fromtimestamp(header.timestamp / 1000000)
        video_start_time = video_start_time.astimezone(Clip.local_tz)

        clip.set_video_stats(video_start_time)
        if camera_model == "lepton3.5":
            weight_add = 1 / self.weighting_percent
        else:
            weight_add = 0.1 / self.weighting_percent

        frame = reader.next_frame()
        clip.update_background(frame.pix)
        clip._background_calculated()
        self.background_alg = WeightedBackground(
            clip.crop_rectangle.x,
            clip.crop_rectangle,
            clip.res_x,
            clip.res_y,
            weight_add,
            clip.temp_thresh,
        )
        self.background_alg.process_frame(frame.pix)

    def parse_clip(self, clip, process_background=False):
        """
        Loads a cptv file, and prepares for track extraction.
        """

        self._tracking_time = None
        start = time.time()
        self.init_clip(clip)
        self._track_clip(clip, process_background=process_background)
        if self.calc_stats:
            clip.stats.completed()
        self._tracking_time = time.time() - start
        return True

    def _track_clip(self, clip, process_background=False):

        if clip.background is None:
            logging.error("Clip has no background have you called init_clip first")
            raise Exception("Clip has no background have you called init_clip first")
        reader = CptvReader(str(clip.source_file))
        while True:
            frame = reader.next_frame()

            if frame is None:
                break

            if not process_background and frame.background_frame:
                continue
            self.process_frame(clip, frame)
            if self.update_background or self.background_alg.background is None:
                # use mean of last 45 frames to update background, this will help
                # when pixels become cooler for a very short time i.e. tracked object is cooler than background
                last_avg = np.mean(
                    [f.thermal for f in clip.frame_buffer.get_last_x(x=45)], axis=0
                )
                self.background_alg.process_frame(last_avg)

        if not clip.from_metadata and self.do_tracking:
            self.apply_track_filtering(clip)

    @property
    def tracking_time(self):
        return self._tracking_time

    def start_tracking(
        self, clip, frames, track_frames=True, background_alg=None, **args
    ):
        # no need to retrack all of preview
        do_tracking = self.do_tracking
        self.background_alg = background_alg
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
        mask = None
        filtered = None
        if self.do_tracking or self.calculate_filtered or self.calculate_thumbnail_info:
            filtered = np.float32(frame.pix) - self.background_alg.background
        if self.do_tracking or self.calculate_thumbnail_info:
            obj_filtered, threshold = self._get_filtered_frame(
                clip, thermal, denoise=self.config.denoise
            )
            _, mask, component_details, centroids = detect_objects(
                obj_filtered, otsus=False, threshold=threshold, kernel=(5, 5)
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
