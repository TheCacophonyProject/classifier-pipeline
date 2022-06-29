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
import matplotlib.pyplot as plt

import os
import logging
import numpy as np
import time
import yaml
from datetime import datetime

from cptv import CPTVReader
import cv2

from .clip import Clip
from ml_tools.tools import Rectangle
from track.region import Region
from track.track import Track
from piclassifier.motiondetector import is_affected_by_ffc
from ml_tools.imageprocessing import (
    detect_objects,
    normalize,
    detect_objects_ir,
    theshold_saliency,
    detect_objects_both,
)
from track.cliptracker import ClipTracker


class IRTrackExtractor(ClipTracker):

    PREVIEW = "preview"
    VERSION = 10
    TYPE = "IR"

    @property
    def tracker_version(self):
        return f"IRTrackExtractor-{IRTrackExtractor.VERSION}"

    @property
    def type(self):
        return IRTrackExtractor.TYPE

    @property
    def tracking_time(self):
        return self._tracking_time

    def __init__(
        self,
        config,
        cache_to_disk=False,
        keep_frames=True,
        calc_stats=True,
        verbose=False,
        scale=None,
    ):
        super().__init__(
            config,
            cache_to_disk,
            keep_frames,
            calc_stats,
            verbose,
        )
        self.scale = scale
        self.saliency = None
        if self.scale:
            self.frame_padding = int(scale * self.frame_padding)
            self.min_dimension = int(scale * self.min_dimension)

    def parse_clip(self, clip, process_background=False, track=True):
        """
        Loads a cptv file, and prepares for track extraction.
        """
        clip.type = self.type
        self._tracking_time = None
        start = time.time()
        clip.set_frame_buffer(
            False,
            self.cache_to_disk,
            False,
            self.keep_frames,
        )
        _, ext = os.path.splitext(clip.source_file)
        count = 0
        background = None
        vidcap = cv2.VideoCapture(clip.source_file)
        while True:
            success, image = vidcap.read()
            if not success:
                break
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if count == 0:
                background = gray
                self.init_saliency(gray.shape[1], gray.shape[0])
                clip.set_res(gray.shape[1], gray.shape[0])
                clip.set_model("IR")
                clip.set_video_stats(datetime.now())
                self._get_filtered_frame_ir(gray, repeats=2)

            else:
                background = np.minimum(background, gray)

            count += 1

        vidcap.release()
        if background is None:
            return False
        background = cv2.GaussianBlur(background, (15, 15), 0)

        clip.update_background(background)
        vidcap = cv2.VideoCapture(clip.source_file)
        while True:
            success, image = vidcap.read()
            if not success:
                break
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.process_frame(clip, gray, track=track)
            # if clip.current_frame > 200:
            #     break
        vidcap.release()

        if not clip.from_metadata and track:
            self.apply_track_filtering(clip)

        if self.calc_stats:
            clip.stats.completed(clip.current_frame, clip.res_y, clip.res_x)
        self._tracking_time = time.time() - start
        return True

    def start_tracking(self, clip, frames):
        if len(frames) == 0:
            return
        res_x = clip.res_x
        res_y = clip.res_y
        if self.scale is not None:
            res_x = int(self.res_x * self.scale)
            res_y = int(self.res_y * self.scale)

            # clip.resized_background = cv2.resize(clip.background, self.resize_dims)
            # logging.info("Resizing backgorund %s", clip.tracking_background.shape)
        self.init_saliency(res_x, res_y)
        for frame in frames[-9:]:
            self.process_frame(clip, frame.pix.copy())

    def init_saliency(self, width, height):
        self.saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        self.saliency.setImagesize(width, height)
        self.saliency.init()

    def process_frame(self, clip, frame, ffc_affected=False, track=True):
        if self.saliency is None:
            if self.resie_dims is not None:
                self.init_saliency(
                    int(self.res_x * self.scale), int(self.res_y * self.scale)
                )
            else:
                self.init_saliency(clip.res_x, clip.res_y)

        if ffc_affected:
            self.print_if_verbose("{} ffc_affected".format(clip.current_frame))
        clip.ffc_affected = ffc_affected

        self._process_frame(clip, frame, ffc_affected, track=track)

    def _get_filtered_frame_ir(self, thermal, repeats=1):
        for _ in range(repeats):
            (success, saliencyMap) = self.saliency.computeSaliency(thermal)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        return saliencyMap, 0

    # merge all regions that the midpoint is within the max(width,height) from the midpoint of another region
    # keep merging until no more merges are possible, tihs works paticularly well from the IR videos where
    # the filtered image is quite fragmented
    def merge_components(self, rectangles):
        MAX_GAP = 40
        rect_i = 0
        rectangles = list(rectangles)
        while rect_i < len(rectangles):
            rect = rectangles[rect_i]
            merged = False
            mid_x = rect[2] / 2.0 + rect[0]
            mid_y = rect[3] / 2.0 + rect[1]
            index = 0
            while index < len(rectangles):
                r_2 = rectangles[index]
                if r_2[0] == rect[0]:
                    index += 1
                    continue
                r_mid_x = r_2[2] / 2.0 + r_2[0]
                r_mid_y = r_2[3] / 2.0 + r_2[1]
                distance = (mid_x - r_mid_x) ** 2 + (r_mid_y - mid_y) ** 2
                distance = distance ** 0.5

                # widest = max(rect[2], rect[3])
                # hack short cut just take line from mid points as shortest distance subtract biggest width or hieght from each
                distance = (
                    distance - max(rect[2], rect[3]) / 2.0 - max(r_2[2], r_2[3]) / 2.0
                )
                within = r_2[0] > rect[0] and (r_2[0] + r_2[2]) <= (rect[0] + rect[2])
                within = (
                    within
                    and r_2[1] > rect[1]
                    and (r_2[1] + r_2[3]) <= (rect[1] + rect[3])
                )

                if distance < MAX_GAP or within:
                    rect[0] = min(rect[0], r_2[0])
                    rect[1] = min(rect[1], r_2[1])
                    rect[2] = max(rect[0] + rect[2], r_2[0] + r_2[2])
                    rect[3] = max(rect[1] + rect[3], r_2[1] + r_2[3])
                    rect[2] -= rect[0]
                    rect[3] -= rect[1]
                    # print("second merged ", rect)
                    merged = True
                    # break
                    del rectangles[index]
                else:
                    index += 1
                    # print("not mered", rect, r_2, distance)
            if merged:
                rect_i = 0
            else:
                rect_i += 1
        return rectangles

    def _process_frame(self, clip, thermal, ffc_affected=False, track=True):

        wait = 1
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
            If specified background subtraction algorithm will be used.
        """
        repeats = 1
        if clip.current_frame < 6:
            repeats = 8

        tracking_thermal = thermal
        if self.scale:
            tracking_thermal = cv2.resize(
                thermal, (int(self.res_x * self.scale), int(self.res_y * self.scale))
            )

        saliencyMap = None
        if track:
            saliencyMap, _ = self._get_filtered_frame_ir(
                tracking_thermal, repeats=repeats
            )
            if self.scale:
                saliencyMap = cv2.resize(
                    saliencyMap, (clip.res_x, clip.res_y), cv2.INTER_NEAREST
                )
        backsub, _ = get_ir_back_filtered(clip.background, thermal)
        clip.add_frame(thermal, backsub, saliencyMap, ffc_affected)
        if not track:
            return
        prev_filtered = clip.frame_buffer.get_last_filtered()
        threshold = 0
        if np.amin(saliencyMap) == 255:
            num = 0
            mask = saliencyMap.copy()
            component_details = []
            saliencyMap[:] = 0
        else:
            num, mask, component_details = theshold_saliency(saliencyMap)
            component_details = self.merge_components(component_details[1:])

        if clip.from_metadata:
            for track in clip.tracks:
                if clip.current_frame in track.frame_list:
                    track.add_frame_for_existing_region(
                        clip.frame_buffer.get_last_frame(),
                        threshold,
                        prev_filtered,
                    )
        else:
            regions = []
            if ffc_affected:
                clip.active_tracks = set()
            else:
                regions = self._get_regions_of_interest(
                    clip, component_details, backsub, prev_filtered
                )
                self._apply_region_matchings(clip, regions)

            clip.region_history.append(regions)

    def filter_track(self, clip, track, stats):
        # return not track.stable
        # return False
        # discard any tracks that are less min_duration
        # these are probably glitches anyway, or don't contain enough information.

        if len(track) < self.config.min_duration_secs * clip.frames_per_second:
            self.print_if_verbose("Track filtered. Too short, {}".format(len(track)))
            clip.filtered_tracks.append(("Track filtered.  Too much overlap", track))
            return True

        # discard tracks that do not move enough

        if (
            stats.max_offset < self.config.track_min_offset
            or stats.frames_moved < self.config.min_moving_frames
        ):
            self.print_if_verbose(
                "Track filtered.  Didn't move {}".format(stats.max_offset)
            )
            clip.filtered_tracks.append(("Track filtered.  Didn't move", track))
            return True

        # if stats.blank_percent > self.config.max_blank_percent:
        #     self.print_if_verbose("Track filtered.  Too Many Blanks")
        #     clip.filtered_tracks.append(("Track filtered. Too Many Blanks", track))
        #     return True

        highest_ratio = 0
        for other in clip.tracks:
            if track == other:
                continue
            highest_ratio = max(track.get_overlap_ratio(other), highest_ratio)

        if highest_ratio > self.config.track_overlap_ratio:
            self.print_if_verbose(
                "Track filtered.  Too much overlap {}".format(highest_ratio)
            )
            clip.filtered_tracks.append(("Track filtered.  Too much overlap", track))
            return True

        return False


def get_ir_back_filtered(background, thermal):
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

    # filtered[filtered > 10] += 30
    return filtered, 0
