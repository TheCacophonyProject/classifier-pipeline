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
from ml_tools.imageprocessing import (
    detect_objects,
    normalize,
    detect_objects_ir,
    theshold_saliency,
    detect_objects_both,
)
from track.cliptracker import ClipTracker

DO_SALIENCY = False
DEBUG_TRAP = False


class Line:
    def __init__(self, m, c):
        self.m = m
        self.c = c

    def is_above(self, point):
        y_value = self.y_res(point[0])
        if point[1] > y_value:
            return True
        return False

    def is_below(self, point):
        return not self.is_above(point)

    def is_left(self, point):
        x_value = self.x_res(point[1])
        if point[0] < x_value:
            return True
        return False

    def is_right(self, point):
        return not self.is_left(point)

    def y_res(self, x):
        return x * self.m + self.c

    def x_res(self, y):
        return (y - self.c) / self.m
        # return x * self.m + self.c


class IRTrackExtractor(ClipTracker):

    PREVIEW = "preview"
    VERSION = 10
    TYPE = "IR"

    LEFT_BOTTOM = Line(5 / 14, 160)
    RIGHT_BOTTOM = Line(-5 / 12, 421.7)
    # BACK_TOP = Line(0, 250)
    # BACK_BOTTOM = Line(0, 170)

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
        do_tracking=True,
        on_trapped=None,
    ):
        super().__init__(
            config,
            cache_to_disk,
            keep_frames,
            calc_stats,
            verbose,
            do_tracking=do_tracking,
        )
        self.scale = scale
        self.on_trapped = on_trapped
        self.saliency = None
        if self.scale:
            self.frame_padding = int(scale * self.frame_padding)
            self.min_dimension = int(scale * self.min_dimension)
        self.background = None
        self.res_x = None
        self.res_y = None

    def parse_clip(self, clip, process_background=False):
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
            max_frames=None
            if self.keep_frames
            else 51,  # enough to cover back comparison
        )

        _, ext = os.path.splitext(clip.source_file)
        background = None
        vidcap = cv2.VideoCapture(clip.source_file)
        fail_count = 0
        while True:
            success, image = vidcap.read()
            if not success:
                if fail_count < 1:
                    fail_count += 1
                    # try once more if its first fail as the mp4s from pi have errors at key frames
                    continue
                break
            fail_count = 0
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if clip.current_frame == -1:
                self.res_x = gray.shape[0]
                self.res_y = gray.shape[1]
                clip.set_res(gray.shape[1], gray.shape[0])
                if clip.from_metadata:
                    for track in clip.tracks:
                        track.crop_regions()
                background = np.uint8(gray)
                # cv2.imshow("bak", np.uint8(background))
                # cv2.waitKey(1000)
                self.start_tracking(clip, background=gray, background_frames=50)
            self.process_frame(clip, gray)
        vidcap.release()
        if not clip.from_metadata and self.do_tracking:
            self.apply_track_filtering(clip)

        if self.calc_stats:
            clip.stats.completed(clip.current_frame, clip.res_y, clip.res_x)
        self._tracking_time = time.time() - start
        return True

    def start_tracking(self, clip, frames=None, background=None, background_frames=1):

        self.res_x = clip.res_x
        self.res_y = clip.res_y
        if DO_SALIENCY:
            self.init_saliency()
        clip.set_model("IR")
        clip.set_video_stats(datetime.now())
        self.background = Background()
        if background is not None:
            if self.scale:
                background = cv2.resize(
                    background,
                    (int(self.res_x * self.scale), int(self.res_y * self.scale)),
                )
            self.background.set_background(background, background_frames)
        self.init_saliency()
        if frames is not None:
            for frame in frames:
                self.process_frame(clip, frame)

    def init_saliency(self):
        res_x = self.res_x
        res_y = self.res_y
        if self.scale is not None:
            res_x = int(self.res_x * self.scale)
            res_y = int(self.res_y * self.scale)
        self.saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        self.saliency.setImagesize(res_x, res_y)
        self.saliency.init()

    def process_frame(self, clip, frame, ffc_affected=False):
        start = time.time()
        if len(frame.shape) == 3:
            # in rgb so convert to gray
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.saliency is None and DO_SALIENCY:
            if self.resize_dims is not None:
                self.init_saliency(
                    int(self.res_x * self.scale), int(self.res_y * self.scale)
                )
            else:
                self.init_saliency(clip.res_x, clip.res_y)

        if ffc_affected:
            self.print_if_verbose("{} ffc_affected".format(clip.current_frame))
        clip.ffc_affected = ffc_affected
        self._process_frame(clip, frame, ffc_affected)

    def _get_filtered_frame_ir(self, thermal, repeats=1):
        if not DO_SALIENCY:
            return thermal, 0
        for _ in range(repeats):
            (success, saliencyMap) = self.saliency.computeSaliency(thermal)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        return saliencyMap, 0

    # merge all regions that the midpoint is within the max(width,height) from the midpoint of another region
    # keep merging until no more merges are possible, tihs works paticularly well from the IR videos where
    # the filtered image is quite fragmented
    def merge_components(self, rectangles):
        min_mass = 10
        min_size = 4
        rectangles = [
            r
            for r in rectangles
            if r[4] > min_mass or (r[2] > min_size and r[3] > min_size)
        ]
        # filter out regions with small mass  and samll width / height
        #  numbers may need adjusting
        MAX_GAP = 20
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
                    cur_right = rect[0] + rect[2]
                    cur_bottom = rect[0] + rect[2]

                    rect[0] = min(rect[0], r_2[0])
                    rect[1] = min(rect[1], r_2[1])
                    rect[2] = max(cur_right, r_2[0] + r_2[2])
                    rect[3] = max(rect[1] + rect[3], r_2[1] + r_2[3])
                    rect[2] -= rect[0]
                    rect[3] -= rect[1]
                    rect[4] += r_2[4]

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

    def _process_frame(self, clip, frame, ffc_affected=False):

        wait = 1
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
            If specified background subtraction algorithm will be used.
        """
        repeats = 1
        if clip.current_frame < 6:
            # helps init the saliency, when ir tracks have a 5 second preview this shouldnt be needed
            repeats = 6
        start = time.time()
        tracking_thermal = frame.copy()
        if self.scale:
            tracking_thermal = cv2.resize(
                tracking_thermal,
                (int(self.res_x * self.scale), int(self.res_y * self.scale)),
            )
        if self.background._background is None:
            self.background.set_background(tracking_thermal.copy())
        saliencyMap = None
        if self.do_tracking:
            saliencyMap, _ = self._get_filtered_frame_ir(
                tracking_thermal, repeats=repeats
            )
        start = time.time()

        backsub, _ = get_ir_back_filtered(
            self.background.background, tracking_thermal, clip.background_thresh
        )

        threshold = 0
        if np.amin(saliencyMap) == 255:
            num = 0
            mask = saliencyMap.copy()
            component_details = []
            saliencyMap[:] = 0
        else:
            pass
            # backsub = np.where(saliencyMap > 0, saliencyMap, backsub)

        cur_frame = clip.add_frame(frame, backsub, saliencyMap, ffc_affected)
        start = time.time()
        self.background.update_background(tracking_thermal, backsub)
        clip.set_background(self.background.background)

        if not self.do_tracking:
            return
        start = time.time()

        num, mask, component_details = theshold_saliency(backsub, threshold=0)
        component_details = component_details[1:]
        component_details = self.merge_components(component_details)

        if clip.from_metadata:
            for track in clip.tracks:
                if clip.current_frame in track.frame_list:
                    track.add_frame_for_existing_region(
                        cur_frame,
                        threshold,
                        clip.frame_buffer.current_frame.filtered,
                    )
        else:
            regions = []
            if ffc_affected:
                clip.active_tracks = set()
            else:
                s = time.time()
                regions = self._get_regions_of_interest(clip, component_details)

                self._apply_region_matchings(clip, regions)
            for track in clip.active_tracks:
                if track.trap_reported:
                    continue
                inside_trap(track, self.scale)
                if track.in_trap:
                    filtered = self.filter_track(clip, track, track.get_stats())
                    if not filtered:
                        track.trigger_frame = cur_frame.frame_number
                        # fire trapping event
                        if self.on_trapped is not None:
                            track.trap_reported = True
                            self.on_trapped(track)
            clip.region_history.append(regions)
            if DEBUG_TRAP:
                if len(clip.tracks) > 0:
                    self.show_trap_info(clip, frame)

    def show_trap_info(self, clip, frame):
        image = frame.copy()
        in_trap = False
        wait = 100
        start = (0, 480 - int(IRTrackExtractor.LEFT_BOTTOM.y_res(0)))
        end = (int(IRTrackExtractor.LEFT_BOTTOM.x_res(240)), 480 - 240)
        image = cv2.line(image, start, end, (0, 255, 0), 10)
        start = (640, 480 - int(IRTrackExtractor.RIGHT_BOTTOM.y_res(640)))
        end = (int(IRTrackExtractor.RIGHT_BOTTOM.x_res(240)), 480 - 240)

        image = cv2.line(image, start, end, (0, 255, 0), 10)

        for track in clip.active_tracks:
            filtered = self.filter_track(clip, track, track.get_stats())
            in_trap = in_trap or (track.in_trap and not filtered)
            region = track.bounds_history[-1]
            if self.scale:
                region = region.copy()
                region.rescale(1 / self.scale)
                region.in_trap = track.last_bound.in_trap
            start_point = (int(region.left), int(region.top))
            end_point = (int(region.right), int(region.bottom))
            image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)
            if region.in_trap:
                image = cv2.putText(
                    image,
                    f"Trapped?{ track.in_trap} f?{filtered}",
                    start_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                )
                # wait = 300
        if in_trap:
            image = cv2.putText(
                image,
                f"CAUGHT EM",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
            )
            # wait = 300
        cv2.imshow("id", image)
        cv2.waitKey(wait)
        # 1 / 0

    def filter_components(self, component_details):
        filtered = []
        for component in component_details:
            region = Region(
                component[0],
                component[1],
                component[2],
                component[3],
            )
            p = (region.right, 480 - region.bottom)

            filter = (
                IRTrackExtractor.LEFT_BOTTOM.is_above(p)
                and IRTrackExtractor.LEFT_BOTTOM.is_left(p)
                # or IRTrackExtractor.RIGHT_BOTTOM.is_above(region)
                # or IRTrackExtractor.BACK_BOTTOM.is_above(region)
            )
            p = (region.left, 480 - region.bottom)

            filter = filter or (
                IRTrackExtractor.RIGHT_BOTTOM.is_above(p)
                and IRTrackExtractor.RIGHT_BOTTOM.is_right(p)
            )
            p = (region.left, 480 - region.bottom)

            # filter = filter or (IRTrackExtractor.BACK_TOP.is_above(p))
            if not filter:
                filtered.append(component)
            else:
                logging.info("Filtered components %s", region)
        return filtered

    def filter_track(self, clip, track, stats):
        # return False
        # return not track.stable
        # discard any tracks that are less min_duration
        # these are probably glitches anyway, or don't contain enough information.
        if len(track) < self.config.min_duration_secs * clip.frames_per_second:
            self.print_if_verbose("Track filtered. Too short, {}".format(len(track)))
            clip.filtered_tracks.append(("Track filtered.  Too short", track))
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

    def get_delta_frame(self, clip):
        # GP used to be 50 frames ago
        frame = clip.frame_buffer.current_frame
        prev_i = max(0, min(10, clip.current_frame - 10))
        s = time.time()
        prev_frame = clip.frame_buffer.get_frame_ago(prev_i)
        if prev_i == frame.frame_number:
            return None, None

        s = time.time()

        delta_filtered = np.abs(frame.filtered - prev_frame.filtered)
        delta_thermal = np.abs(frame.thermal - prev_frame.thermal)
        return delta_thermal, delta_filtered


def get_ir_back_filtered(background, thermal, back_thresh):
    """
    Calculates filtered frame from thermal
    :param thermal: the thermal frame
    :param background: (optional) used for background subtraction
    :return: uint8 filtered frame and adjusted clip threshold for normalized frame
    """

    filtered = np.float32(thermal.copy())

    avg_change = 0
    filtered = abs(filtered - background)
    filtered[filtered < back_thresh] = 0
    filtered, stats = normalize(filtered, new_max=255)

    # filtered[filtered > 10] += 30
    return filtered, 0


class Background:
    def __init__(self):
        self.frames = 1
        self._background = None

    def set_background(self, background, frames=1):
        self.frames = frames
        self._background = np.float32(background) * self.frames
        return

    def update_background(self, thermal, filtered):
        background = self.background
        new_thermal = np.where(filtered > 0, background, thermal)
        self._background += new_thermal
        self.frames += 1

    @property
    def background(self):
        return self._background / self.frames


LEFT = 1
BOTTOM = 2
RIGHT = 4
TOP = 8
MIDDLE = 16


def inside_trap(track, scale=None):
    region = track.last_bound.copy()
    if scale:
        region.rescale(1 / scale)
    if region.width < 60 or region.height < 40:
        # dont want small regions
        return False
    if track.direction == 0:
        if region.left < 100:
            track.direction |= LEFT
        if region.right > (640 - 100):
            track.direction |= RIGHT
        if region.bottom > (480 - 100):
            track.direction |= BOTTOM
        if track.direction == 0:
            if region.bottom < 300:
                track.direction |= TOP
            else:
                track.direction = MIDDLE
    p = (region.right, 480 - region.top)

    inside = (
        IRTrackExtractor.LEFT_BOTTOM.is_below(p)
        and IRTrackExtractor.LEFT_BOTTOM.is_right(p)
        # or IRTrackExtractor.RIGHT_BOTTOM.is_above(region)
        # or IRTrackExtractor.BACK_BOTTOM.is_above(region)
    )
    x_pos = IRTrackExtractor.LEFT_BOTTOM.x_res(p[1])
    x_diff = abs(p[0] - x_pos)
    left_percent = x_diff / region.width

    p = (region.left, 480 - region.top)

    inside = inside and (
        IRTrackExtractor.RIGHT_BOTTOM.is_below(p)
        and IRTrackExtractor.RIGHT_BOTTOM.is_left(p)
    )
    x_pos = IRTrackExtractor.RIGHT_BOTTOM.x_res(p[1])
    # could try using bottom  rather than region.top here
    x_diff = abs(p[0] - x_pos)
    right_percent = x_diff / region.width
    #
    # print(
    #     f"checking direction of {region.left}x{region.top} - {region.right}x{region.bottom}",
    #     "from dir",
    #     track.direction,
    #     "x pos",
    #     x_pos,
    #     "distance",
    #     x_diff,
    #     " regin.width",
    #     region.width,
    #     "l percent",
    #     left_percent,
    #     "r percent",
    #     right_percent,
    # )

    if not inside:
        return False
    in_trap = False
    if left_percent < 0.5 and right_percent < 0.5:
        return False
    if track.direction & LEFT and region.left > 40 and left_percent > 0.5:
        # print("Track from left")
        in_trap = True

    elif track.direction & RIGHT and region.right < 580 and right_percent > 0.5:
        # print("track from rgiht")
        in_trap = True

    if track.direction == TOP and region.bottom > 300:
        # print("track from top")
        in_trap = True
    if track.direction == BOTTOM and region.bottom < 480 - 50:
        # print("track from bottom")
        in_trap = True

    if track.direction == MIDDLE and region.left > 40 and region.right < 580:
        # print("MIDDLE PASS")
        in_trap = True
    track.last_bound.in_trap = in_trap
    track.update_trapped_state()
    return in_trap
