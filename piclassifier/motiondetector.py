from threading import Lock
from datetime import datetime, timedelta
import logging

from astral import Location
import numpy as np

from ml_tools import tools
from ml_tools.tools import Rectangle
from .processor import Processor


class SlidingWindow:
    def __init__(self, shape, dtype):
        self.lock = Lock()
        self.frames = np.empty(shape, dtype)
        self.last_index = None
        self.size = len(self.frames)
        self.oldest_index = None

    @property
    def current(self):
        with self.lock:
            if self.last_index is not None:
                return self.frames[self.last_index]
            return None

    def current_copy(self):
        with self.lock:
            if self.last_index is not None:
                return self.frames[self.last_index].copy()
            return None

    def get_frames(self):
        with self.lock:
            if self.last_index is None:
                return []
            frames = []
            cur = self.oldest_index
            end_index = (self.last_index + 1) % self.size
            while len(frames) == 0 or cur != end_index:
                frames.append(self.frames[cur])
                cur = (cur + 1) % self.size
            return frames

    def get(self, i):
        i = i % self.size
        with self.lock:
            return self.frames[i]

    @property
    def oldest(self):
        with self.lock:
            if self.oldest_index is not None:
                return self.frames[self.oldest_index]
            return None

    def add(self, frame):
        with self.lock:
            if self.last_index is None:
                self.oldest_index = 0
                self.frames[0] = frame
                self.last_index = 0
            else:
                new_index = (self.last_index + 1) % self.size
                if new_index == self.oldest_index:
                    self.oldest_index = (self.oldest_index + 1) % self.size
                self.frames[new_index] = frame
                self.last_index = new_index

    def reset(self):
        with self.lock:
            self.last_index = None
            self.oldest_index = None


class MotionDetector(Processor):
    FFC_PERIOD = timedelta(seconds=9.9)
    BACKGROUND_WEIGHTING_PER_FRAME = 0.99
    BACKGROUND_WEIGHT_EVERY = 3

    def __init__(
        self, res_x, res_y, thermal_config, dynamic_thresh, recorder,
    ):
        self._res_x = res_x
        self._res_y = res_y
        self.config = thermal_config.motion
        self.location_config = thermal_config.location
        self.preview_frames = (
            thermal_config.recorder.preview_secs * thermal_config.recorder.frame_rate
        )
        self.compare_gap = self.config.frame_compare_gap + 1
        edge = self.config.edge_pixels
        self.min_frames = (
            thermal_config.recorder.min_secs * thermal_config.recorder.frame_rate
        )
        self.max_frames = (
            thermal_config.recorder.max_secs * thermal_config.recorder.frame_rate
        )
        self.clipped_window = SlidingWindow(
            (self.compare_gap, res_y - edge * 2, res_x - edge * 2), np.int32
        )
        self.diff_window = SlidingWindow(
            (self.compare_gap, res_y - edge * 2, res_x - edge * 2), np.int32
        )

        self.thermal_window = SlidingWindow(
            (self.preview_frames, res_y, res_x), np.uint16
        )
        self.processed = 0
        self.num_frames = 0
        self.thermal_thresh = 0
        self.background = None
        self.last_background_change = None
        self.background_weight = MotionDetector.BACKGROUND_WEIGHTING_PER_FRAME
        self.movement_detected = False
        self.dynamic_thresh = dynamic_thresh
        self.temp_thresh = self.config.temp_thresh
        self.crop_rectangle = Rectangle(edge, edge, res_x - 2 * edge, res_y - 2 * edge)

        self.start_rec = thermal_config.recorder.start_rec
        self.end_rec = thermal_config.recorder.end_rec
        self.use_sunrise = (
            thermal_config.recorder.start_rec.is_relative
            or thermal_config.recorder.end_rec.is_relative
        )

        self.last_sunrise_check = None
        self.location = None
        self.sunrise = None
        self.sunset = None
        self.recording = False
        if self.use_sunrise:
            self.set_location(self.location_config)

        self.recorder = recorder
        self.ffc_affected = False

    def set_location(self, location_config):
        self.location = Location()
        lat, lng = location_config.get_lat_long(use_default=True)
        self.location.latitude = lat
        self.location.longitude = lng

        self.location.altitude = location_config.altitude
        self.location.timezone = tools.get_timezone_str(lat, lng)

    def get_sunrise_sunet(self):
        date = datetime.now().date()
        if self.last_sunrise_check is None or date > self.last_sunrise_check:
            sun = self.location.sun()
            if self.start_rec.is_relative:
                self.start_rec.time = (
                    sun["sunset"] + timedelta(seconds=self.start_rec.offset_s)
                ).time()
            if self.end_rec.is_relative:
                self.end_rec.time = (
                    sun["sunrise"] + timedelta(seconds=self.end_rec.offset_s)
                ).time()
            self.last_sunrise_check = date
            logging.info(
                "start_rec is {} end_rec is {} next check is {}".format(
                    self.start_rec.time, self.end_rec.time, self.last_sunrise_check
                )
            )

    def calc_temp_thresh(self, thermal_frame, prev_ffc):
        if self.dynamic_thresh:
            temp_changed = False

            if prev_ffc:
                new_background = thermal_frame
                back_changed = True
            else:
                new_background = np.where(
                    self.background < thermal_frame * self.background_weight,
                    self.background,
                    thermal_frame,
                )
                back_changed = np.amax(self.background != new_background)

            if back_changed:
                self.last_background_change = self.processed
                self.background = new_background

                old_temp = self.temp_thresh
                self.temp_thresh = int(round(np.average(self.background)))
                if self.temp_thresh != old_temp:
                    logging.debug(
                        "{} MotionDetector temp threshold changed from {} to {} new background average is {} weighting was {}".format(
                            self.num_frames,
                            old_temp,
                            self.temp_thresh,
                            np.average(self.background),
                            self.background_weight,
                        )
                    )
                    temp_changed = True
                    self.background_weight = (
                        MotionDetector.BACKGROUND_WEIGHTING_PER_FRAME
                    )

            if (
                not temp_changed
                and self.processed % MotionDetector.BACKGROUND_WEIGHT_EVERY == 0
            ):
                self.background_weight = (
                    self.background_weight
                    * MotionDetector.BACKGROUND_WEIGHTING_PER_FRAME
                )

        else:
            self.temp_thresh = self.config.temp_thresh

    def detect(self, clipped_frame):
        oldest = self.clipped_window.oldest
        delta_frame = clipped_frame - oldest

        if not self.config.warmer_only:
            delta_frame = abs(delta_frame)
        if self.config.one_diff_only:
            diff = len(delta_frame[delta_frame > self.config.delta_thresh])
        else:
            if self.processed > 2:
                delta_frame2 = self.diff_window.oldest
                delta_frame[
                    delta_frame >= self.config.delta_thresh
                ] = self.config.delta_thresh
                delta_frame = delta_frame2 + delta_frame
                diff = len(delta_frame[delta_frame == self.config.delta_thresh * 2])
            else:
                delta_frame[
                    delta_frame >= self.config.delta_thresh
                ] = self.config.delta_thresh
                diff = 0

        self.diff_window.add(delta_frame)

        if diff > self.config.count_thresh:
            if not self.movement_detected:
                print(
                    "{} MotionDetector motion detected thresh {} count {}".format(
                        timedelta(seconds=self.num_frames / 9), self.temp_thresh, diff
                    )
                )
                logging.debug(
                    "{} MotionDetector motion detected thresh {} count {}".format(
                        timedelta(seconds=self.num_frames / 9), self.temp_thresh, diff
                    )
                )
            return True

        if self.movement_detected:
            print(
                "{} MotionDetector motion stopped thresh {} count {}".format(
                    timedelta(seconds=self.num_frames / 9), self.temp_thresh, diff
                )
            )
            logging.debug(
                "{} MotionDetector motion stopped thresh {} count {}".format(
                    timedelta(seconds=self.num_frames / 9), self.temp_thresh, diff
                )
            )
        return False

    def get_recent_frame(self):
        return self.thermal_window.current_copy()

    def can_record(self):
        if self.use_sunrise:
            self.get_sunrise_sunet()

        return self.start_rec.is_after() and self.end_rec.is_before()

    def disconnected(self):
        self.clipped_window.reset()
        self.thermal_window.reset()
        self.diff_window.reset()
        self.processed = 0
        self.recorder.force_stop()

    def process_frame(self, lepton_frame):
        if self.can_record() or (self.recorder and self.recorder.recording):
            cropped_frame = self.crop_rectangle.subimage(lepton_frame.pix)
            frame = np.int32(cropped_frame)
            prev_ffc = self.ffc_affected
            self.ffc_affected = is_affected_by_ffc(lepton_frame)
            if not self.ffc_affected:
                self.thermal_window.add(lepton_frame.pix)
                if self.background is None:
                    self.background = lepton_frame.pix
                    self.last_background_change = self.processed
                else:
                    self.calc_temp_thresh(lepton_frame.pix, prev_ffc)

            clipped_frame = np.clip(frame, a_min=self.temp_thresh, a_max=None)
            self.clipped_window.add(clipped_frame)

            if self.ffc_affected or prev_ffc:
                logging.debug("{} MotionDetector FFC".format(self.num_frames))
                self.movement_detected = False
                self.clipped_window.oldest_index = self.clipped_window.last_index
            elif self.processed != 0:
                self.movement_detected = self.detect(clipped_frame)
            self.processed += 1
            if self.recorder:
                self.recorder.process_frame(
                    self.movement_detected, lepton_frame, self.temp_thresh
                )
        else:
            self.movement_detected = False
        self.num_frames += 1

    @property
    def res_x(self):
        return self._res_x

    @property
    def res_y(self):
        return self._res_y


def is_affected_by_ffc(lepton_frame):
    if lepton_frame.time_on is None or lepton_frame.last_ffc_time is None:
        return False

    return (
        lepton_frame.time_on - lepton_frame.last_ffc_time
    ) < MotionDetector.FFC_PERIOD
