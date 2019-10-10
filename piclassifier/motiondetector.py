from datetime import datetime, timedelta
from astral import Location
import logging
import numpy as np
from ml_tools import tools
from ml_tools.tools import Rectangle


class SlidingWindow:
    def __init__(self, shape, dtype):
        self.frames = np.empty(shape, dtype)
        self.last_index = None
        self.size = len(self.frames)
        self.oldest_index = None

    @property
    def current(self):
        if self.last_index is not None:
            return self.frames[self.last_index]
        return None

    def get_frames(self):
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
        return self.frames[i]

    @property
    def oldest(self):
        if self.oldest_index is not None:
            return self.frames[self.oldest_index]
        return None

    def add(self, frame):

        if self.last_index is None:
            self.last_index = 0
            self.oldest_index = 0
            self.frames[0] = frame
        else:
            self.last_index = (self.last_index + 1) % self.size
            if self.last_index == self.oldest_index:
                self.oldest_index = (self.oldest_index + 1) % self.size
            self.frames[self.last_index] = frame

    def reset(self):
        self.last_index = None
        self.oldest_index = None


class MotionDetector:
    FFC_PERIOD = timedelta(seconds=9.9)
    BACKGROUND_WEIGHTING_PER_FRAME = 0.99
    BACKGROUND_WEIGHT_EVERY = 3

    def __init__(
        self,
        res_x,
        res_y,
        config,
        location_config,
        recorder_config,
        dynamic_thresh,
        recorder,
    ):
        self.config = config
        self.location_config = location_config
        self.preview_frames = recorder_config.preview_secs * recorder_config.frame_rate
        self.compare_gap = config.frame_compare_gap + 1
        edge = config.edge_pixels
        self.min_frames = recorder_config.min_secs * recorder_config.frame_rate
        self.max_frames = recorder_config.max_secs * recorder_config.frame_rate
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
        self.temp_thresh = config.temp_thresh
        self.crop_rectangle = Rectangle(edge, edge, res_x - 2 * edge, res_y - 2 * edge)
        self.use_sunrise = recorder_config.use_sunrise_sunset

        self.last_sunrise_check = None
        self.location = None
        self.sunrise = None
        self.sunset = None
        self.recording = False
        if self.use_sunrise:
            self.sunrise_offset = recorder_config.sunrise_offset
            self.sunset_offset = recorder_config.sunset_offset
            self.set_location(location_config)

        self.recorder = recorder
        self.ffc_affected = False

    def set_location(self, location_config):
        self.location = Location()
        self.location.latitude = location_config.latitude
        self.location.longitude = location_config.longitude

        self.location.altitude = location_config.altitude
        self.location.timezone = tools.get_timezone_str(
            location_config.latitude, location_config.longitude
        )

    def set_sunrise_sunet(self):
        date = datetime.now().date()
        if self.last_sunrise_check is None or date > self.last_sunrise_check:
            sun = self.location.sun()

            self.sunrise = (
                sun["sunrise"] + timedelta(minutes=self.sunrise_offset)
            ).time()
            self.sunset = (sun["sunset"] + timedelta(minutes=self.sunset_offset)).time()
            self.last_sunrise_check = date
            logging.info(
                "sunrise is {} sunset is {} next check is {}".format(
                    self.sunrise, self.sunset, self.last_sunrise_check
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
                self.temp_thresh = int(
                    round(min(self.config.temp_thresh, np.average(self.background)))
                )
                if self.temp_thresh != old_temp:
                    logging.debug(
                        "{} MotionDetector temp threshold changed from {} to {} new backgroung average is {} weighting was {}".format(
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
                temp_changed is False
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

        self.diff_window.add(delta_frame)

        if diff > self.config.count_thresh:
            if not self.movement_detected:
                logging.debug(
                    "{} MotionDetector motion detected thresh {} count {}".format(
                        timedelta(seconds=self.num_frames / 9), self.temp_thresh, diff
                    )
                )
            return True

        if self.movement_detected:
            logging.debug(
                "{} MotionDetector motion stopped thresh {} count {}".format(
                    timedelta(seconds=self.num_frames / 9), self.temp_thresh, diff
                )
            )
        return False

    def can_record(self):
        if self.use_sunrise:
            self.set_sunrise_sunet()
            time = datetime.now().time()
            return time > self.sunset or time < self.sunrise
        return True

    def force_stop(self):
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
            self.ffc_affected = MotionDetector.is_affected_by_ffc(lepton_frame)
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
                self.recorder.process_frame(self.movement_detected, lepton_frame)
        else:
            self.movement_detected = False
        self.num_frames += 1

    @staticmethod
    def is_affected_by_ffc(lepton_frame):
        if lepton_frame.time_on is None or lepton_frame.last_ffc_time is None:
            return False

        return (
            lepton_frame.time_on - lepton_frame.last_ffc_time
        ) < MotionDetector.FFC_PERIOD
