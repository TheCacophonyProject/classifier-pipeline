import numpy as np
from datetime import timedelta
import time
import logging

from .motiondetector import SlidingWindow, MotionDetector, WeightedBackground
from ml_tools.tools import Rectangle


class CPTVMotionDetector(MotionDetector):
    FFC_PERIOD = timedelta(seconds=9.9)
    BACKGROUND_WEIGHT_ADD = 0.1

    def __init__(self, thermal_config, dynamic_thresh, headers, detect_after=None):
        super().__init__(thermal_config, headers)
        self.headers = headers
        if headers.model and headers.model.lower() == "lepton3.5":
            CPTVMotionDetector.BACKGROUND_WEIGHT_ADD = 1
        self.config = thermal_config.motion
        self.location_config = thermal_config.location
        self.num_preview_frames = thermal_config.recorder.preview_secs * headers.fps
        self.compare_gap = self.config.frame_compare_gap + 1
        edge = self.config.edge_pixels
        self.min_frames = thermal_config.recorder.min_secs * headers.fps
        self.max_frames = thermal_config.recorder.max_secs * headers.fps

        if not self.config.one_diff_only:
            self.diff_window = SlidingWindow(
                self.compare_gap,
                np.int32,
            )

        self.thermal_window = SlidingWindow(self.num_preview_frames + 1, "O")
        self.processed = 0
        self.thermal_thresh = 0
        self.crop_rectangle = Rectangle(
            edge, edge, headers.res_x - 2 * edge, headers.res_y - 2 * edge
        )
        self._background = WeightedBackground(
            edge,
            self.crop_rectangle,
            self.res_x,
            self.res_y,
            CPTVMotionDetector.BACKGROUND_WEIGHT_ADD,
            self.config.temp_thresh,
        )
        self.movement_detected = False
        self.dynamic_thresh = dynamic_thresh

        self.triggered = 0

        self.ffc_affected = False
        if detect_after is None:
            self.detect_after = self.thermal_window.size * 2
        else:
            self.detect_after = detect_after

    @property
    def calibrating(self):
        return self.ffc_affected

    def preview_frames(self):
        return self.thermal_window.get_frames()[:-1]

    @property
    def temp_thresh(self):
        return self._background.average

    def detect(self, clipped_frame, received_at=None):
        oldest = self.crop_rectangle.subimage(self.thermal_window.oldest_nonffc.pix)
        oldest = np.clip(oldest, a_min=self.temp_thresh, a_max=None)
        clipped_frame = np.clip(clipped_frame, a_min=self.temp_thresh, a_max=None)
        delta_frame = clipped_frame - oldest
        if not self.config.warmer_only:
            delta_frame = abs(delta_frame)
        if self.config.one_diff_only:
            diff = len(delta_frame[delta_frame > self.config.delta_thresh])
        else:
            if self.processed > 2:
                delta_frame2 = self.diff_window.oldest_nonffc
                delta_frame[delta_frame >= self.config.delta_thresh] = (
                    self.config.delta_thresh
                )
                delta_combined = delta_frame2 + delta_frame
                diff = len(
                    delta_combined[delta_combined == self.config.delta_thresh * 2]
                )
            else:
                delta_frame[delta_frame >= self.config.delta_thresh] = (
                    self.config.delta_thresh
                )
                diff = 0

            self.diff_window.add(delta_frame, self.ffc_affected)
        if diff > self.config.count_thresh:
            if not self.movement_detected:
                logging.debug(
                    "{} MotionDetector motion detected {} thresh {} count {} ".format(
                        timedelta(seconds=self.num_frames / 9),
                        self.processed,
                        self.temp_thresh,
                        diff,
                    )
                )
            return True

        if self.movement_detected:
            logging.debug(
                "{} MotionDetector motion stopped thresh {} count {}".format(
                    timedelta(seconds=self.num_frames / 9),
                    self.temp_thresh,
                    diff,
                )
            )
        return False

    @property
    def background(self):
        return self._background.background

    def get_recent_frame(self):
        return self.thermal_window.current

    def disconnected(self):
        self.thermal_window.reset()
        if not self.config.one_diff_only:
            self.diff_window.reset()
        self.processed = 0

    def process_frame(self, cptv_frame, force_process=False):
        prev_ffc = self.ffc_affected
        self.ffc_affected = is_affected_by_ffc(cptv_frame)
        if self.can_record() or force_process:
            self.thermal_window.add(cptv_frame, self.ffc_affected)

            cropped_frame = np.int32(self.crop_rectangle.subimage(cptv_frame.pix))
            if not self.ffc_affected:
                self._background.process_frame(cropped_frame)
            if self.ffc_affected or prev_ffc:
                logging.debug("{} MotionDetector FFC".format(self.num_frames))
                self.movement_detected = False
                self.triggered = 0
                if prev_ffc:
                    self.thermal_window.non_ffc_index = self.thermal_window.last_index
            elif self.processed > self.detect_after:
                movement = self.detect(cropped_frame)
                if movement:
                    self.triggered += 1
                else:
                    self.triggered = 0
                if self.triggered >= self.config.trigger_frames:
                    self.movement_detected = True
                else:
                    self.movement_detected = False
            self.processed += 1
        else:
            self.thermal_window.update_current_frame(cptv_frame, self.ffc_affected)
            self.movement_detected = False

        self.num_frames += 1
        return self.movement_detected

    def skip_frame(self):
        return


def is_affected_by_ffc(cptv_frame):
    if cptv_frame.time_on is None or cptv_frame.last_ffc_time is None:
        return False

    return (
        cptv_frame.time_on - cptv_frame.last_ffc_time
    ) < CPTVMotionDetector.FFC_PERIOD
