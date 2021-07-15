from threading import Lock
from datetime import timedelta
import logging
import time
import numpy as np
import time
from ml_tools.tools import Rectangle
from .processor import Processor


class SlidingWindow:
    def __init__(self, shape, dtype):
        self.lock = Lock()
        self.frames = np.empty(shape, dtype)
        self.last_index = None
        self.size = len(self.frames)
        self.oldest_index = None

    def update_current_frame(self, frame):
        with self.lock:
            if self.last_index is None:
                self.oldest_index = 0
                self.last_index = 0
            self.frames[self.last_index] = frame

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

    def __init__(self, thermal_config, dynamic_thresh, recorder, headers):
        self._output_dir = thermal_config.recorder.output_dir
        self.headers = headers
        self.config = thermal_config.motion
        self.location_config = thermal_config.location
        self.preview_frames = thermal_config.recorder.preview_secs * headers.fps
        self.compare_gap = self.config.frame_compare_gap + 1
        edge = self.config.edge_pixels
        self.min_frames = thermal_config.recorder.min_secs * headers.fps
        self.max_frames = thermal_config.recorder.max_secs * headers.fps
        self.clipped_window = SlidingWindow(
            (self.compare_gap, headers.res_y - edge * 2, headers.res_x - edge * 2),
            np.int32,
        )
        self.diff_window = SlidingWindow(
            (self.compare_gap, headers.res_y - edge * 2, headers.res_x - edge * 2),
            np.int32,
        )

        self.thermal_window = SlidingWindow(
            (self.preview_frames, headers.res_y, headers.res_x), np.uint16
        )
        self.processed = 0
        self.num_frames = 0
        self.thermal_thresh = 0
        self.background = None
        self.last_background_change = None
        self.background_weight = np.zeros((headers.res_y, headers.res_x))
        self.background_weight[:] = MotionDetector.BACKGROUND_WEIGHTING_PER_FRAME
        self.movement_detected = False
        self.dynamic_thresh = dynamic_thresh
        self.temp_thresh = self.config.temp_thresh
        self.crop_rectangle = Rectangle(
            edge, edge, headers.res_x - 2 * edge, headers.res_y - 2 * edge
        )
        self.rec_window = thermal_config.recorder.rec_window
        self.use_sunrise = self.rec_window.use_sunrise_sunset()
        self.last_sunrise_check = None
        self.location = None
        self.sunrise = None
        self.sunset = None
        self.recording = False

        if self.rec_window.use_sunrise_sunset():
            self.rec_window.set_location(
                *self.location_config.get_lat_long(use_default=True),
                self.location_config.altitude,
            )

        self.recorder = recorder
        self.ffc_affected = False

    def calc_temp_thresh(self, thermal_frame, prev_ffc):
        logging.debug(
            "frame pixels are %s back max %s",
            np.amax(thermal_frame),
            np.amax(self.background),
        )
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
                new_weights = self.background_weight[self.background != new_background]
                # these have changed so reset weighting
                new_weights[:] = MotionDetector.BACKGROUND_WEIGHTING_PER_FRAME

                # these havent changed, increase weighting
                if self.processed % MotionDetector.BACKGROUND_WEIGHT_EVERY == 0:
                    self.background_weight[
                        self.background == new_background
                    ] *= MotionDetector.BACKGROUND_WEIGHTING_PER_FRAME
                back_changed = new_weights.size > 0
                # np.amax(self.background != new_background)

            if back_changed:
                self.last_background_change = self.processed
                self.background = new_background

                old_temp = self.temp_thresh
                self.temp_thresh = int(round(np.average(self.background)))
                if self.temp_thresh != old_temp:
                    logging.debug(
                        "{} MotionDetector temp threshold changed from {} to {} new background average is {}".format(
                            self.num_frames,
                            old_temp,
                            self.temp_thresh,
                            np.average(self.background),
                        )
                    )
                    temp_changed = True
                    # self.background_weight[
                    #     :
                    # ] = MotionDetector.BACKGROUND_WEIGHTING_PER_FRAME

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
                delta_combined = delta_frame2 + delta_frame
                diff = len(
                    delta_combined[delta_combined == self.config.delta_thresh * 2]
                )
            else:
                delta_frame[
                    delta_frame >= self.config.delta_thresh
                ] = self.config.delta_thresh
                diff = 0

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

    def get_recent_frame(self):
        return self.thermal_window.current_copy()

    def can_record(self):
        return self.rec_window.inside_window()

    def disconnected(self):
        self.clipped_window.reset()
        self.thermal_window.reset()
        self.diff_window.reset()
        self.processed = 0
        self.recorder.force_stop()

    def process_frame(self, cptv_frame):
        if self.can_record() or (self.recorder and self.recorder.recording):

            cropped_frame = self.crop_rectangle.subimage(cptv_frame.pix)
            frame = np.int32(cropped_frame)
            prev_ffc = self.ffc_affected
            self.ffc_affected = is_affected_by_ffc(cptv_frame)
            if not self.ffc_affected:
                self.thermal_window.add(cptv_frame.pix)
                if self.background is None:
                    self.background = cptv_frame.pix
                    logging.debug(
                        "Setting background with %s", np.amax(cropped_frame.pix)
                    )
                    import matplotlib.pyplot as plt

                    imgplot = plt.imshow(self.background)
                    plt.savefig("background{}-{}".format(time.time(), self.processed))
                    self.last_background_change = self.processed
                else:
                    self.calc_temp_thresh(cptv_frame.pix, prev_ffc)

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
                    self.movement_detected, cptv_frame, self.temp_thresh
                )
        else:
            self.thermal_window.update_current_frame(cptv_frame.pix)
            self.movement_detected = False

        self.num_frames += 1

    def skip_frame(self):
        return

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def res_x(self):
        return self.headers.res_x

    @property
    def res_y(self):
        return self.headers.res_y


def is_affected_by_ffc(cptv_frame):
    if cptv_frame.time_on is None or cptv_frame.last_ffc_time is None:
        return False

    return (cptv_frame.time_on - cptv_frame.last_ffc_time) < MotionDetector.FFC_PERIOD
