from abc import ABC, abstractmethod
from threading import Lock
from datetime import timedelta
import logging
import time
import numpy as np


class SlidingWindow:
    def __init__(self, shape, dtype):
        self.lock = Lock()
        # if dtype == "O":
        self.frames = [None] * shape
        # else:
        # self.frames = np.empty(shape, dtype)
        self.last_index = None
        self.size = len(self.frames)
        self.oldest_index = None
        self.non_ffc_index = None
        self.ffc = False

    def update_current_frame(self, frame, ffc=False):
        with self.lock:
            if self.last_index is None:
                self.oldest_index = 0
                self.last_index = 0
                if not ffc:
                    self.non_ffc_index = self.oldest_index
            if not ffc and self.ffc:
                self.non_ffc_index = self.last_index

            self.frames[self.last_index] = frame
            self.ffc = ffc

    @property
    def current(self):
        with self.lock:
            if self.last_index is not None:
                return self.frames[self.last_index]
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
    def oldest_nonffc(self):
        with self.lock:
            if self.non_ffc_index is not None:
                return self.frames[self.non_ffc_index]
            return None

    @property
    def oldest(self):
        with self.lock:
            if self.oldest_index is not None:
                return self.frames[self.oldest_index]
            return None

    def add(self, frame, ffc=False):
        with self.lock:
            if self.last_index is None:
                self.oldest_index = 0
                self.frames[0] = frame
                self.last_index = 0
                if not ffc:
                    self.non_ffc_index = self.oldest_index
            else:
                new_index = (self.last_index + 1) % self.size
                if new_index == self.oldest_index:
                    if self.oldest_index == self.non_ffc_index and not ffc:
                        self.non_ffc_index = (self.oldest_index + 1) % self.size
                    self.oldest_index = (self.oldest_index + 1) % self.size
                self.frames[new_index] = frame
                self.last_index = new_index
            if not ffc and self.ffc:
                self.non_ffc_index = self.last_index
            self.ffc = ffc

    def reset(self):
        with self.lock:
            self.last_index = None
            self.oldest_index = None


MIN_ADJUST = 0.05
ADJUST_FRAMES = 9 * 60 * 60
# adjust the norm min and max by MIN_ADJUST percent of the frame min
# every ADJUST_FRAMES
# i.e if current frame min is 20K adjust min by 1000  (20000 * 0.05) every (Adjust Frames) hour when comparing
#  to current frame min, this will force the min and max to update occasionally
# rather than getting stuck at the coolest and hottest values (although i don't think this would matter either)


class MotionDetector(ABC):
    def __init__(self, thermal_config, headers):
        self.movement_detected = False
        self.num_frames = 0
        self.rec_window = thermal_config.recorder.rec_window
        self.location_config = thermal_config.location
        self.use_sunrise = self.rec_window.use_sunrise_sunset()
        self.last_sunrise_check = None
        self.location = None
        self.sunrise = None
        self.sunset = None
        self.recording = False
        self.is_normalized = False
        if self.rec_window.use_sunrise_sunset():
            self.rec_window.set_location(
                *self.location_config.get_lat_long(use_default=True),
                self.location_config.altitude,
            )
        self.headers = headers
        self.norm_min = None
        self.norm_max = None
        self.norm_adjust_amount = 0
        self.norm_adjust = 0

    def get_background(self, original=True):
        if not self.is_normalized:
            return self.background
        else:
            return (np.float32(self.background) / 255) * (
                self.norm_max - self.norm_min
            ) + self.norm_min

    def update_norms(self, frame):
        f_min = np.amin(frame)
        f_max = np.amax(frame)
        if self.norm_min is None or f_min < (self.norm_min + self.norm_adjust):
            logging.info("Updating norm min from %s to %s", self.norm_min, f_min)
            self.norm_min = f_min
            self.norm_adjust_amount = f_min * MIN_ADJUST / ADJUST_FRAMES
        if self.norm_max is None or f_max > (self.norm_max - self.norm_adjust):
            logging.info("Updating norm min from %s to %s", self.norm_max, f_max)
            self.norm_max = f_max * 1.2
        self.norm_adjust += self.norm_adjust_amount

    @property
    def res_x(self):
        return self.headers.res_x

    @property
    def res_y(self):
        return self.headers.res_y

    @abstractmethod
    def process_frame(self, clipped_frame, received_at=None):
        """Tracker type IR or Thermal"""

    @abstractmethod
    def preview_frames(self):
        """Tracker type IR or Thermal"""

    @abstractmethod
    def get_recent_frame(self):
        """Tracker type IR or Thermal"""

    def can_record(self):
        return self.rec_window.inside_window()

    @abstractmethod
    def disconnected(self):
        """Tracker type IR or Thermal"""

    @abstractmethod
    def calibrating(self):
        """Tracker type IR or Thermal"""

    @property
    @abstractmethod
    def background(self):
        """Tracker type IR or Thermal"""


class WeightedBackground:
    def __init__(
        self, edge_pixels, crop_rectangle, res_x, res_y, weight_add, init_average
    ):
        self.edge_pixels = edge_pixels
        self.crop_rectangle = crop_rectangle
        self._background = None
        self.weight_add = weight_add
        self.background_weight = np.zeros(
            (res_y - edge_pixels * 2, res_x - edge_pixels * 2)
        )
        self.average = init_average

    def process_frame(self, frame):
        if self._background is None:
            res_y, res_x = frame.shape
            self._background = np.empty(
                (res_y + self.edge_pixels * 2, res_x + self.edge_pixels * 2)
            )
            self._background[
                self.edge_pixels : res_y + self.edge_pixels,
                self.edge_pixels : res_x + self.edge_pixels,
            ] = frame
            self.average = np.average(frame)
            self.set_background_edges()

            return
        edgeless_back = self.crop_rectangle.subimage(self.background)
        new_background = np.where(
            edgeless_back < frame - self.background_weight,
            edgeless_back,
            frame,
        )
        # update weighting
        self.background_weight = np.where(
            edgeless_back < frame - self.background_weight,
            self.background_weight + self.weight_add,
            0,
        )
        back_changed = new_background != edgeless_back
        back_changed = np.any(back_changed == True)
        if back_changed:
            edgeless_back[:, :] = new_background
            old_temp = self.average
            self.average = int(round(np.average(edgeless_back)))
            if self.average != old_temp:
                logging.debug(
                    "MotionDetector temp threshold changed from {} to {} ".format(
                        old_temp,
                        self.average,
                    )
                )
            set_background_edges(self._background, self.edge_pixels)

    # def set_background_edges(self):
    #     for i in range(self.edge_pixels):
    #         self._background[i] = self._background[self.edge_pixels]
    #         self._background[-i - 1] = self._background[-self.edge_pixels - 1]
    #         self._background[:, i] = self._background[:, self.edge_pixels]
    #         self._background[:, -i - 1] = self._background[:, -1 - self.edge_pixels]

    def set_background(self, back):
        self._background = back

    @property
    def background(self):
        return self._background


def set_background_edges(frame, edge_pixels):
    for i in range(edge_pixels):
        frame[i] = frame[edge_pixels]
        frame[-i - 1] = frame[-edge_pixels - 1]
        frame[:, i] = frame[:, edge_pixels]
        frame[:, -i - 1] = frame[:, -1 - edge_pixels]
