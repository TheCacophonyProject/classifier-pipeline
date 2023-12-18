import numpy as np
import cv2
from .motiondetector import SlidingWindow, MotionDetector
from cptv import Frame
import logging
from track.cliptracker import (
    CVBackground,
)
from datetime import timedelta
from ml_tools.tools import Rectangle
from .cptvmotiondetector import is_affected_by_ffc

from ml_tools.imageprocessing import normalize

WINDOW_SIZE = 50
MIN_FRAMES = 10  # 10 * 10  # 10seconds

THRESHOLD = 12
# maybe put to 12
TRIGGER_FRAMES = 2


class CVMotionDetector(MotionDetector):
    def __init__(self, thermal_config, headers, tracking_alg="knn", detect_after=None):
        super().__init__(thermal_config, headers)
        self.num_frames = 0
        self.headers = headers
        self.config = thermal_config.motion
        self.location_config = thermal_config.location
        self.num_preview_frames = thermal_config.recorder.preview_secs * headers.fps
        self.compare_gap = self.config.frame_compare_gap + 1
        edge = self.config.edge_pixels
        self.tracking_alg = tracking_alg
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
        self.background_alg = CVBackground(self.tracking_alg)

        self.movement_detected = False

        self.triggered = 0

        self.ffc_affected = False
        if detect_after is None:
            self.detect_after = self.thermal_window.size * 2
        else:
            self.detect_after = detect_after
        self.norm_min = None
        self.norm_max = None

    @property
    def background(self):
        return self.background_alg.background

    def get_recent_frame(self):
        return self.thermal_window.current

    def disconnected(self):
        self.thermal_window.reset()
        self.processed = 0

    @property
    def calibrating(self):
        return self.ffc_affected

    def preview_frames(self):
        return self.thermal_window.get_frames()[:-1]

    def get_recent_frame(self):
        return self.thermal_window.current

    # Processes a frame returning True if there is motion.
    def process_frame(self, frame, force_process=False):
        self.num_frames += 1
        self.ffc_affected = is_affected_by_ffc(frame)
        self.update_norms(frame.pix)

        thermal, _ = normalize(
            frame.pix.copy(), min=self.norm_min, max=self.norm_max, new_max=255
        )
        fg = self.background_alg.update_background(thermal)
        self.thermal_window.add(frame)

        fg_cropped = self.crop_rectangle.subimage(fg)
        motion_pixels = len(fg_cropped[fg_cropped > 0])
        # print(motion_pixels, self.num_frames)
        # cv2.imshow("a", np.uint8(thermal))

        # cv2.imshow("f", np.uint8(fg_cropped))
        # cv2.imshow("b", np.uint8(self.background_alg.background))
        # cv2.moveWindow("b", 600, 0)

        # cv2.moveWindow("f", 0, 0)
        # cv2.waitKey()

        if self.num_frames < MIN_FRAMES:
            return False
        if motion_pixels > self.config.count_thresh:
            self.triggered += 1
            if not self.can_record() and not force_process:
                return False
            if (
                not self.config.one_diff_only
                and self.triggered < 2
                or self.num_frames < MIN_FRAMES
            ):
                return False
            if not self.movement_detected:
                logging.info(
                    "%s MotionDetector motion detected %s  count %s triggered consec %s ",
                    timedelta(seconds=self.num_frames / 9),
                    self.processed,
                    motion_pixels,
                    self.triggered,
                )
            self.movement_detected = True
            return True
        else:
            self.triggered = 0
        if self.movement_detected:
            logging.info(
                "{} MotionDetector motion stopped thresh {} count {}".format(
                    timedelta(seconds=self.num_frames / 9),
                    self.temp_thresh,
                    motion_pixels,
                )
            )
        self.movement_detected = False
        return False

    @property
    def temp_thresh(self):
        return None
