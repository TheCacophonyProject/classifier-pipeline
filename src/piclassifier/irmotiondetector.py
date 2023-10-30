import numpy as np
import cv2
from .motiondetector import SlidingWindow, MotionDetector
from cptv import Frame
import logging
from track.cliptracker import (
    ClipTracker,
    CVBackground,
    DiffBackground,
    Background,
    get_diff_back_filtered,
)


class RollingBackground(Background):
    AVERAGE_OVER = 1000

    def __init__(self, background_thresh=15):
        self._background = None
        self.frames = 0
        self.background_thresh = background_thresh

    def update_background(self, frame):
        if self._background is None:
            self._background = np.float32(frame.copy())
            return
        if self.frames < Background.AVERAGE_OVER:
            self._background = (self._background * self.frames + frame) / (
                self.frames + 1
            )
        else:
            self._background = (
                self._background * (Background.AVERAGE_OVER - 1) + frame
            ) / (Background.AVERAGE_OVER)

        self.frames += 1

    @property
    def background(self):
        return np.uint8(self._background)

    @property
    def frames(self):
        return min(self._frames, AVERAGE_OVER)

    def compute_filtered(self, thermal):
        return get_diff_back_filtered(
            self.background,
            thermal,
            self.background_thresh,
        )


WINDOW_SIZE = 50
MIN_FRAMES = 10 * 10  # 10 * 10  # 10seconds

THRESHOLD = 12
# maybe put to 12
TRIGGER_FRAMES = 2


class IRMotionDetector(MotionDetector):
    def __init__(self, thermal_config, headers):
        super().__init__(thermal_config, headers)
        self.num_preview_frames = thermal_config.recorder.preview_secs * headers.fps
        self.rgb_window = SlidingWindow(self.num_preview_frames, dtype=np.uint8)
        self.gray_window = SlidingWindow(self.num_preview_frames, dtype=np.uint8)
        # self._background = Background()
        self._background = CVBackground()
        self.kernel_trigger = np.ones(
            (15, 15), "uint8"
        )  # kernel for erosion when not recording
        self.kernel_recording = np.ones(
            (10, 10), "uint8"
        )  # kernel for erosion when recording
        self.movement_detected = False
        self.triggered = 0
        self.show = False
        self.prev_triggered = False

    def disconnected(self):
        self.rgb_window.reset()
        # self.gray_window.reset()
        self.processed = 0

    @property
    def calibrating(self):
        return False

    @property
    def background(self):
        return self._background.background

    def get_kernel(self):
        if self.movement_detected:
            return self.kernel_recording
        else:
            return self.kernel_trigger

    def preview_frames(self):
        return self.rgb_window.get_frames()[:-1]

    def get_recent_frame(self):
        return self.rgb_window.current

    # Processes a frame returning True if there is motion.
    def process_frame(self, frame, force_process=False):
        if self.can_record() or force_process:
            self.rgb_window.add(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.gray_window.add(gray)
            #
            if self.gray_window.oldest is None:
                return False
            learning_rate = 0 if self.movement_detected else -1
            self._background.update_background(gray, learning_rate=learning_rate)
            if self.num_frames > MIN_FRAMES:
                # Filter and get diff from background
                delta = cv2.absdiff(
                    self.gray_window.oldest, gray
                )  # Get delta from current frame and background

                threshold = cv2.threshold(delta, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
                #
                erosion_image = cv2.erode(threshold, self.get_kernel())
                diff_erosion_pixels = len(erosion_image[erosion_image > 0])
                erosion_image = cv2.erode(
                    self._background.compute_filtered(None), self.get_kernel()
                )
                erosion_pixels = len(erosion_image[erosion_image > 0])

                # assert erosion_pixels == self._background.detect_motion()
                # if any have no pixels lets stop motion detection, wiht not updating
                # background when motion is detected if the background actually changes
                # this could cause a problem, this should catch that
                if self.movement_detected:
                    erosion_pixels = min(diff_erosion_pixels, erosion_pixels)
                # to do find a value that suites the number of pixesl we want to move
                # Calculate if there was motion in the current frame
                # TODO Chenage how much ioldests added to the triggered depending on how big the motion is
                self.prev_triggered = erosion_pixels > 0
                if erosion_pixels > 0:
                    self.triggered += 1
                    self.triggered = min(self.triggered, 30)
                else:
                    self.triggered -= 1
                    self.triggered = max(self.triggered, 0)
                # Check if motion has started or ended
                if not self.movement_detected and self.triggered >= TRIGGER_FRAMES:
                    self.movement_detected = True

                elif self.movement_detected and self.triggered <= 0:
                    self.movement_detected = False
        else:
            self.rgb_window.update_current_frame(frame)
        self.num_frames += 1
        return self.movement_detected

    @property
    def temp_thresh(self):
        return None
