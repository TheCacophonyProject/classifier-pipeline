import numpy as np

from .motiondetector import SlidingWindow, MotionDetector


class Background:
    AVERAGE_OVER = 1000

    def __init__(self):
        self._background = None
        self.frames = 0

    def process_frame(self, frame):
        self.frames += 1
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


class IRMotionDetector(MotionDetector):
    def __init__(self, thermal_config, headers):
        super().__init__(thermal_config, headers)

        self.rgb_window = SlidingWindow(WINDOW_SIZE, np.uint8)
        self.frames_gray = SlidingWindow(WINDOW_SIZE, np.uint8)
        self._background = Background()
        self.kernel_trigger = np.ones(
            (15, 15), "uint8"
        )  # kernel for erosion when not recording
        self.kernel_recording = np.ones(
            (10, 10), "uint8"
        )  # kernel for erosion when recording
        self.movement_detected = False
        self.triggered = 0
        self.show = False

    @property
    def background(self):
        return self._background.background

    def get_kernel(self):
        if self.movement_detected:
            return self.kernel_recording
        else:
            return self.kernel_trigger

    def preview_frames(self):
        return self.gray_window.get_frames()[:-1]

    def get_recent_frame(self):
        return self.rgb_window.current

    # Processes a frame returning True if there is motion.
    def process_frame(self, frame, force_process=False):
        if self.can_record() or force_process:

            self.rgb_window.add(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.gray_window.add(frame)

            if self.gray_window.oldest is None:
                return False

            # Filter and get diff from background
            delta = cv2.absdiff(
                self.gray_window.oldest, frame
            )  # Get delta from current frame and background
            threshold = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]

            erosion_image = cv2.erode(threshold, self.get_kernel())
            erosion_pixels = len(erosion_image[erosion_image > 0])
            # to do find a value that suites the number of pixesl we want to move
            self.gray_window.add(frame)
            self._background.process_frame(frame)
            # Calculate if there was motion in the current frame
            # TODO Chenage how much ioldests added to the triggered depending on how big the motion is
            if erosion_pixels > 0:
                self.triggered += 1
                self.triggered = min(self.triggered, 30)
            else:
                self.triggered -= 1
                self.triggered = max(self.triggered, 0)

            # Check if motion has started or ended
            if not self.motion and self.triggered > 10:
                self.movement_detected = True

            elif self.motion and self.triggered <= 0:
                self.movement_detected = False
        else:
            self.rgb_window.update_current_frame(frame)

        self.num_frames += 1
        return self.motion
