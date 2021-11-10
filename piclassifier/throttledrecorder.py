import time
import logging
from piclassifier.recorder import Recorder
from piclassifier.cptvrecorder import CPTVRecorder
from piclassifier.eventreporter import throttled_event


class ThrottledRecorder(Recorder):
    def __init__(self, thermal_config, headers, on_recording_stopping):
        self.bucket_size = thermal_config.throttler.bucket_size * headers.fps
        self.throttling = False
        self.tokens = self.bucket_size
        self.recorder = CPTVRecorder(thermal_config, headers, on_recording_stopping)
        self.last_rec = None
        self.last_motion = None
        self.fps = headers.fps
        self.no_motion = thermal_config.throttler.no_motion * headers.fps
        self.max_throttling_seconds = (
            thermal_config.throttler.max_throttling_minutes * 60
        )
        self.min_recording = self.recorder.min_frames
        self.throttled_at = None

    @property
    def recording(self):
        return self.recorder.recording

    def force_stop(self):
        self.recorder.force_stop()

    def process_frame(self, movement_detected, cptv_frame):
        if movement_detected:
            self.last_motion = time.time()
        self.recorder.process_frame(movement_detected, cptv_frame)
        self.take_token()
        if self.throttling:
            logging.info("Throttling recording")
            self.stop_recording()

    def update_tokens(self):
        if self.last_motion is None:
            return

        update_from = self.last_motion
        if not self.last_rec and self.last_rec > self.last_motion:
            update_from = self.last_motion

        since_motion = time.time() - update_from
        # if we have been throttled wait for no motion before adding any tokens back
        if self.throttling:
            since_throttle = time.time() - self.throttled_at
            if (
                self.max_throttling_seconds is None
                or since_throttle < self.max_throttling_seconds
            ):
                since_motion -= self.no_motion
                logging.debug(
                    "Updating tokens %s seconds since motion", round(since_motion)
                )
                if since_motion < 0:
                    return
            else:
                # give it a few tokens to get going
                self.tokens = self.min_frames // 2
                logging.info(
                    "Giving a few free tokens %s has been %s seconds since motion",
                    self.tokens,
                    round(since_motion),
                )

        else:
            logging.debug(
                "Updating tokens %s seconds since motion has earnt %s tokens",
                round(since_motion),
                since_motion * self.fps,
            )
            self.tokens += since_motion * self.fps
        self.throttling = False
        self.throttled_at = None
        self.tokens = max(self.tokens, self.bucket_size)

    def start_recording(self, background_frame, preview_frames, temp_thresh):
        logging.debug("Attempting rec have %s tokens", self.tokens)
        self.update_tokens()
        self.last_motion = time.time()
        if self.throttling or self.tokens < self.min_recording:
            return False
        self.recorder.start_recording(background_frame, preview_frames, temp_thresh)
        return True

    def stop_recording(self):
        self.last_rec = time.time()
        self.recorder.stop_recording()

    def take_token(self):
        self.tokens -= 1
        if self.tokens == 0:
            logging.info("Throttling")
            self.throttling = True
            self.throttled_at = time.time()
            throttled_event()
