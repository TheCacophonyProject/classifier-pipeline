import time
import logging
from piclassifier.recorder import Recorder
from piclassifier.cptvrecorder import CPTVRecorder
from piclassifier.eventreporter import throttled_event
from datetime import datetime


class ThrottledRecorder(Recorder):
    def __init__(self, recorder, thermal_config, headers, on_recording_stopping):
        self.bucket_size = thermal_config.throttler.bucket_size * headers.fps
        self.throttling = False
        self.tokens = self.bucket_size
        self.recorder = recorder
        self.last_rec = None
        self.last_motion = None
        self.fps = headers.fps
        self.no_motion = thermal_config.throttler.no_motion
        self.max_throttling_seconds = (
            thermal_config.throttler.max_throttling_minutes * 60
        )
        self.min_recording = self.recorder.min_frames
        self.throttled_at = None
        self.constant_recorder = False

    def final_name(self):
        return self.recorder.final_name()

    @property
    def name(self):
        return self.recorder.name

    @property
    def filename(self):
        return self.recorder.filename

    @filename.setter
    def filename(self, a):
        self.recorder.filename = a

    @property
    def recording(self):
        return self.recorder.recording

    @recording.setter
    def recording(self, a):
        self.recorder.recording = a

    @property
    def write_until(self):
        return self.recorder.write_until

    @write_until.setter
    def write_until(self, a):
        self.recorder.write_until = a

    @property
    def min_frames(self):
        return self.recorder.min_frames

    def force_stop(self):
        if self.recorder.recording:
            self.last_rec = time.time()
        self.recorder.force_stop()

    def process_frame(self, movement_detected, cptv_frame, received_at):
        if movement_detected:
            self.last_motion = received_at

        was_recording = self.recorder.recording
        self.recorder.process_frame(movement_detected, cptv_frame, received_at)
        self.take_token(received_at)
        if was_recording and not self.recorder.recording:
            self.last_rec = received_at

        if self.throttling and self.recorder.recording:
            logging.info("Throttling recording")
            self.stop_recording(received_at)

    def update_tokens(self, frame_time):
        if self.last_motion is None:
            return

        update_from = self.last_motion
        if self.last_rec and self.last_rec > self.last_motion:
            update_from = self.last_rec

        since_motion = frame_time - update_from
        # if we have been throttled wait for no motion before adding any tokens back
        if self.throttling:
            since_throttle = frame_time - self.throttled_at
            logging.debug(
                "Updating tokens %s seconds since motion with no motion %s since throttle %s, max throttle %s",
                round(since_motion),
                self.no_motion,
                datetime.fromtimestamp(since_throttle),
                self.max_throttling_seconds,
            )
            since_motion -= self.no_motion

            if since_motion < 0:
                if (
                    self.max_throttling_seconds
                    and since_throttle >= self.max_throttling_seconds
                ):
                    self.tokens = self.recorder.min_frames // 2
                    logging.info(
                        "Giving a few free tokens %s has been %s seconds since motion",
                        self.tokens,
                        round(since_motion),
                    )
                else:
                    return
            else:
                self.tokens += since_motion * self.fps
                logging.debug(
                    "Updating tokens %s seconds since motion has earnt %s tokens",
                    round(since_motion),
                    since_motion * self.fps,
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
        self.tokens = int(self.tokens)
        self.tokens = min(self.tokens, self.bucket_size)

    def new_recording(self, background_frame, preview_frames, temp_thresh, frame_time):
        logging.debug("Attempting rec have %s tokens", self.tokens)
        self.update_tokens(frame_time)
        self.last_motion = frame_time
        if self.throttling:
            throttled_event()
            return False
        if self.tokens < self.min_recording:
            self.throttle(frame_time)
            return False
        self.take_token(frame_time, len(preview_frames))
        self.recorder.new_recording(
            background_frame, preview_frames, temp_thresh, frame_time
        )
        return True

    def can_record(self, frame_time):
        prev_throttled = self.throttling
        self.update_tokens(frame_time)
        logging.debug(
            "Attempting rec have %s tokens was throttled %s throttled  %s tokens %s",
            self.tokens,
            prev_throttled,
            self.throttling,
            self.tokens,
        )
        self.last_motion = frame_time
        if self.throttling:
            if not prev_throttled:
                throttled_event()
            return False
        if self.tokens < self.min_recording:
            self.throttle(frame_time)
            return False
        return True

    def stop_recording(self, frame_time):
        if self.recorder.recording:
            self.last_rec = frame_time
            self.recorder.stop_recording(frame_time)

    def new_temp_name(self, frame_time):
        return self.recorder.new_temp_name(frame_time)

    def throttle(self, frame_time):
        logging.info("Throttling")
        self.throttling = True
        self.throttled_at = frame_time
        throttled_event()

    def take_token(self, frame_time, num_tokens=1):
        self.tokens -= num_tokens
        if self.tokens <= 0:
            self.tokens = 0
            self.throttle(frame_time)
