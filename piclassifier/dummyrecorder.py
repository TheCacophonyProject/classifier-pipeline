import logging
from piclassifier.recorder import Recorder


class DummyRecorder(Recorder):
    def __init__(self, thermal_config, headers, on_recording_stopping=None):
        self.recording = False
        self.frames = 0
        self.min_frames = thermal_config.recorder.min_secs * headers.fps
        self.max_frames = thermal_config.recorder.max_secs * headers.fps

    def process_frame(self, movement_detected, cptv_frame, received_at):
        if self.recording:
            self.write_frame(cptv_frame)
            if movement_detected:
                self.write_until = self.frames + self.min_frames
            elif self.has_minimum():
                self.stop_recording(received_at)
                return

            if self.frames == self.max_frames:
                self.stop_recording(received_at)
        return

    def has_minimum(self):
        return self.frames > self.write_until

    def write_frame(self, cptv_frame):
        self.frames += 1

    def start_recording(
        self, background_frame, preview_frames, temp_thresh, frame_time
    ):
        if self.recording:
            return False
        self.recording = True
        self.frames = 0
        for f in preview_frames:
            self.write_frame(f)
        self.write_until = self.frames + self.min_frames
        logging.info("Dummy recording started temp_thresh: %s", temp_thresh)

        return True

    def stop_recording(self, frame_time):
        self.recording = False
        self.write_until = 0
        logging.info("Dummy recording ended  %s frames", self.frames)

    def force_stop(self):
        self.recording = False
        self.write_until = 0
