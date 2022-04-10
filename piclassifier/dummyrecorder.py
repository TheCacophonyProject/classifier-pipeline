from piclassifier.recorder import Recorder


class DummyRecorder(Recorder):
    def __init__(self, thermal_config, headers, on_recording_stopping=None):
        self.recording = False
        self.frames = 0
        self.min_frames = thermal_config.recorder.min_secs * headers.fps
        self.max_frames = thermal_config.recorder.max_secs * headers.fps

    def process_frame(self, movement_detected, cptv_frame):
        return

    def start_recording(
        self, background_frame, preview_frames, temp_thresh, frame_time
    ):
        if self.recording:
            return False
        self.recording = True
        self.frames = 0
        self.write_until = self.frames + self.min_frames
        return True

    def stop_recording(self, frame_time):
        self.recording = False
        self.write_until = 0

    def force_stop(self):
        self.recording = False
        self.write_until = 0
