from abc import ABC, abstractmethod


class Recorder(ABC):
    @abstractmethod
    def process_frame(self, movement_detected, cptv_frame, received_at):
        ...

    @abstractmethod
    def start_recording(
        self, background_frame, preview_frames, temp_thresh, frame_time
    ):
        ...

    @abstractmethod
    def stop_recording(self, frame_time):
        ...

    @abstractmethod
    def force_stop(self):
        ...
