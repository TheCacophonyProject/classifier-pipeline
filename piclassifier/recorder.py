

from abc import ABC, abstractmethod


class Recorder(ABC):

    @abstractmethod
    def process_frame(self, movement_detected, cptv_frame):
        """The function to process frame"""
        ...

    @abstractmethod
    def start_recording(self, background_frame, preview_frames, temp_thresh):
        """The function to start rec."""
        ...
    @abstractmethod
    def stop_recording(self):
        """The function to stop rec."""
        ...

    @property
    @abstractmethod
    def recording(self):
        """Recording property."""
        ...
