from abc import ABC, abstractmethod
import logging
import multiprocessing
import shutil
import time
from pathlib import Path

from ml_tools.logs import init_logging


class Recorder(ABC):
    def __init__(
        self,
        thermal_config,
        headers,
        name,
        constant_recorder,
        on_recording_stopping=None,
    ):
        self.name = name
        self.constant_recorder = constant_recorder
        self.location_config = thermal_config.location
        self.device_config = thermal_config.device
        self.output_dir = Path(thermal_config.recorder.output_dir)
        if constant_recorder:
            self.output_dir = self.output_dir / "constant-recordings"
            self.output_dir.mkdirs(parents=True, exist_ok=True)
        self.motion = thermal_config.motion
        self.preview_secs = thermal_config.recorder.preview_secs
        self.filename = None
        self.recording = False
        self.frames = 0
        self.headers = headers
        self.min_frames = thermal_config.recorder.min_secs * headers.fps
        self.max_frames = thermal_config.recorder.max_secs * headers.fps
        self.min_recording = self.preview_secs * headers.fps + self.min_frames
        self.write_until = 0
        self.rec_time = 0
        self.on_recording_stopping = on_recording_stopping
        self.frame_q = multiprocessing.Queue()
        self.rec_p = None

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

    def has_minimum(self):
        return self.frames >= self.write_until

    def write_frame(self, frame):
        start = time.time()
        self.frame_q.put(frame)
        # self.writer.next_frame(frame)
        self.frames += 1
        self.rec_time += time.time() - start

    def can_record(self):
        if self.constant_recorder:
            stat = shutil.disk_usage(self.output_dir)

    def force_stop(self):
        if not self.recording:
            return
        if self.frames > self.min_recording:
            self.stop_recording(time.time())
        else:
            logging.info(
                "%s Recording stopped early deleting short recording", self.name
            )
            self.delete_recording()

    def delete_recording(self):
        if self.recording:
            self.frame_q.put(0)
            self.rec_p.join()
            self.frame_q = multiprocessing.Queue()
            self.rec_p = None
            self.recording = False
        self.filename.unlink()

    def stop_recording(self, frame_time):
        start = time.time()
        self.rec_time += time.time() - start
        self.recording = False
        final_name = self.final_name()
        logging.info("%s Waiting for recorder to finish", self.name)
        self.frame_q.put(0)
        self.rec_p.join()
        self.frame_q = multiprocessing.Queue()
        self.rec_p = None
        logging.info(
            "%s recording %s ended %s frames %s time recording %s per frame ",
            self.name,
            final_name,
            self.frames,
            self.rec_time,
            self.rec_time / self.frames,
        )
        self.rec_time = 0
        self.write_until = 0
        # write metadata first
        if self.on_recording_stopping is not None:
            self.on_recording_stopping(final_name)

        self.filename.rename(final_name)

    @abstractmethod
    def final_name(self):
        ...

    @abstractmethod
    def start_recording(
        self, background_frame, preview_frames, temp_thresh, frame_time
    ):
        ...
