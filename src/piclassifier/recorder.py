from abc import ABC, abstractmethod
from datetime import datetime
import logging
import multiprocessing
import shutil
import time
from pathlib import Path

from ml_tools.logs import init_logging

TEMP_DIR = "temp"


class Recorder(ABC):
    def __init__(
        self,
        thermal_config,
        headers,
        name,
        file_extention,
        constant_recorder=False,
        on_recording_stopping=None,
        file_suffix=None,
    ):
        self.file_suffix = file_suffix
        self.file_extention = file_extention
        self.name = name
        self.constant_recorder = constant_recorder
        self.location_config = thermal_config.location
        self.device_config = thermal_config.device
        self.output_dir = Path(thermal_config.recorder.output_dir)
        if constant_recorder:
            self.output_dir = self.output_dir / "constant-recordings"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.temp_dir = self.output_dir / TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.motion = thermal_config.motion
        self.preview_secs = thermal_config.recorder.preview_secs
        self.filename = None
        self.recording = False
        self.frames = 0
        self.headers = headers
        self.min_disk_space = thermal_config.recorder.min_disk_space
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

    def can_record(self, frame_time):
        stat = shutil.disk_usage(self.output_dir)
        free = stat[2] * 0.000001
        if free < self.min_disk_space:
            logging.warn(
                "%s cannot record as only have %s MB and need %s MB",
                self.name,
                free,
                self.min_disk_space,
            )
        return free > self.min_disk_space

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
            self.rec_time / self.frames if self.frames > 0 else 0,
        )
        self.rec_time = 0
        self.write_until = 0
        # write metadata first
        logging.info("Stopped %s", final_name)
        if self.on_recording_stopping is not None:
            self.on_recording_stopping(final_name)

        self.filename.rename(final_name)

    def delete_excess(self):
        stat = shutil.disk_usage(self.output_dir)
        free_percent = stat[2] / stat[0]
        if free_percent >= 0.3:
            return
        recordings = list(self.output_dir.glob(f"*{self.file_extention}"))
        recordings.sort()
        if len(recordings) == 0:
            return
        while free_percent < 0.3 and len(recordings) > 0:
            logging.info("Deleting %s", recordings[0])
            recordings[0].unlink()
            meta = recordings[0].with_suffix(".txt")
            if meta.exists:
                logging.info("Deleting %s", meta)
                meta.unlink()
            recordings = recordings[1:]
            stat = shutil.disk_usage(self.output_dir)
            free_percent = stat[2] / stat[0]

    def start_recording(
        self, background_frame, preview_frames, temp_thresh, frame_time
    ):
        if self.constant_recorder:
            self.delete_excess()

        start = time.time()
        if self.recording:
            logging.warn("%s Already recording, stop recording first", self.name)
            return False

        if not self.can_record(frame_time):
            # logging.warn("%s Cannot record", self.name)
            return False
        self.frames = 0

        self.filename = self.new_temp_name(frame_time)
        started = self.new_recording(
            background_frame, preview_frames, temp_thresh, frame_time
        )
        if not started:
            return False
        self.rec_time = time.time() - start
        self.write_until = self.frames + self.min_frames
        self.recording = True

        logging.info(
            "%s recording %s started temp_thresh: %s",
            self.name,
            self.filename,
            temp_thresh,
        )

        return True

    def new_temp_name(self, frame_time):
        file_name = datetime.fromtimestamp(frame_time).strftime("%Y%m%d-%H%M%S.%f")
        if self.file_suffix is not None:
            file_name = f"{file_name}{self.file_suffix}"
        return self.temp_dir / f"{file_name}{self.file_extention}"

    @abstractmethod
    def new_recording(
        self, background_frame, preview_frames, temp_thresh, frame_time
    ): ...

    @abstractmethod
    def final_name(self): ...
