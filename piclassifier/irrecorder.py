from datetime import datetime
import logging
import os
import yaml
from load.cliptrackextractor import ClipTrackExtractor

from datetime import timedelta
import time
import psutil
from piclassifier.recorder import Recorder
import cv2
from pathlib import Path

AVI_TEMP_EXT = ".avi"
TEMP_DIR = "temp"

fourcc = cv2.VideoWriter_fourcc(*"XVID")


class IRRecorder(Recorder):
    def __init__(self, thermal_config, headers, on_recording_stopping=None):
        self.location_config = thermal_config.location
        self.device_config = thermal_config.device
        self.output_dir = Path(thermal_config.recorder.output_dir)
        self.motion = thermal_config.motion
        self.preview_secs = thermal_config.recorder.preview_secs
        self.writer = None
        self.filename = None
        self.recording = False
        self.frames = 0
        self.headers = headers
        self.min_frames = thermal_config.recorder.min_secs * headers.fps
        self.max_frames = thermal_config.recorder.max_secs * headers.fps
        self.res_x = headers.res_x
        self.res_y = headers.res_y
        self.fps = headers.fps
        self.write_until = 0
        self.rec_time = 0
        self.on_recording_stopping = on_recording_stopping
        self.temp_dir = self.output_dir / TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def force_stop(self):
        if not self.recording:
            return
        if self.has_minimum():
            self.stop_recording(time.time())
        else:
            logging.info("Recording stopped early deleting short recording")
            self.delete_recording()

    def process_frame(self, movement_detected, cptv_frame):
        if self.recording:
            self.write_frame(cptv_frame)
            if movement_detected:
                self.write_until = self.frames + self.min_frames
            elif self.has_minimum():

                self.stop_recording(cptv_frame.received_at)
                return

            if self.frames == self.max_frames:
                self.stop_recording(cptv_frame.received_at)

    def has_minimum(self):
        return self.frames > self.write_until

    def start_recording(
        self, background_frame, preview_frames, temp_thresh, frame_time
    ):

        start = time.time()
        if self.recording:
            logging.warn("Already recording, stop recording first")
            return False
        self.frames = 0
        self.filename = new_temp_name(frame_time)

        self.filename = self.temp_dir / self.filename
        self.writer = cv2.VideoWriter(
            str(self.filename), fourcc, self.fps, (self.res_x, self.res_y), 0
        )
        self.writer.write(background_frame)
        default_thresh = self.motion.temp_thresh

        self.recording = True
        for frame in preview_frames:
            self.write_frame(frame)
        self.write_until = self.frames + self.min_frames

        logging.info("recording %s started temp_thresh: %d", self.filename, temp_thresh)

        self.rec_time += time.time() - start
        return True

    def write_frame(self, cptv_frame):
        start = time.time()
        self.writer.write(cptv_frame.pix)
        self.frames += 1
        self.rec_time += time.time() - start

    def stop_recording(self, frame_time):
        start = time.time()
        self.rec_time += time.time() - start
        self.recording = False
        logging.info(
            "recording ended %s frames %s time recording %s per frame ",
            self.frames,
            self.rec_time,
            self.rec_time / self.frames,
        )
        self.rec_time = 0

        self.write_until = 0
        if self.writer is None:
            return
        final_name = self.output_dir / self.filename.name

        if self.on_recording_stopping is not None:
            self.on_recording_stopping(final_name)
        self.writer.release()
        self.filename.rename(final_name)
        # .rename(self.filename, final_name)
        self.writer = None

    def delete_recording(self):
        self.recording = False
        if self.writer is None:
            return

        self.writer.release()
        self.filename.unlink(missing_ok=True)
        self.writer = None


def new_temp_name(frame_time):
    return datetime.fromtimestamp(frame_time).strftime(
        "%Y%m%d.%H%M%S.%f" + AVI_TEMP_EXT
    )
