import numpy as np
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
from ml_tools.mpeg_creator import MPEGCreator

TEMP_DIR = "temp"

VIDEO_EXT = ".mp4"
FOURCC = cv2.VideoWriter_fourcc(*"avc1")
# JUST FOR TEST
# VIDEO_EXT = ".avi"
# FOURCC = cv2.VideoWriter_fourcc("M", "J", "P", "G")


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
        self.min_recording = self.preview_secs * headers.fps + self.min_frames
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
        if self.frames > self.min_recording:
            self.stop_recording(time.time())
        else:
            logging.info("Recording stopped early deleting short recording")
            # self.stop_recording(time.time())
            self.delete_recording()

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
        self.writer = MPEGCreator(self.filename, fps=self.fps, codec="h264_v4l2m2m")
        # self.writer = cv2.VideoWriter(
        #     str(self.filename), FOURCC, self.fps, (self.res_x, self.res_y)
        # )
        print(background_frame.shape)
        back = background_frame[:, :, np.newaxis]
        back = np.repeat(back, 3, axis=2)
        print(back.shape)
        self.writer.next_frame(back)
        default_thresh = self.motion.temp_thresh

        self.recording = True
        for frame in preview_frames:
            self.write_frame(frame)
        self.write_until = self.frames + self.min_frames
        logging.info("recording %s started", self.filename.resolve())
        self.rec_time += time.time() - start
        return True

    def write_frame(self, frame):
        start = time.time()

        self.writer.next_frame(frame)
        self.frames += 1
        self.rec_time += time.time() - start

    def stop_recording(self, frame_time):
        start = time.time()
        self.rec_time += time.time() - start
        self.recording = False
        final_name = self.output_dir / self.filename.name

        logging.info(
            "recording %s ended %s frames %s time recording %s per frame ",
            final_name,
            self.frames,
            self.rec_time,
            self.rec_time / self.frames,
        )
        self.rec_time = 0
        self.write_until = 0
        if self.writer is None:
            return

        if self.on_recording_stopping is not None:
            self.on_recording_stopping(final_name)
        # self.writer.release()
        self.writer.close()
        self.filename.rename(final_name)
        self.writer = None

    def delete_recording(self):
        self.recording = False
        if self.writer is None:
            return
        self.writer.close()

        # self.writer.release()
        self.filename.unlink()
        self.writer = None


def new_temp_name(frame_time):
    return datetime.fromtimestamp(frame_time).strftime("%Y%m%d-%H%M%S-%f" + VIDEO_EXT)
