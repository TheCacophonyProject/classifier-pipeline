import numpy as np
from datetime import datetime
import logging
import os
import yaml
import multiprocessing
from datetime import timedelta
import time
import psutil
import cv2
from pathlib import Path

from ml_tools.mpeg_creator import MPEGCreator
from piclassifier.recorder import Recorder
from load.cliptrackextractor import ClipTrackExtractor
from ml_tools.logs import init_logging

TEMP_DIR = "temp"
VIDEO_EXT = ".mp4"

# FOURCC = cv2.VideoWriter_fourcc(*"avc1")
# JUST FOR TEST
# VIDEO_EXT = ".avi"
# FOURCC = cv2.VideoWriter_fourcc("M", "J", "P", "G")

# gp this should work on pi, but causing lots of headaches
# wont set bitrate, wont play back on ubuntu, should check again in future
# as is very fast
# CODEC = "h264_v4l2m2m"
CODEC = "libx264"
BITRATE = "1M"
PIX_FMT = "yuv420p"


class IRRecorder(Recorder):
    def __init__(
        self,
        thermal_config,
        headers,
        on_recording_stopping=None,
        name="IRRecorder",
        constant_recorder=False,
    ):
        super().__init__(name)
        self.res_x = headers.res_x
        self.res_y = headers.res_y
        self.fps = headers.fps

        self.temp_dir = self.output_dir / TEMP_DIR

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.frame_q = multiprocessing.Queue()

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

    def start_recording(
        self, background_frame, preview_frames, temp_thresh, frame_time
    ):
        start = time.time()
        if self.recording:
            logging.warn("%s Already recording, stop recording first", self.name)
            return False
        self.frames = 0
        self.filename = new_temp_name(frame_time)

        self.filename = self.temp_dir / self.filename

        # self.writer = MPEGCreator(self.filename, fps=self.fps, codec=CODEC)
        back = None
        if background_frame is not None:
            back = background_frame[:, :, np.newaxis]
            back = np.repeat(back, 3, axis=2)
        # preview_frames.insert(0, back)

        self.rec_p = multiprocessing.Process(
            target=record,
            args=(
                self.frame_q,
                self.filename,
                self.fps,
                back,
                preview_frames,
                self.name,
            ),
        )
        self.rec_p.start()
        # self.writer.next_frame(back)

        self.recording = True
        # for frame in preview_frames:
        #     self.write_frame(frame)
        self.write_until = self.frames + self.min_frames
        logging.info("%s recording %s started", self.name, self.filename.resolve())
        self.rec_time += time.time() - start
        return True

    def write_frame(self, frame):
        start = time.time()
        self.frame_q.put(frame)
        # self.writer.next_frame(frame)
        self.frames += 1
        self.rec_time += time.time() - start

    def final_name(self):
        self.output_dir / self.filename.name


def new_temp_name(frame_time):
    return datetime.fromtimestamp(frame_time).strftime("%Y%m%d-%H%M%S.%f" + VIDEO_EXT)


def write_frame(writer, frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    writer.next_frame(yuv, frame.shape)


def record(queue, filename, fps, background=None, init_frames=None, name="IRRecorder"):
    init_logging()
    frames = 0
    try:
        logging.info("%s Recorder %s started", name, filename.resolve())

        writer = MPEGCreator(
            filename, fps=fps, codec=CODEC, bitrate=BITRATE, pix_fmt=PIX_FMT
        )
        if background is not None:
            write_frame(writer, background)
            frames += 1
        if init_frames is not None:
            for frame in init_frames:
                write_frame(writer, frame)
                frames += 1
        while True:
            frame = queue.get()
            if isinstance(frame, int) and frame == 0:
                writer.close()
                break
            write_frame(writer, frame)
            frames += 1

    except:
        logging.error("%s Error Recording %s", name, filename.resolve(), exc_info=True)
        try:
            writer.close()
        except:
            pass
    logging.info("%s Recorder %s Written %s", name, filename.resolve(), frames)
