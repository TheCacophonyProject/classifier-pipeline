from datetime import datetime
import logging
import os
import yaml
from load.cliptrackextractor import ClipTrackExtractor
from cptv import CPTVWriter
from cptv import Frame
from datetime import timedelta
import time
import psutil
from piclassifier.recorder import Recorder
from pathlib import Path
from ml_tools.logs import init_logging
import multiprocessing

CPTV_TEMP_EXT = ".temp"


class CPTVRecorder(Recorder):
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
        self.write_until = 0
        self.rec_time = 0
        self.on_recording_stopping = on_recording_stopping
        self.frame_q = multiprocessing.Queue()
        self.rec_p = None

    def force_stop(self):
        if not self.recording:
            return

        if self.frames > self.min_recording:
            self.stop_recording(time.time())
        else:
            logging.info("Recording stopped early deleting short recording")
            self.delete_recording()

    def has_minimum(self):
        return self.frames >= self.write_until

    def start_recording(
        self, background_frame, preview_frames, temp_thresh, frame_time
    ):
        start = time.time()
        if self.recording:
            logging.warn("Already recording, stop recording first")
            return False
        self.frames = 0
        self.filename = new_temp_name(frame_time)
        self.filename = self.output_dir / self.filename
        self.rec_p = multiprocessing.Process(
            target=record,
            args=(
                self.frame_q,
                self.filename,
                temp_thresh,
                self.preview_secs,
                self.motion,
                self.headers,
                self.location_config,
                self.device_config,
                background_frame,
                preview_frames,
            ),
        )
        self.rec_p.start()
        self.recording = True

        self.write_until = self.frames + self.min_frames

        logging.info("recording %s started temp_thresh: %d", self.filename, temp_thresh)

        self.rec_time += time.time() - start
        return True

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

    def write_frame(self, frame):
        start = time.time()
        self.frame_q.put(frame)
        # self.writer.next_frame(frame)
        self.frames += 1
        self.rec_time += time.time() - start

    def stop_recording(self, frame_time):
        start = time.time()
        self.rec_time += time.time() - start
        self.recording = False
        logging.info("Waiting for recorder to finish")
        self.frame_q.put(0)
        self.rec_p.join()
        self.frame_q = multiprocessing.Queue()
        self.rec_p = None

        logging.info(
            "recording ended %s frames %s time recording %s per frame ",
            self.frames,
            self.rec_time,
            self.rec_time / self.frames,
        )
        self.rec_time = 0
        self.write_until = 0
        final_name = self.filename.with_suffix(".cptv")
        # write metadata first
        if self.on_recording_stopping is not None:
            self.on_recording_stopping(final_name)
        self.filename.rename(final_name)

    def delete_recording(self):
        self.recording = False
        self.filename.unlink()


def new_temp_name(frame_time):
    return datetime.fromtimestamp(frame_time).strftime(
        "%Y%m%d.%H%M%S.%f" + CPTV_TEMP_EXT
    )


def record(
    queue,
    filename,
    temp_thresh,
    preview_secs,
    motion_config,
    headers,
    location_config,
    device_config,
    background_frame=None,
    init_frames=None,
):
    init_logging()
    frames = 0
    try:
        logging.info("Recorder %s started", filename.resolve())
        f = open(filename, "wb")
        writer = CPTVWriter(f)
        writer.timestamp = datetime.now()
        writer.latitude = location_config.latitude
        writer.longitude = location_config.longitude
        writer.preview_secs = preview_secs
        motion_config.temp_thresh = temp_thresh
        writer.motion_config = yaml.dump(motion_config.as_dict()).encode()[:255]
        if headers.model:
            writer.model = headers.model.encode()
        if headers.brand:
            writer.brand = headers.brand.encode()
        if headers.firmware:
            writer.firmware = headers.firmware.encode()
        writer.camera_serial = headers.serial
        if background_frame is not None:
            f = Frame(background_frame, timedelta(), timedelta(), 0, 0)
            f.background_frame = True
            writer.background_frame = f
        # add brand model fps etc to cptv when python-cptv supports
        if device_config.name:
            writer.device_name = device_config.name.encode()
        if device_config.device_id:
            writer.device_id = device_config.device_id

        writer.write_header()
        if init_frames is not None:
            for frame in init_frames:
                writer.write_frame(frame)
                frames += 1
        while True:
            frame = queue.get()
            if isinstance(frame, int) and frame == 0:
                writer.close()
                break
            writer.write_frame(frame)
            frames += 1

    except:
        logging.error("Error Recording %s", filename.resolve(), exc_info=True)
        try:
            writer.close()
        except:
            pass
    logging.info("Recorder %s Written %s", filename.resolve(), frames)
