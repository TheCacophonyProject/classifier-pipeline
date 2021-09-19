from datetime import datetime
import logging
import os
import yaml
from load.cliptrackextractor import ClipTrackExtractor
from cptv import CPTVWriter
from cptv import Frame
from datetime import timedelta
import time

CPTV_TEMP_EXT = ".cptv.temp"

CACHE_COUNT = 20


class CPTVRecorder:
    def __init__(self, thermal_config, headers, on_recording_stopping=None):
        self.location_config = thermal_config.location
        self.device_config = thermal_config.device
        self.output_dir = thermal_config.recorder.output_dir
        self.motion = thermal_config.motion
        self.preview_secs = thermal_config.recorder.preview_secs
        self.writer = None
        self.filename = None
        self.recording = False
        self.frames = 0
        self.headers = headers
        self.min_frames = thermal_config.recorder.min_secs * headers.fps
        self.max_frames = thermal_config.recorder.max_secs * headers.fps
        self.write_until = 0
        self.rec_time = 0
        self.cache_frames = []
        self.on_recording_stopping = on_recording_stopping

    def force_stop(self):
        if not self.recording:
            return
        self.stop_recording()
        return
        if self.has_minimum():
            self.stop_recording()
        else:
            self.delete_recording()

    def process_frame(self, movement_detected, cptv_frame):
        if movement_detected:
            self.write_until = self.frames + self.min_frames
            self.write_frame(cptv_frame)
        elif self.recording:
            if self.has_minimum():
                self.stop_recording()
            else:
                self.write_frame(cptv_frame)

            if self.frames == self.max_frames:
                self.stop_recording()

    def has_minimum(self):
        return self.frames > self.write_until

    def start_recording(self, background_frame, preview_frames, temp_thresh):
        start = time.time()
        if self.recording:
            logging.warn("Already recording, stop recording first")
            return
        self.frames = 0
        self.filename = new_temp_name()
        self.filename = os.path.join(self.output_dir, self.filename)
        f = open(self.filename, "wb")
        self.writer = CPTVWriter(f)
        self.writer.timestamp = datetime.now()
        self.writer.latitude = self.location_config.latitude
        self.writer.longitude = self.location_config.longitude
        self.writer.preview_secs = self.preview_secs
        default_thresh = self.motion.temp_thresh
        self.motion.temp_thresh = temp_thresh
        print(self.motion.as_dict())
        self.writer.motion_config = yaml.dump(self.motion.as_dict()).encode()[:255]
        self.motion.temp_thresh = default_thresh

        f = Frame(background_frame, timedelta(), timedelta(), 0, 0)

        f.background_frame = True
        self.writer.background_frame = f
        # add brand model fps etc to cptv when python-cptv supports

        if self.device_config.name:
            self.writer.device_name = self.device_config.name.encode()
        if self.device_config.device_id:
            self.writer.device_id = self.device_config.device_id

        self.writer.write_header()

        self.recording = True
        for frame in preview_frames:
            self.write_frame(frame)
        self.write_until = self.frames + self.min_frames

        logging.info("recording %s started temp_thresh: %d", self.filename, temp_thresh)
        self.rec_time += time.time() - start

    def write_frame(self, cptv_frame):
        start = time.time()
        self.cache_frames.append(cptv_frame)
        if len(self.cache_frames) == CACHE_COUNT:
            for frame in self.cache_frames:
                self.writer.write_frame(frame)
            self.cache_frames = []
        self.frames += 1
        self.rec_time += time.time() - start

    def stop_recording(self):
        start = time.time()
        for frame in self.cache_frames:
            self.writer.write_frame(frame)
        self.cache_frames = []
        self.rec_time += time.time() - start
        self.recording = False
        logging.info(
            "recording ended %s %s per frame",
            self.rec_time,
            self.rec_time / self.frames,
        )
        self.rec_time = 0

        self.write_until = 0
        if self.writer is None:
            return
        final_name = os.path.splitext(self.filename)[0]
        if self.on_recording_stopping is not None:
            self.on_recording_stopping(final_name)
        self.writer.close()
        os.rename(self.filename, final_name)
        self.writer = None

    def delete_recording(self):
        self.recording = False
        if self.writer is None:
            return

        self.writer.close()
        os.remove(self.filename)
        self.writer = None


def new_temp_name():
    return datetime.now().strftime("%Y%m%d.%H%M%S.%f" + CPTV_TEMP_EXT)
