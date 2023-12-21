from datetime import datetime
import logging
import os
import yaml
from track.cliptrackextractor import ClipTrackExtractor
from cptv import CPTVWriter
from cptv import Frame
from datetime import timedelta
import time
from piclassifier.recorder import Recorder
from pathlib import Path
from ml_tools.logs import init_logging
import multiprocessing
from .eventreporter import log_event

CPTV_EXT = ".cptv"


class CPTVRecorder(Recorder):
    def __init__(
        self,
        thermal_config,
        headers,
        name="CPTVRecorder",
        **args,
    ):
        super().__init__(
            thermal_config,
            headers,
            name,
            CPTV_EXT,
            **args,
        )

    def new_recording(self, background_frame, preview_frames, temp_thresh, frame_time):
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
        return True

    def final_name(self):
        return self.output_dir / self.filename.with_suffix(self.file_extention).name


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
    name="CPTVRecorder",
):
    init_logging()
    frames = 0
    try:
        logging.info("%s Recorder %s started", name, filename.resolve())
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

    except Exception as ex:
        logging.error("%s Error Recording %s", name, filename.resolve(), exc_info=True)
        log_event("error-recording", ex)
        try:
            writer.close()
        except:
            pass
    logging.info("%s Recorder %s Written %s", name, filename.resolve(), frames)
