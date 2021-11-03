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
import tracemalloc

CPTV_TEMP_EXT = ".cptv.temp"


class CPTVRecorder(Recorder):
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
        self.on_recording_stopping = on_recording_stopping

    def force_stop(self):
        if not self.recording:
            return

        if self.has_minimum():
            self.stop_recording()
        else:
            self.delete_recording()

    def process_frame(self, movement_detected, cptv_frame):
        if self.recording:
            self.write_frame(cptv_frame)
            if movement_detected:
                self.write_until = self.frames + self.min_frames
            elif self.has_minimum():
                logging.debug(
                    "Stopping recording with frame recevied at %s",
                    cptv_frame.received_at,
                )
                self.stop_recording()
                return

            if self.frames == self.max_frames:
                self.stop_recording()

    def has_minimum(self):
        return self.frames > self.write_until

    def start_recording(self, background_frame, preview_frames, temp_thresh):
        tracemalloc.start()

        self.snapshot = tracemalloc.take_snapshot()

        start = time.time()
        if self.recording:
            logging.warn("Already recording, stop recording first")
            return False
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
        self.writer.motion_config = yaml.dump(self.motion.as_dict()).encode()[:255]
        self.motion.temp_thresh = default_thresh
        if self.headers.model:
            self.writer.model = self.headers.model.encode()
        if self.headers.brand:
            self.writer.brand = self.headers.brand.encode()
        if self.headers.firmware:
            self.writer.firmware = self.headers.firmware.encode()
        self.writer.camera_serial = self.headers.serial
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

        logging.info(
            "%s memory",
            psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
        )

        self.rec_time += time.time() - start
        return True

    def write_frame(self, cptv_frame):
        start = time.time()
        self.writer.write_frame(cptv_frame)
        self.frames += 1
        self.rec_time += time.time() - start
        self.writer.fileobj.flush()
        logging.info(
            "%s memory on write",
            psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
        )

    def stop_recording(self):
        start = time.time()
        self.rec_time += time.time() - start
        self.recording = False
        logging.info(
            "recording ended %s frames %s time recording %s per frame ",
            self.frames,
            self.rec_time,
            self.rec_time / self.frames,
        )
        logging.info(
            "%s memory on stop",
            psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
        )
        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(self.snapshot, "lineno")

        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            logging.info("Stats %s", stat)
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


def get_size(obj, name="base", seen=None, depth=0):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if attr.has(obj):
        obj = attr.asdict(obj)
        size += sum(
            [get_size(v, f"{name}.{k}", seen, depth + 1) for k, v in obj.items()]
        )
        size += sum([get_size(k, f"{name}.{k}", seen, depth + 1) for k in obj.keys()])
    if isinstance(obj, dict):
        size += sum(
            [get_size(v, f"{name}.{k}", seen, depth + 1) for k, v in obj.items()]
        )
        size += sum([get_size(k, f"{name}.{k}", seen, depth + 1) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):

        size += get_size(obj.__dict__, f"{name}.dict", seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        # print("iter??")

        size += sum([get_size(i, f"{name}.iter", seen, depth + 1) for i in obj])
    # if size * 1e-6 > 0.1 and depth <= 1:
    #     print(name, " size ", round(size * 1e-6, 2), "MB")
    return size
