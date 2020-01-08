from pathlib import Path
import attr
import toml
import datetime
import portalocker
import os

from .locationconfig import LocationConfig

CONFIG_FILENAME = "config.toml"
CONFIG_DIRS = [Path(__file__).parent.parent, Path("/etc/cacophony")]


class LockSafeConfig:
    def __init__(self, filename):
        self.lock_file = filename + ".lock"
        self.filename = filename
        self.f = None
        self.lock = portalocker.Lock(
            self.lock_file, "r", flags=portalocker.LOCK_SH, timeout=1
        )
        if not os.path.exists(self.lock_file):
            f = open(self.lock_file, "w+")
            f.close()

    def __enter__(self):
        # note: we might not have to lock when in read only mode?
        # this could improve performance
        self.lock.acquire()
        self.f = open(self.filename)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.f.close()
        finally:
            self.lock.release()


@attr.s
class MotionConfig:
    temp_thresh = attr.ib()
    delta_thresh = attr.ib()
    count_thresh = attr.ib()
    frame_compare_gap = attr.ib()
    one_diff_only = attr.ib()
    trigger_frames = attr.ib()
    verbose = attr.ib()
    edge_pixels = attr.ib()
    warmer_only = attr.ib()
    dynamic_thresh = attr.ib()
    run_classifier = attr.ib()

    @classmethod
    def load(cls, motion):
        return cls(
            temp_thresh=motion.get("temp-thresh", 2750),
            delta_thresh=motion.get("delta-thresh", 20),
            count_thresh=motion.get("count-thresh", 1),
            frame_compare_gap=motion.get("frame-compare-gap", 45),
            one_diff_only=motion.get("use-one-diff-only", False),
            trigger_frames=motion.get("trigger-frames", 1),
            verbose=motion.get("verbose", True),
            edge_pixels=motion.get("edge-pixels", 3),
            warmer_only=motion.get("warmer-only", False),
            dynamic_thresh=motion.get("dynamic-thresh", True),
            run_classifier=motion.get("run-classifier", False),
        )


class RelAbsTime:
    def __init__(self, time_str, default_offset=None, default_time=None):
        self.is_relative = False
        self.offset_s = 0
        self.time = None

        if time_str == "":
            self.any_time = True
            return

        try:
            self.any_time = False
            self.time = datetime.datetime.strptime(time_str, "%H:%M").time()
        except:
            if default_time:
                self.time = default_time
            else:
                self.is_relative = True
                self.offset_s = self.parse_duration(time_str, default_offset)

    def is_after(self):
        return self.any_time or datetime.datetime.now().time() > self.time

    def is_before(self):
        return self.any_time or datetime.datetime.now().time() < self.time

    def parse_duration(self, time_str, default_offset=None):

        if not time_str:
            return default_offset

        time_type = time_str[-1]
        if time_type.isalpha():
            try:
                offset = int(time_str[:-1])
            except ValueError:
                return default_offset
            if time_type == "s":
                return offset
            elif time_type == "m":
                return offset * 60
            elif time_type == "h":
                return offset * 60 * 60
            return offset
        else:
            try:
                offset = int(time_str[:-1])
                return offset * 60
            except ValueError:
                pass
            return default_offset


@attr.s
class RecorderConfig:
    preview_secs = attr.ib()
    min_secs = attr.ib()
    max_secs = attr.ib()
    frame_rate = attr.ib()
    start_rec = attr.ib()
    end_rec = attr.ib()
    output_dir = attr.ib()

    @classmethod
    def load(cls, recorder, window):
        return cls(
            min_secs=recorder.get("min-secs", 2),
            max_secs=recorder.get("max-secs", 10),
            preview_secs=recorder.get("preview-secs", 5),
            frame_rate=recorder.get("frame-rate", 9),
            start_rec=RelAbsTime(window.get("start-recording"), default_offset=30 * 60),
            end_rec=RelAbsTime(window.get("stop-recording"), default_offset=30 * 60),
            output_dir=recorder["output-dir"],
        )


@attr.s
class DeviceConfig:
    device_id = attr.ib()
    name = attr.ib()

    @classmethod
    def load(cls, device):
        return cls(name=device.get("name"), device_id=device.get("id"))


@attr.s
class ThermalConfig:
    motion = attr.ib()
    recorder = attr.ib()
    device = attr.ib()
    location = attr.ib()

    @classmethod
    def load_from_file(cls, filename=None):
        if not filename:
            filename = ThermalConfig.find_config()
        with LockSafeConfig(filename) as stream:
            return cls.load_from_stream(stream)

    @classmethod
    def load_from_stream(cls, stream):
        raw = toml.load(stream)
        if raw is None:
            raw = {}
        return cls(
            motion=MotionConfig.load(raw.get("thermal-motion", {})),
            recorder=RecorderConfig.load(
                raw.get("thermal-recorder", {}), raw.get("windows", {})
            ),
            device=DeviceConfig.load(raw.get("device", {})),
            location=LocationConfig.load(raw.get("location", {})),
        )

    def validate(self):
        return True

    @staticmethod
    def find_config():
        for directory in CONFIG_DIRS:
            p = directory / CONFIG_FILENAME
            if p.is_file():
                return str(p)
        raise FileNotFoundError(
            "No configuration file found.  Looking for file named '{}' in dirs {}".format(
                CONFIG_FILENAME, CONFIG_DIRS
            )
        )
