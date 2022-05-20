from pathlib import Path
import attr
import toml
import portalocker
import os

from .locationconfig import LocationConfig
from .timewindow import RelAbsTime, TimeWindow

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
class ThrottlerConfig:
    bucket_size = attr.ib()
    activate = attr.ib()
    no_motion = attr.ib()
    max_throttling_minutes = attr.ib()

    @classmethod
    def load(cls, throttler):
        return cls(
            bucket_size=RelAbsTime(
                throttler.get("bucket-size"), default_offset=10 * 60
            ).offset_s,
            activate=throttler.get("activate", True),
            no_motion=throttler.get("no-motion", 5 * 60),
            max_throttling_minutes=throttler.get("max-throttling-minutes", 60),
        )

    def as_dict(self):
        return attr.asdict(self)


@attr.s
class CameraMotionConfig:
    temp_thresh = attr.ib()
    delta_thresh = attr.ib()
    count_thresh = attr.ib()
    frame_compare_gap = attr.ib()
    one_diff_only = attr.ib()
    trigger_frames = attr.ib()
    edge_pixels = attr.ib()
    warmer_only = attr.ib()
    dynamic_thresh = attr.ib()
    run_classifier = attr.ib()

    @classmethod
    def defaults_for(cls, model):
        if model == "lepton3.5":
            return cls(
                temp_thresh=28000,
                delta_thresh=200,
                count_thresh=3,
                frame_compare_gap=45,
                one_diff_only=True,
                trigger_frames=2,
                edge_pixels=1,
                warmer_only=True,
                dynamic_thresh=True,
                run_classifier=True,
            )
        else:
            return cls(
                temp_thresh=2750,
                delta_thresh=50,
                count_thresh=3,
                frame_compare_gap=45,
                one_diff_only=True,
                trigger_frames=2,
                edge_pixels=1,
                warmer_only=True,
                dynamic_thresh=True,
                run_classifier=True,
            )

    @classmethod
    def load(cls, motion, model=None):
        default = CameraMotionConfig.defaults_for(model)
        motion = cls(
            temp_thresh=motion.get("temp-thresh", default.temp_thresh),
            delta_thresh=motion.get("delta-thresh", default.delta_thresh),
            count_thresh=motion.get("count-thresh", default.count_thresh),
            frame_compare_gap=motion.get(
                "frame-compare-gap", default.frame_compare_gap
            ),
            one_diff_only=motion.get("use-one-diff-only", default.one_diff_only),
            trigger_frames=motion.get("trigger-frames", default.trigger_frames),
            edge_pixels=motion.get("edge-pixels", default.edge_pixels),
            warmer_only=motion.get("warmer-only", default.warmer_only),
            dynamic_thresh=motion.get("dynamic-thresh", default.dynamic_thresh),
            run_classifier=motion.get("run-classifier", default.run_classifier),
        )
        return motion

    def as_dict(self):
        return attr.asdict(self)


@attr.s
class RecorderConfig:
    preview_secs = attr.ib()
    min_secs = attr.ib()
    max_secs = attr.ib()
    rec_window = attr.ib()
    output_dir = attr.ib()
    send_lora_classification = attr.ib()
    send_lora_recording = attr.ib()

    @classmethod
    def load(cls, recorder, window):
        return cls(
            min_secs=recorder.get("min-secs", 10),
            max_secs=recorder.get("max-secs", 600),
            preview_secs=recorder.get("preview-secs", 5),
            send_lora_classification=recorder.get("send-lora-classification", False),
            send_lora_recording=recorder.get("send-lora-recording", False),
            rec_window=TimeWindow(
                RelAbsTime(window.get("start-recording"), default_offset=30 * 60),
                RelAbsTime(window.get("stop-recording"), default_offset=30 * 60),
            ),
            output_dir=recorder.get("output-dir", "."),
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
    throttler = attr.ib()

    @classmethod
    def load_from_file(cls, filename=None, model=None):
        if not filename:
            filename = ThermalConfig.find_config()
        with LockSafeConfig(filename) as stream:
            return cls.load_from_stream(stream, model)

    @classmethod
    def load_from_stream(cls, stream, model=None):
        raw = toml.load(stream)
        if raw is None:
            raw = {}
        return cls(
            throttler=ThrottlerConfig.load(raw.get("thermal-throttler", {})),
            motion=CameraMotionConfig.load(raw.get("thermal-motion", {}), model),
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
