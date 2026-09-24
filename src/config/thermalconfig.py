from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any
import toml
import fcntl

from .locationconfig import LocationConfig
from .timewindow import RelAbsTime, TimeWindow

CONFIG_FILENAME = "config.toml"
CONFIG_DIRS = [Path(__file__).parent.parent, Path("/etc/cacophony")]


class LockSafeConfig:
    def __init__(self, filename):
        self.lock_file = filename + ".lock"
        self.filename = filename
        self.lock_f = None
        self.f = None

    def __enter__(self):
        # shared lock so we don't read while another process is writing the config
        self.lock_f = open(self.lock_file, "a")
        fcntl.flock(self.lock_f, fcntl.LOCK_SH)
        self.f = open(self.filename)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.f.close()
        finally:
            fcntl.flock(self.lock_f, fcntl.LOCK_UN)
            self.lock_f.close()


@dataclass(slots=True)
class ThrottlerConfig:
    bucket_size: Any
    activate: Any
    no_motion: Any
    max_throttling_minutes: Any

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
        return asdict(self)


@dataclass(slots=True)
class CameraMotionConfig:
    temp_thresh: Any
    delta_thresh: Any
    count_thresh: Any
    frame_compare_gap: Any
    one_diff_only: Any
    trigger_frames: Any
    edge_pixels: Any
    warmer_only: Any
    dynamic_thresh: Any

    # TODO these need to be moved into a different configf that isn't dependent on model info
    run_classifier: Any = False
    bluetooth_beacons: Any = False
    tracking_events: Any = False
    do_tracking: Any = False
    postprocess: Any = False
    postprocess_events: Any = False

    @classmethod
    def defaults_for(cls, model):
        if model == "lepton3":
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
            )
        else:
            return cls(
                temp_thresh=28000,
                delta_thresh=150,
                count_thresh=3,
                frame_compare_gap=45,
                one_diff_only=True,
                trigger_frames=2,
                edge_pixels=1,
                warmer_only=True,
                dynamic_thresh=True,
            )
            

    @classmethod
    def load(cls, motion):
        motion = cls(
            temp_thresh=motion.get("temp-thresh"),
            delta_thresh=motion.get("delta-thresh"),
            count_thresh=motion.get("count-thresh"),
            frame_compare_gap=motion.get("frame-compare-gap"),
            one_diff_only=motion.get("use-one-diff-only"),
            trigger_frames=motion.get("trigger-frames"),
            edge_pixels=motion.get("edge-pixels"),
            warmer_only=motion.get("warmer-only"),
            dynamic_thresh=motion.get("dynamic-thresh"),
            run_classifier=motion.get("run-classifier", False),
            bluetooth_beacons=motion.get("bluetooth-beacons", False),
            tracking_events=motion.get("tracking-events", False),
            do_tracking=motion.get("do-tracking", False),
            postprocess=motion.get("postprocess", False),
            postprocess_events=motion.get("postprocess-events", False),
        )
        return motion
    
    def as_dict(self):
        return asdict(self)

    def use_defaults_for(self, model):
        default = CameraMotionConfig.defaults_for(model)

        def value_for(field):
            current = getattr(self, field)
            return current if current is not None else getattr(default, field)

        return CameraMotionConfig(
            temp_thresh=value_for("temp_thresh"),
            delta_thresh=value_for("delta_thresh"),
            count_thresh=value_for("count_thresh"),
            frame_compare_gap=value_for("frame_compare_gap"),
            one_diff_only=value_for("one_diff_only"),
            trigger_frames=value_for("trigger_frames"),
            edge_pixels=value_for("edge_pixels"),
            warmer_only=value_for("warmer_only"),
            dynamic_thresh=value_for("dynamic_thresh"),
            run_classifier=self.run_classifier,
            bluetooth_beacons=self.bluetooth_beacons,
            tracking_events=self.tracking_events,
            do_tracking=self.do_tracking,
            postprocess=self.postprocess,
            postprocess_events=self.postprocess_events,
        )


@dataclass(slots=True)
class RecorderConfig:
    preview_secs: Any
    min_secs: Any
    max_secs: Any
    rec_window: Any
    output_dir: Any
    disable_recordings: Any
    constant_recorder: Any
    use_low_power_mode: Any
    min_disk_space_mb: Any
    instant_classify: Any

    @classmethod
    def load(cls, recorder, window, location_config):
        return cls(
            constant_recorder=recorder.get("constant-recorder", False),
            disable_recordings=recorder.get("disable-recordings", False),
            min_secs=recorder.get("min-secs", 5),
            max_secs=recorder.get("max-secs", 600),
            preview_secs=recorder.get("preview-secs", 5),
            rec_window=TimeWindow(
                RelAbsTime(window.get("start-recording"), default_offset=-30 * 60),
                RelAbsTime(window.get("stop-recording"), default_offset=30 * 60),
                *location_config.get_lat_long(use_default=True),
                location_config.altitude,
            ),
            min_disk_space_mb=recorder.get("min-disk-space-mb", 200),
            output_dir=recorder.get("output-dir", "/var/spool/cptv"),
            use_low_power_mode=recorder.get("use-low-power-mode", False),
            instant_classify = recorder.get("instant-classify",False),
        )


@dataclass(slots=True)
class DeviceSetup:
    ir: Any = False
    trap_size: Any = None
    # S or L for small or large

    @classmethod
    def load(cls, device):
        size = device.get("trap-size", "L")
        if size is not None:
            size = size.upper()
        return cls(ir=device.get("ir", False), trap_size=size)


@dataclass(slots=True)
class DeviceConfig:
    device_id: Any
    name: Any

    @classmethod
    def load(cls, device):
        return cls(
            name=device.get("name"),
            device_id=device.get("id"),
        )


@dataclass(slots=True)
class ThermalConfig:
    base_motion: Any
    recorder: Any
    device: Any
    location: Any
    throttler: Any
    device_setup: Any
    config_file: Any

    @classmethod
    def load_from_file(cls, filename=None):
        if not filename:
            filename = ThermalConfig.find_config()
        with LockSafeConfig(filename) as stream:
            return cls.load_from_stream(filename, stream)

    @classmethod
    def load_from_stream(cls, filename, stream):
        raw = toml.load(stream)
        if raw is None:
            raw = {}

        location_config = LocationConfig.load(raw.get("location", {}))
        return cls(
            config_file=filename,
            throttler=ThrottlerConfig.load(raw.get("thermal-throttler", {})),
            base_motion=CameraMotionConfig.load(raw.get("thermal-motion", {})),
            recorder=RecorderConfig.load(
                raw.get("thermal-recorder", {}), raw.get("windows", {}), location_config
            ),
            device=DeviceConfig.load(raw.get("device", {})),
            device_setup=DeviceSetup.load(raw.get("device-setup", {})),
            location=location_config,
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
