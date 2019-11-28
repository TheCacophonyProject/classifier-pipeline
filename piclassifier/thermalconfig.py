from pathlib import Path
import attr
import yaml

CONFIG_FILENAME = "thermal-recorder.yaml"
CONFIG_DIRS = [Path(__file__).parent.parent, Path("/etc")]


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
            one_diff_only=motion.get("one-diff-only", False),
            trigger_frames=motion.get("trigger-frames", 1),
            verbose=motion.get("verbose", True),
            edge_pixels=motion.get("edge-pixels", 3),
            warmer_only=motion.get("warmer-only", False),
            dynamic_thresh=motion.get("dynamic-thresh", True),
            run_classifier=motion.get("run_classifier", False),
        )


@attr.s
class RecorderConfig:
    preview_secs = attr.ib()
    min_secs = attr.ib()
    max_secs = attr.ib()
    frame_rate = attr.ib()
    use_sunrise_sunset = attr.ib()
    sunrise_offset = attr.ib()
    sunset_offset = attr.ib()

    @classmethod
    def load(cls, recorder):
        return cls(
            min_secs=recorder.get("min-secs", 2),
            max_secs=recorder.get("max-secs", 10),
            preview_secs=recorder.get("preview-secs", 5),
            frame_rate=recorder.get("frame-rate", 9),
            use_sunrise_sunset=recorder.get("sunrise-sunset", False),
            sunrise_offset=recorder.get("sunrise-offset", 0),
            sunset_offset=recorder.get("sunset-offset", 0),
        )


@attr.s
class ThermalConfig:
    motion = attr.ib()
    recorder = attr.ib()
    output_dir = attr.ib()

    @classmethod
    def load_from_file(cls, filename=None):
        if not filename:
            filename = ThermalConfig.find_config()
        with open(filename) as stream:
            return cls.load_from_stream(stream)

    @classmethod
    def load_from_stream(cls, stream):
        raw = yaml.safe_load(stream)
        if raw is None:
            raw = {}
        return cls(
            output_dir=raw["output-dir"],
            motion=MotionConfig.load(raw.get("motion", {})),
            recorder=RecorderConfig.load(raw.get("recorder", {})),
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
