from pathlib import Path
import os.path as path

import attr
import yaml

from load.loadconfig import LoadConfig
from track.trackingconfig import TrackingConfig
from train.config import TrainConfig
from classify.classifyconfig import ClassifyConfig
from build.buildconfig import BuildConfig
from evaluate.evaluateconfig import EvaluateConfig
from .defaultconfig import DefaultConfig

CONFIG_FILENAME = "classifier.yaml"
CONFIG_DIRS = [Path(__file__).parent.parent, Path("/etc/cacophony")]


@attr.s
class Config(DefaultConfig):

    DEFAULT_LABELS = ["bird", "false-positive", "hedgehog", "possum", "rat", "stoat"]
    EXCLUDED_TAGS = ["untagged", "unidentified"]

    source_folder = attr.ib()
    load = attr.ib()
    tracks_folder = attr.ib()
    labels = attr.ib()
    build = attr.ib()
    tracking = attr.ib()
    train = attr.ib()
    classify_tracking = attr.ib()
    classify = attr.ib()
    evaluate = attr.ib()
    excluded_tags = attr.ib()
    reprocess = attr.ib()
    previews_colour_map = attr.ib()
    use_gpu = attr.ib()
    worker_threads = attr.ib()
    debug = attr.ib()
    use_opt_flow = attr.ib()
    res_x = attr.ib()
    res_y = attr.ib()

    @classmethod
    def load_from_file(cls, filename=None):
        if not filename:
            filename = find_config()
        with open(filename) as stream:
            return cls.load_from_stream(stream)

    @classmethod
    def load_from_stream(cls, stream):
        raw = yaml.safe_load(stream)
        default = Config.get_defaults()
        if raw is None:
            raw = {}
        # Configuration from "tracking" section is used in
        # "classify_tracking" when not specified.
        deep_copy_map_if_key_not_exist(raw["tracking"], raw["classify_tracking"])
        deep_copy_map_if_key_not_exist(default.as_dict(), raw)

        base_folder = raw.get("base_data_folder")
        if base_folder is None:
            raise KeyError("base_data_folder not found in configuration file")
        base_folder = path.expanduser(base_folder)

        return cls(
            source_folder=path.join(base_folder, raw["source_folder"]),
            tracks_folder=path.join(base_folder, raw.get("tracks_folder", "tracks")),
            tracking=TrackingConfig.load(raw["tracking"]),
            load=LoadConfig.load(raw["load"]),
            train=TrainConfig.load(raw["train"], base_folder),
            classify_tracking=TrackingConfig.load(raw["classify_tracking"]),
            classify=ClassifyConfig.load(raw["classify"], base_folder),
            evaluate=EvaluateConfig.load(raw["evaluate"], base_folder),
            excluded_tags=raw["excluded_tags"],
            reprocess=raw["reprocess"],
            previews_colour_map=raw["previews_colour_map"],
            use_gpu=raw["use_gpu"],
            worker_threads=raw["worker_threads"],
            labels=raw["labels"],
            build=BuildConfig.load(raw["build"]),
            debug=raw["debug"],
            use_opt_flow=raw["use_opt_flow"],
            res_x=raw["res_x"],
            res_y=raw["res_y"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            source_folder="",
            tracks_folder="tracks",
            labels=Config.DEFAULT_LABELS,
            excluded_tags=Config.EXCLUDED_TAGS,
            reprocess=True,
            previews_colour_map="custom_colormap.dat",
            use_gpu=False,
            worker_threads=0,
            build=BuildConfig.get_defaults(),
            tracking=TrackingConfig.get_defaults(),
            load=LoadConfig.get_defaults(),
            train=TrainConfig.get_defaults(),
            classify_tracking=TrackingConfig.get_defaults(),
            classify=ClassifyConfig.get_defaults(),
            evaluate=EvaluateConfig.get_defaults(),
            debug=False,
            use_opt_flow=False,
            res_x=160,
            res_y=120,
        )

    def validate(self):
        self.build.validate()
        self.tracking.validate()
        self.load.validate()
        self.train.validate()
        self.classify.validate()
        self.evaluate.validate()
        return True

    def as_dict(self):
        return attr.asdict(self)


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


def parse_options_param(name, value, options):
    if value.lower() not in options:
        raise Exception(
            "Cannot parse {} as '{}'.  Valid options are {}.".format(
                name, value, options
            )
        )
    return value.lower()


def deep_copy_map_if_key_not_exist(from_map, to_map):
    for key in from_map:
        if isinstance(from_map[key], dict):
            if key not in to_map:
                to_map[key] = {}
            deep_copy_map_if_key_not_exist(from_map[key], to_map[key])
        elif key not in to_map:
            to_map[key] = from_map[key]
