from pathlib import Path
import os.path as path

import attr
import yaml

from load.loadconfig import LoadConfig
from track.trackingconfig import TrackingConfig
from train.config import TrainConfig
from classify.classifyconfig import ClassifyConfig
from evaluate.evaluateconfig import EvaluateConfig

CONFIG_FILENAME = "classifier.yaml"
CONFIG_DIRS = [Path(__file__).parent.parent, Path("/etc/cacophony")]


@attr.s
class Config:
    source_folder = attr.ib()
    loader = attr.ib()
    tracks_folder = attr.ib()
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

    @classmethod
    def load_from_file(cls, filename=None):
        if not filename:
            filename = find_config()
        with open(filename) as stream:
            return cls.load_from_stream(stream)

    @classmethod
    def load_from_stream(cls, stream):
        raw = yaml.safe_load(stream)

        # "classify_tracking" params are overrides, add other parameters from "tracking"
        deep_copy_map_if_key_not_exist(raw["tracking"], raw["classify_tracking"])

        base_folder = path.expanduser(raw["base_data_folder"])
        return cls(
            source_folder=path.join(base_folder, raw["source_folder"]),
            tracks_folder=path.join(base_folder, raw.get("tracks_folder", "tracks")),
            tracking=TrackingConfig.load(raw["tracking"]),
            loader=LoadConfig.load(raw["load"]),
            train=TrainConfig.load(raw["train"], base_folder),
            classify_tracking=TrackingConfig.load(raw["classify_tracking"]),
            classify=ClassifyConfig.load(raw["classify"], base_folder),
            evaluate=EvaluateConfig.load(raw["evaluate"], base_folder),
            excluded_tags=raw["excluded_tags"],
            reprocess=raw["reprocess"],
            previews_colour_map=raw["previews_colour_map"],
            use_gpu=raw["use_gpu"],
            worker_threads=raw["worker_threads"],
        )


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
