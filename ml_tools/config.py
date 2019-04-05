from pathlib import Path
import os.path as path

import attr
import yaml

from track.trackingconfig import TrackingConfig
from track.extractconfig import ExtractConfig
from classify.classifyconfig import ClassifyConfig

CONFIG_FILENAME = "classifier.yaml"
CONFIG_DIRS = [Path(__file__).parent.parent, Path("/etc/cacophony")]


@attr.s
class Config:
    tracking = attr.ib()
    extract = attr.ib()
    classify_tracking = attr.ib()
    classify = attr.ib()
    source_folder = attr.ib()
    excluded_folders = attr.ib()
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
            tracking=TrackingConfig.load(raw["tracking"]),
            extract=ExtractConfig.load(raw["extract"], base_folder),
            classify_tracking=TrackingConfig.load(raw["classify_tracking"]),
            classify=ClassifyConfig.load(raw["classify"], base_folder),
            source_folder=path.join(base_folder, raw["source_folder"]),
            excluded_folders=raw["excluded_folders"],
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


def args_to_config(args):
    """Convert an argparse args dict to a usable Config object."""
    if args.config_file:
        conf = Config.read_config_file(args.config_file)
    else:
        conf = Config.read_default_config_file()
    return Config.load_from_map(conf)
