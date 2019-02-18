from collections import namedtuple
from pathlib import Path
import os.path as path
import yaml

from track.trackingconfig import TrackingConfig
from track.extractconfig import ExtractConfig
from classify.classifyconfig import ClassifyConfig

CONFIG_FILENAME = "classifier.yaml"
CONFIG_DIRS = [Path(__file__).parent.parent, Path("/etc/cacophony")]


ConfigBaseTuple = namedtuple(
    "Config",
    [
        "tracking",
        "extract",
        "classify_tracking",
        "classify",
        "source_folder",
        "excluded_folders",
        "reprocess",
        "previews_colour_map",
        "use_gpu",
        "worker_threads",
    ],
)

class Config(ConfigBaseTuple):
    @classmethod
    def load(cls):
        filename = find_config()
        with open(filename) as stream:
            yaml_map = yaml.load(stream)
            cls.load_from_map(yaml_map)

    @classmethod
    def load_from_map(cls, config):
        base_folder = config["base_data_folder"]
        # "classify_tracking" params are overrides, add other parameters from "tracking"
        deep_copy_map_if_key_not_exist(config["tracking"], config["classify_tracking"])
        return cls(
            tracking=TrackingConfig.load(config["tracking"]),
            extract=ExtractConfig.load(config["extract"], base_folder),
            classify_tracking=TrackingConfig.load(config["classify_tracking"]),
            classify=ClassifyConfig.load(config["classify"], base_folder),
            source_folder = path.join(base_folder, config["source_folder"]),
            excluded_folders = config["excluded_folders"],
            reprocess = config["reprocess"],
            previews_colour_map = config["previews_colour_map"],
            use_gpu=config["use_gpu"],
            worker_threads=config["worker_threads"],
        )

    @classmethod
    def read_default_config_file(cls):
        filename = find_config()
        return load_to_yaml(filename)

    @classmethod
    def read_config_file(cls, filename):
        if path.isfile(filename):
            return load_to_yaml(filename)
        else:
            raise FileNotFoundError("Configuration file '{}' was not found".format(filename))


def load_to_yaml(filename):
    with open(filename) as stream:
        yaml_map = yaml.load(stream)
    return yaml_map


def find_config():
    for directory in CONFIG_DIRS:
        p = directory / CONFIG_FILENAME
        if p.is_file():
            return str(p)
    raise FileNotFoundError("No configuration file found.  Looking for file named '{}' in dirs {}".format(CONFIG_FILENAME, CONFIG_DIRS))


def parse_options_param(name, value, options):
    if value.lower() not in options:
        raise Exception("Cannot parse {} as '{}'.  Valid options are {}.".format(name, value, options))
    return value.lower()

def deep_copy_map_if_key_not_exist(from_map, to_map):
    for key in from_map:
        if isinstance(from_map[key], dict):
            if key not in to_map:
                to_map[key] = {}
            deep_copy_map_if_key_not_exist(from_map[key], to_map[key])
        elif key not in to_map:
            to_map[key] = from_map[key]
