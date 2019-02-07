from collections import namedtuple
from pathlib import Path
from track.trackingconfig import TrackingConfig

import os.path as path
import yaml


CONFIG_FILENAME = "classifier.yaml"
CONFIG_DIRS = [Path(__file__).parent.parent, Path("/etc/cacophony")]


configTuple = namedtuple(
    "Config",
    [
        "tracking",
        "model",
        "classify_tracking",
        "source_folder",
        "tracks_folder",
        "excluded_folders",
        "overwrite_mode",
        "previews_colour_map",
        "use_gpu",
    ],
)

class Config(configTuple):
    @classmethod
    def load(self):
        filename = find_config()
        with open(filename) as stream:
            yamlMap = yaml.load(stream)
            self.load_from_map(yamlMap)

    @classmethod
    def load_from_map(self, config):
            base_folder = config["base_data_folder"]
            return self(
                tracking=TrackingConfig.load(config["tracking"]),
                classify_tracking=TrackingConfig.load(config["classify_tracking"]),
                model=config["model"],
                source_folder = path.join(base_folder, config["source_folder"]),
                tracks_folder = path.join(base_folder, config["tracks_folder"]),
                excluded_folders = config["excluded_folders"],
                overwrite_mode = config["overwrite_mode"],
                previews_colour_map = config["previews_colour_map"],
                use_gpu=config["use_gpu"],
            )

    @classmethod
    def read_default_config_file(self):
        filename = find_config()
        with open(filename) as stream:
            yamlMap = yaml.load(stream)
        return yamlMap


def find_config():
    for directory in CONFIG_DIRS:
        p = directory / CONFIG_FILENAME
        if p.is_file():
            return str(p)
    raise FileNotFoundError("no configuration file found")
