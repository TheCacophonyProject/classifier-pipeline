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
        "source_folder",
        "tracks_folder",
        "excluded_folders",
        "overwrite_mode",
        "previews_colour_map",
    ],
)

class Config(configTuple):
    @classmethod
    def load(self):
        filename = find_config()
        return self.load_from(filename)

    @classmethod
    def load_from(self, filename):
        with open(filename) as stream:
            y = yaml.load(stream)
            base_folder = y["base_data_folder"]
            return self(
                tracking=TrackingConfig.load(y["tracking"]),
                source_folder = path.join(base_folder, y["source_folder"]),
                tracks_folder = path.join(base_folder, y["tracks_folder"]),
                excluded_folders = path.join(base_folder, y["excluded_folders"]),
                overwrite_mode = path.join(base_folder, y["overwrite_mode"]),
                previews_colour_map = path.join(base_folder, y["previews_colour_map"]),
            )

def find_config():
    for directory in CONFIG_DIRS:
        p = directory / CONFIG_FILENAME
        if p.is_file():
            return str(p)
    raise FileNotFoundError("no configuration file found")
