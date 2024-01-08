from pathlib import Path
import os.path as path

import attr
import logging
import yaml

from .loadconfig import LoadConfig
from .trackingconfig import TrackingConfig
from .trainconfig import TrainConfig
from .classifyconfig import ClassifyConfig
from .buildconfig import BuildConfig
from .defaultconfig import DefaultConfig, deep_copy_map_if_key_not_exist

CONFIG_FILENAME = "classifier.yaml"
CONFIG_DIRS = [Path("/etc/cacophony"), Path(__file__).parent.parent]


@attr.s
class Config(DefaultConfig):
    DEFAULT_LABELS = [
        "bird",
        "cat",
        "false-positive",
        "hedgehog",
        "insect",
        "leporidae",
        "mustelid",
        "possum",
        "rodent",
        "wallaby",
    ]
    base_folder = attr.ib()
    load = attr.ib()
    labels = attr.ib()
    build = attr.ib()
    tracking = attr.ib()
    train = attr.ib()
    classify = attr.ib()
    reprocess = attr.ib()
    previews_colour_map = attr.ib()
    worker_threads = attr.ib()
    debug = attr.ib()
    use_opt_flow = attr.ib()
    verbose = attr.ib()

    @classmethod
    def load_from_file(cls, filename=None):
        if filename is None or not Path(filename).exists():
            filename = find_config()
        if filename is None:
            return Config.get_defaults()
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
        deep_copy_map_if_key_not_exist(default.as_dict(), raw)
        base_folder = Path(raw.get("base_data_folder", "."))
        return cls(
            base_folder=Path(base_folder),
            tracking=TrackingConfig.load(raw["tracking"]),
            load=LoadConfig.load(raw["load"]),
            train=TrainConfig.load(raw["train"], base_folder),
            classify=ClassifyConfig.load(raw["classify"]),
            reprocess=raw["reprocess"],
            previews_colour_map=raw["previews_colour_map"],
            worker_threads=raw["worker_threads"],
            labels=raw["labels"],
            build=BuildConfig.load(raw["build"]),
            debug=raw["debug"],
            use_opt_flow=raw["use_opt_flow"],
            verbose=raw["verbose"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            base_folder=".",
            labels=Config.DEFAULT_LABELS,
            reprocess=True,
            previews_colour_map="custom_colormap.dat",
            worker_threads=0,
            build=BuildConfig.get_defaults(),
            tracking=TrackingConfig.get_defaults(),
            load=LoadConfig.get_defaults(),
            train=TrainConfig.get_defaults(),
            classify=ClassifyConfig.get_defaults(),
            debug=False,
            use_opt_flow=False,
            verbose=False,
        )

    def validate(self):
        self.build.validate()
        for tracker in self.tracking.values():
            tracker.validate()
        self.load.validate()
        self.train.validate()
        self.classify.validate()
        return True

    def as_dict(self):
        return attr.asdict(self)


def find_config():
    for directory in CONFIG_DIRS:
        p = directory / CONFIG_FILENAME
        logging.info("Looking for config %s", p)
        if p.is_file():
            return str(p)
    return None


def parse_options_param(name, value, options):
    if value is None:
        lower_value = value
    else:
        lower_value = value.lower()
    if lower_value not in options:
        raise Exception(
            "Cannot parse {} as '{}'.  Valid options are {}.".format(
                name, value, options
            )
        )
    return lower_value
