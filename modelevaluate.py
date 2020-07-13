import argparse
import logging
import os
import sys
from config.config import Config
from datetime import datetime

from ml_tools.newmodel import NewModel
from ml_tools.trackdatabase import TrackDatabase
from classify.trackprediction import Predictions, TrackPrediction


class ModelEvalute:
    def __init__(self, config, model_file):
        self.model_file = model_file
        self.classifier = None
        self.config = config
        self.load_classifier(model_file)
        self.db = TrackDatabase(os.path.join(config.tracks_folder, "dataset.hdf5"))

    def load_classifier(self, model_file):
        """
        Returns a classifier object, which is created on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        t0 = datetime.now()
        logging.info("classifier loading")

        self.classifier = NewModel(train_config=self.config.train)
        self.classifier.load_model(model_file)
        logging.info("classifier loaded ({})".format(datetime.now() - t0))

    def evaluate(self, labels=None):
        track_ids = self.db.get_all_track_ids()
        for clip_id, track_id in track_ids:
            clip_meta = self.db.get_clip_meta(clip_id)
            track_meta = self.db.get_track_meta(clip_id, track_id)
            tag = track_meta.get("tag")
            if not tag:
                continue
            if labels and tag not in labels:
                continue
            track_data = self.db.get_track(clip_id, track_id)
            track_prediction = self.classifier.classify_track(track_id, track_data)
            print("clip", clip_id, "track", track_id)
            print(
                "tagged as",
                tag,
                "label",
                self.classifier.labels[track_prediction.best_label_index],
                " accuracy:",
                round(track_prediction.score(), 2),
            )
            # break


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-file",
        help="Path to model file to use, will override config model",
    )

    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    args = parser.parse_args()
    return args


def init_logging(timestamps=False):
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(levelname)7s %(message)s"
    if timestamps:
        fmt = "%(asctime)s " + fmt
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


args = load_args()
init_logging()
config = Config.load_from_file(args.config_file)
model_file = config.classify.model
if args.model_file:
    model_file = args.model_file
ev = ModelEvalute(config, model_file)
ev.evaluate()
