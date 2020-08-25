from ml_tools.framedataset import dataset_db_path

import numpy as np
import pickle
from dateutil.parser import parse
import argparse
import logging
import os
import sys
from config.config import Config
from datetime import datetime

from ml_tools.kerasmodel import KerasModel
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
        logging.info("classifier loading %s", model_file)

        self.classifier = KerasModel(train_config=self.config.train)
        self.classifier.load_weights(model_file)

        logging.info("classifier loaded ({})".format(datetime.now() - t0))

    def save_confusion(self, dataset_file, output_file):
        datasets = pickle.load(open(dataset_file, "rb"))
        self.classifier.confusion(datasets[1], output_file)

    def evaluate_dataset(self, dataset_file, tracks=False):
        datasets = pickle.load(open(dataset_file, "rb"))
        dataset = datasets[2]
        dataset.db = self.db
        dataset.binarize(
            ["wallaby"],
            lbl_one="wallaby",
            lbl_two="not",
            keep_fp=False,
            scale=False,
            shuffle=False,
        )
        for label in dataset.labels:
            print(label, dataset.get_counts(label))
        if tracks:
            self.evaluate_tracks(dataset)
            return

        print()
        results = self.classifier.evaluate(dataset)
        print("Dataset", dataset_file, "loss,acc", results)

    def evaluate_tracks(self, dataset):
        labels = dataset.labels
        stats = {}
        total = 0
        for track in dataset.tracks:
            tag = track.label
            if tag != "wallaby":
                continue
            if not tag:
                continue
            if labels and tag not in labels:
                continue
            total += 1
            print("Classifying clip", track.clip_id, "track", track.track_id)

            stat = stats.setdefault(tag, {"correct": 0, "incorrect": []})
            track_data = self.db.get_track(track.clip_id, track.track_id)
            track_prediction = self.classifier.classify_track(
                track.track_id, track_data, regions=track.track_bounds
            )
            mean = np.mean(track_prediction.original, axis=0)
            if track_prediction.best_label_index is not None:
                predicted_lbl = self.classifier.labels[
                    track_prediction.best_label_index
                ]
            else:
                predicted_lbl = "nothing"
            print(
                "tagged as",
                tag,
                "label",
                predicted_lbl,
                " accuracy:",
                track_prediction.score(),
                " rolling average:",
                mean,
            )

            # if tag != wallaby and predicted_lbl == "not"
            if track_prediction.score() is None or track_prediction.score() < 0.85:
                predicted_lbl = "notconfident"
            if predicted_lbl == tag:
                stat["correct"] += 1
            else:
                stat["incorrect"].append(predicted_lbl)
            # break
        print("total is", total)
        print(stats)
        # break


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-file",
        help="Path to model file to use, will override config model",
    )
    parser.add_argument("-t", "--dataset", help="Dataset file to use")
    parser.add_argument("--confusion", help="Save confusion matrix image")
    parser.add_argument(
        "--tracks", action="count", help="Evaluate whole track rather than samples"
    )

    parser.add_argument("-d", "--date", help="Use clips after this")
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
date = None
if args.date:
    date = parse(args.date)

if args.dataset:
    dataset_file = args.dataset
else:
    dataset_file = dataset_db_path(config)
if args.confusion is not None:
    ev.save_confusion(dataset_file, args.confusion)
else:
    ev.evaluate_dataset(dataset_file, args.tracks)
