import argparse
import logging
import pickle
import time
import sys
import os
import json
import joblib
from dateutil.parser import parse
from config.config import Config
from datetime import datetime, timedelta
import numpy as np
import cv2
from multiprocessing import Process, Queue
from ml_tools.dataset import Dataset
from ml_tools.kerasmodel import KerasModel
from ml_tools.imageprocessing import normalize, detect_objects
import matplotlib
import matplotlib.pyplot as plt
from ml_tools import tools
from ml_tools.datasetstructures import TrackHeader


def evaluate_dataset(model, dataset, tracks=False):
    results = model.evaluate(dataset)
    print("Dataset", dataset_file, "loss,acc", results)


def evaluate_db_track(model, db, clip_id, track_id):
    clip_meta = db.get_clip_meta(clip_id)
    track_meta = db.get_track_meta(clip_id, track_id)
    predictions = db.get_track_predictions(clip_id, track_id)
    track_header = TrackHeader.from_meta(clip_id, clip_meta, track_meta, predictions)
    return evaluate_track(classifier, track_header), classifier.labels


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-file",
        help="Path to model file to use, will override config model",
    )
    parser.add_argument("-w", "--weights", help="Weights to load into model")

    parser.add_argument(
        "-s",
        "--dataset",
        default="test.dat",
        help="Dataset to use train.dat, validate.dat, test.dat ( Default)",
    )
    parser.add_argument("--confusion", help="Save confusion matrix image")
    parser.add_argument(
        "--tracks", action="count", help="Evaluate whole track rather than samples"
    )
    parser.add_argument("--type", type=int, help="training type")

    parser.add_argument("-d", "--date", help="Use clips after this")
    parser.add_argument("-c", "--config-file", help="Path to config file to use")

    parser.add_argument("--track-id", help="Track id")

    parser.add_argument("--clip-id", help="Clip id")

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
else:
    model_file = config.classify.model

date = None
if args.date:
    date = parse(args.date)
base_dir = config.tracks_folder

dataset = joblib.load(open(os.path.join(base_dir, args.dataset), "rb"))
logging.info("running on %s ", dataset.name)

classifier = KerasModel(train_config=config.train)
classifier.load_model(model_file, training=False, weights=args.weights)

dataset.recalculate_segments(segment_type=1)

dir = os.path.dirname(model_file)
meta = json.load(open(os.path.join(dir, "metadata.txt"), "r"))
mapped_labels = meta.get("mapped_labels")
label_probabilities = meta.get("label_probabilities")
dataset.lbl_p = label_probabilities
if mapped_labels:
    dataset.regroup(mapped_labels)

logging.info(
    "Dataset loaded %s, using labels %s, mapped labels %s",
    dataset.name,
    dataset.labels,
    dataset.label_mapping,
)
logging.info("%s %s / %s / %s", "label", "segments", "frames", "tracks")
for label in dataset.labels:
    segments, frames, tracks, _, _ = dataset.get_counts(label)
    logging.info("%s %s / %s / %s", label, segments, frames, tracks)

logging.info("Mapped labels")
for label in dataset.label_mapping.keys():
    logging.info(
        "%s",
        "{} {:<20} {:<20}".format(
            label,
            dataset.mapped_label(label),
            "{}/{}/{}/{:.1f}".format(*dataset.get_counts(label)),
        ),
    )
if args.confusion is not None:
    if args.tracks:
        model.track_confusion(dataset, output_file)
    else:
        model.confusion(dataset, args.confusion)
else:
    ev.evaluate_dataset(dataset, args.tracks)
