"""

Author Giampaolo Ferraro

Date December 2020

Some tools to evaluate a model

"""
import argparse
import logging
import pickle
import sys
import os
import json
import pickle
from config.config import Config
from ml_tools.kerasmodel import KerasModel
from ml_tools import tools
from ml_tools.trackdatabase import TrackDatabase
from ml_tools.dataset import SegmentType


def evaluate_db_clip(model, db, classifier, clip_id, track_id=None):
    logging.info("Prediction tracks %s", track)
    clip_meta = db.get_clip_meta(clip_id)
    if track_id is None:
        tracks = db.get_clip_tracks(clip_id)
    else:
        tracks = [track_id]
    for track_id in tracks:
        track_meta = db.get_track_meta(clip_id, track_id)
        track_data = db.get_track(clip_id, track_id)

        regions = []
        medians = clip_meta["frame_temp_median"][
            track_meta["start_frame"] : track_meta["start_frame"] + track_meta["frames"]
        ]
        for region in track_meta.track_bounds:
            regions.append(tools.Rectangle.from_ltrb(*region))
        track_prediction = model.classify_track_data(
            track_id, track_data, medians, regions=regions
        )
        logging.info("Predicted %s", track_prediction.predicted_tag(model.labels))


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
        help="Dataset to use train.dat, validation.dat, test.dat ( Default)",
    )
    parser.add_argument(
        "--confusion",
        help="Confusion matrix filename, used if you want to save confusion matrix image",
    )
    parser.add_argument(
        "--tracks", action="count", help="Evaluate whole track rather than samples"
    )
    parser.add_argument("-c", "--config-file", help="Path to config file to use")

    parser.add_argument("--track-id", help="Track id to evaluate from database")

    parser.add_argument("--clip-id", help="Clip id to evaluate from database")

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


if args.model_file:
    model_file = args.model_file
    weights = args.weights
else:
    model_file = config.classify.models[0].model_file
    weights = config.classify.models[0].model_weights

base_dir = config.tracks_folder


model = KerasModel(train_config=config.train)
model.load_model(model_file, training=False, weights=weights)

if args.track_id or args.clip_id:
    if args.clip_id is None:
        logging.error("Need clip id and track id")
        sys.exit(0)
    db = TrackDatabase(os.path.join(config.tracks_folder, "dataset.hdf5"))
    evaluate_db_clip(model, db, args.clip_id, args.track_id)
    sys.exit(0)

dataset = pickle.load(open(os.path.join(base_dir, args.dataset), "rb"))
logging.info("running on %s ", dataset.name)
dataset.recalculate_segments(segment_type=5)

dir = os.path.dirname(model_file)
meta = json.load(open(os.path.join(dir, "metadata.txt"), "r"))
mapped_labels = meta.get("mapped_labels")
label_probabilities = meta.get("label_probabilities")
dataset.lbl_p = label_probabilities
if mapped_labels:
    dataset.regroup(mapped_labels)
print("dataset labels arenow ", dataset.labels)

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
if args.tracks:
    for i in range(8):
        print("EVAL FOR ", i)
        dataset.recalculate_segments(segment_type=i)

        model.track_accuracy(dataset, args.confusion)
elif args.confusion:
    model.confusion(dataset, args.confusion)
else:
    model.evaluate(dataset)
