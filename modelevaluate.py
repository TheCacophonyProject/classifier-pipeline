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
from ml_tools.kerasmodel import KerasModel, plot_confusion_matrix, get_dataset
from ml_tools import tools
from ml_tools.trackdatabase import TrackDatabase
import tensorflow as tf
from ml_tools.dataset import Dataset
import pytz
from ml_tools.datasetstructures import SegmentType
from dateutil.parser import parse as parse_date
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

crop_rectangle = tools.Rectangle(0, 0, 160, 120)
PROB_THRESHOLD = 0.8


def evaluate_db_clips(model, config, after_date, confusion_file="tracks-confusion"):
    if confusion_file is None:
        confusion_file = "tracks-confusion"
    db_file = os.path.join(config.tracks_folder, "dataset.hdf5")
    dataset = Dataset(db_file, "dataset", config)
    dataset.segment_type = SegmentType.ALL_SECTIONS
    tracks_loaded, total_tracks = dataset.load_clips(after_date=after_date)
    logging.info("Samples / Tracks/ Bins/ weight")

    for label in dataset.labels:
        logging.info("%s %s %s %s %s", label, *dataset.get_counts(label))

    samples_by_track = {}
    for s in dataset.samples:
        key = f"{s.clip_id}-{s.track_id}"
        if key in samples_by_track:
            samples_by_track[key].append(s)
        else:
            samples_by_track[key] = [s]

    actual = []
    predicted = []
    probs = []
    for samples in samples_by_track.values():
        s = samples[0]
        background = dataset.db.get_clip_background(s.clip_id)
        track_data = dataset.db.get_track(s.clip_id, s.track_id, channels=[0])
        for f in track_data:
            sub_back = f.region.subimage(background)
            f.filtered = f.thermal - sub_back
            f.resize_with_aspect(
                (model.params.frame_size, model.params.frame_size),
                crop_rectangle,
                True,
            )
        logging.debug(
            f"Evaluating {s.clip_id}-{s.track_id} as {s.label} with {len(samples)} samples"
        )
        for s in samples:
            # make relative
            s.frame_numbers = s.frame_numbers - s.start_frame
        prediction = model.classify_track_data(s.track_id, track_data, samples)
        logging.debug(prediction.description())
        actual.append(s.label)
        predicted.append(prediction.predicted_tag())
        probs.append(prediction.max_score)
    actual = np.array(actual)
    predicted = np.array(predicted)
    probs = np.array(probs)
    logging.info("Saving confusion matrix to %s.png", confusion_file)
    cm = confusion_matrix(actual, predicted, labels=model.labels)
    figure = plot_confusion_matrix(cm, class_names=model.labels)
    plt.savefig(f"{confusion_file}.png", format="png")

    logging.info(
        "Saving predictions above %s confusion matrix to %s-confident.png",
        PROB_THRESHOLD,
        confusion_file,
    )

    probs_mask = probs < PROB_THRESHOLD
    # set all below threshold to be wrong
    predicted[probs_mask] = -1
    cm = confusion_matrix(actual, predicted, labels=model.labels)
    figure = plot_confusion_matrix(cm, class_names=model.labels)
    plt.savefig(f"{confusion_file}-confident.png", format="png")


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
        default="test",
        help="Dataset to use train, validation, test ( Default)",
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
    parser.add_argument("-d", "--date", help="Use clips after this")

    args = parser.parse_args()
    if args.date:
        args.date = parse_date(args.date)
        args.date = args.date.replace(tzinfo=pytz.UTC)
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


def main():
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
    # model = None
    if args.track_id or args.clip_id:
        if args.clip_id is None:
            logging.error("Need clip id and track id")
            sys.exit(0)
        db = TrackDatabase(os.path.join(config.tracks_folder, "dataset.hdf5"))
        evaluate_db_clip(model, db, args.clip_id, args.track_id)
        sys.exit(0)

    if args.tracks:
        # def evaluate_db_clips(model, db, classifier, config, after_date):

        evaluate_db_clips(model, config, args.date, args.confusion)
        sys.exit(0)
    if config.train.tfrecords:
        model.load_training_meta(base_dir)

        files = base_dir + f"/training-data/{args.dataset}"
        dataset, _ = get_dataset(
            files,
            model.type,
            model.labels,
            batch_size=model.params.batch_size,
            image_size=model.params.output_dim[:2],
            preprocess_fn=model.preprocess_fn,
            augment=False,
            resample=False,
            only_features=model.params.mvm,
            one_hot=False,
            deterministic=True,
            reshuffle=False,
        )
        logging.info(
            "Dataset loaded %s, using labels %s",
            args.dataset,
            model.labels,
        )

    else:
        dataset = pickle.load(open(os.path.join(base_dir, args.dataset + ".dat"), "rb"))
        logging.info("running on %s ", dataset.name)

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
        logging.info("%s %s / %s", "label", "samples", "tracks")
        for label in dataset.labels:
            samples, tracks, _, _ = dataset.get_counts(label)
            logging.info("%s/ %s / %s", label, samples, tracks)

        logging.info("Mapped labels")
        for label in dataset.label_mapping.keys():
            logging.info(
                "%s",
                "{} {:<20}".format(
                    label,
                    dataset.mapped_label(label),
                    "{}/{}/{:.1f}".format(*dataset.get_counts(label)),
                ),
            )
    if args.confusion:
        if config.train.tfrecords:
            model.confusion_tfrecords(dataset, args.confusion)
        else:
            model.confusion(dataset, args.confusion)
    else:
        model.evaluate(dataset)


if __name__ == "__main__":
    main()
