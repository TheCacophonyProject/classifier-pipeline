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
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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


if config.train.tfrecords:
    model.load_training_meta(base_dir)

    files = base_dir + f"/training-data/{args.dataset}"
    dataset = model.get_dataset(
        files,
        model.params.batch_size,
        reshuffle=False,
        deterministic=True,
        resample=False,
        stop_on_empty_dataset=False,
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

        # fit bar plot data using curve_fit


def gauss(x, a, b, c):
    # a Gaussian distribution
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


stats = {}
for x, y in dataset:
    prediction = model.model.predict(x)
    prediction = np.argmax(prediction, axis=1)

    for l, p in zip(y, prediction):

        label = l[1].numpy()
        average = l[0].numpy()
        bucket = int(average / 5.0)
        lbl = model.labels[label]
        lbl_stats = stats.setdefault(lbl, {})
        bucket_stats = lbl_stats.setdefault(bucket, {"False": 0, "True": 0})
        bucket_stats[str(label == p)] += 1

for lbl, bins in stats.items():
    bstats = []
    b_keys = list(bins.keys())
    b_keys.sort()
    barKeys = []
    i = 0
    x = []
    y = []
    lbl = lbl.capitalize()

    # for b_key in range(10):
    #     lower = b_key * 5
    #     upper = (b_key * 5) + 5
    #     bstats.append(0)
    #     barKeys.append(f"{lower}-{upper}")
    for b_key in b_keys:
        i = b_key * 5

        b_stat = bins[b_key]
        if (b_stat["False"] + b_stat["True"]) == 0:
            percent = 0
        else:
            percent = 100 * (b_stat["True"] / (b_stat["False"] + b_stat["True"]))
        total = b_stat["False"] + b_stat["True"]
        y.append(i + b_key / 2.0)
        x.append(percent)
        if b_key < len(bstats):
            bstats[b_key] = percent
            barKeys[b_key] += "\n" + f" ({total})"
        else:
            lower = i
            upper = i + 5
            bstats.append(percent)
            barKeys.append(f"{lower}-{upper}\n ({total})")
        i += 5
    fig = plt.figure(figsize=(10, 5))
    # x.append(10)
    # x.append(15)
    #
    # x.append(6)
    # y.append(y[-1] + 5)
    # y.append(y[-1] + 5)
    # y.append(y[-1] + 5)
    #
    # x = np.array(x)
    # y = np.array(y)
    # print("fixing too", x, y)
    if np.sum(x) > 0:
        popt, pcov = curve_fit(gauss, x, y)
        plt.plot(
            x,
            gauss(x, *popt),
            label=lbl,
        )
        # m, b = np.polyfit(x, y, 1)
        # print(m, b)
        # plt.plot(x, m * x + b, label=lbl)

    # creating the bar plot
    plt.bar(barKeys, bstats, width=0.8)
    # fig.subplots_adjust(bottom=0.2)

    plt.xlabel("Average Tracking Width")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy %")
    plt.title(f"{lbl} Accuracy vs Tracking Width")
    plt.savefig(f"{lbl}acc-vs-width.png")
    plt.clf()


print(stats)
sys.exit(0)
if args.tracks:
    model.track_accuracy(dataset, args.confusion)
elif args.confusion:
    if config.train.tfrecords:
        model.confusion_tfrecords(dataset, args.confusion)
    else:
        model.confusion(dataset, args.confusion)
else:
    model.evaluate(dataset)
