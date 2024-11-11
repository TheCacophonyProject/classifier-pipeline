"""

Author Giampaolo Ferraro

Date December 2020

Some tools to evaluate a model

"""

import cv2
import argparse
import logging
import pickle
import sys
import os
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import pickle
from config.config import Config
from ml_tools.kerasmodel import (
    KerasModel,
    plot_confusion_matrix,
    get_dataset,
    get_excluded,
)
from classify.trackprediction import TrackPrediction

from ml_tools import tools
from ml_tools.trackdatabase import TrackDatabase
from ml_tools.rawdb import RawDatabase
import tensorflow as tf
from ml_tools.dataset import Dataset, filter_clip, filter_track
import pytz
from ml_tools.datasetstructures import SegmentType
from dateutil.parser import parse as parse_date
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ml_tools.preprocess import preprocess_frame
from ml_tools.frame import Frame
from ml_tools import imageprocessing
import cv2
from config.buildconfig import BuildConfig
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool
from dateutil.parser import parse as parse_date


root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
        root_logger.removeHandler(handler)
crop_rectangle = tools.Rectangle(0, 0, 160, 120)
PROB_THRESHOLD = 0.8

land_birds = [
    "pukeko",
    "california quail",
    "brown quail",
    "black swan",
    "quail",
    "pheasant",
    "penguin",
    "duck",
]


# basic formula to give a number to compare models
def model_score(cm, labels):
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm = np.nan_to_num(cm)
    fp_index = None
    if "false-positive" in labels:
        fp_index = labels.index("false-positive")
    none_index = None
    unid_index = None
    if "None" in labels:
        none_index = labels.index("None")
    if "unidentified" in labels:
        unid_index = labels.index("unidentified")
    score = 0
    for l_i, l in enumerate(labels):
        fp_acc = 0
        if fp_index is not None:
            fp_acc = cm[l_i][fp_index]
        none_acc = 0
        unid_acc = 0
        accuracy = cm[l_i][l_i]
        if none_index:
            none_acc = cm[l_i][none_index]
        if unid_index:
            unid_acc = cm[l_i][unid_index]
        if l == "bird":
            other_animals = 1 - (fp_acc + none_acc + unid_acc)
            score += accuracy * 1.2 - other_animals
        elif l in ["vehicle", "wallaby"]:
            score += accuracy * 0.8
        elif l in ["mustelid", "human"]:
            score += accuracy * 0.9
        elif l not in ["None", "unidentified"]:
            score += accuracy * 1
    logging.info("Model accuracy score is %s", score)


def get_mappings(label_paths):
    regroup = {}
    for l, path in label_paths.items():
        if l in land_birds:
            regroup[l] = l
            continue
        split_path = path.split(".")
        if len(split_path) == 1:
            regroup[l] = l
        elif path.startswith("all.mammal"):
            if len(split_path) == 4:
                regroup[l] = split_path[-2]
            else:
                regroup[l] = l
        else:
            # print("l", l, " has ", path)
            parent = split_path[-2]
            # print("Parent is", parent, path)
            if parent == "kiwi" or split_path[-1] == "kiwi":
                regroup[l] = "kiwi"
            elif parent == "other":
                regroup[l] = l

            else:
                if "bird." in path:
                    regroup[l] = "bird"

                elif len(split_path) > 2:
                    regroup[l] = split_path[-3]
                else:
                    regroup[l] = split_path[-1]

    return regroup


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
        "--evaluate-dir",
        help="Evalute directory of cptv files",
    )
    parser.add_argument(
        "--model-metadata",
        help="Meta data file for model, used with confusion from meta",
    )
    parser.add_argument("-c", "--config-file", help="Path to config file to use")

    parser.add_argument("-d", "--date", help="Use clips after this")

    parser.add_argument("--split-file", help="Use split for evaluation")
    parser.add_argument(
        "--confusion-from-meta",
        action="count",
        help="Use metadata to produce a confusion matrix",
    )

    parser.add_argument(
        "confusion",
        help="Confusion matrix filename, used if you want to save confusion matrix image",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Prediction threshold default 0.5",
    )
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


def filter_diffs(track_frames, background):
    max_diff = 0
    min_diff = 0
    for f in track_frames:
        r = f.region
        if r.blank or r.width <= 0 or r.height <= 0:
            continue

        diff_frame = np.float32(f.thermal) - r.subimage(background)
        new_max = np.amax(diff_frame)
        new_min = np.amin(diff_frame)
        if new_min < min_diff:
            min_diff = new_min
        if new_max > max_diff:
            max_diff = new_max
    return min_diff, max_diff


# evaluate a confusion matrix from metadata of files, already evaluated by our current model on browse


def metadata_confusion(dir, confusion_file, after_date=None, model_metadata=None):
    with open("label_paths.json", "r") as f:
        label_paths = json.load(f)
    label_mapping = get_mappings(label_paths)
    if model_metadata is not None and Path(model_metadata).exists():
        with open(model_metadata, "r") as t:
            # add in some metadata stats
            model_meta = json.load(t)
        labels = model_meta.get("labels", [])
        excluded_labels = model_meta.get("excluded_labels", {})
        remapped_labels = model_meta.get("remapped_labels", {})
        for k, v in remapped_labels.items():
            if v == "land-bird":
                remapped_labels[k] = "bird"
        if "None" not in labels:
            labels.append("None")
        if "unidentified" not in labels:
            labels.append("unidentified")
    else:
        labels = [
            "bird",
            "cat",
            "deer",
            "dog",
            "falsepositive",
            "hedgehog",
            "human",
            "kiwi",
            "leporidae",
            "mustelid",
            "penguin",
            "possum",
            "rodent",
            "sheep",
            "vehicle",
            "wallaby",
            "landbird",
            "None",
            "unidentified",
        ]
        excluded_labels, remapped_labels = get_excluded("thermal")
    logging.info(
        "Labels are %s excluded %s remapped %s",
        labels,
        excluded_labels,
        remapped_labels,
    )
    y_true = []
    y_pred = []
    dir = Path(dir)
    for cptv_file in dir.glob(f"**/*cptv"):
        meta_f = cptv_file.with_suffix(".txt")
        if not meta_f.exists():
            continue
        meta_data = None
        with open(meta_f, "r") as t:
            # add in some metadata stats
            meta_data = json.load(t)
        rec_time = parse_date(meta_data["recordingDateTime"])
        if after_date is not None and rec_time <= after_date:
            continue
        for track in meta_data.get("Tracks", []):
            tags = track.get("tags", [])
            human_tags = [
                tag.get("what") for tag in tags if tag.get("automatic") == False
            ]
            human_tags = set(human_tags)
            if len(human_tags) > 1:
                print("Conflicting tags for ", track.get("id"), cptv_file)
            if len(human_tags) == 0:
                print("No humans in ", meta_f)
                continue
            human_tag = human_tags.pop()
            human_tag = label_mapping.get(human_tag, human_tag)
            if human_tag in excluded_labels:
                logging.info("Excluding %s", human_tag)
                continue
            if human_tag in remapped_labels:
                logging.info(
                    "Remapping %s to %s", human_tag, remapped_labels[human_tag]
                )
                human_tag = remapped_labels[human_tag]
            # if human_tag not in labels:
            # logging.info("Excluding %s", human_tag)

            ai_tags = []
            for tag in tags:
                if tag.get("automatic") is True:
                    data = tag.get("data", {})
                    if isinstance(data, str):
                        if data == "Master":
                            ai_tags.append(tag["what"])
                    elif data.get("name") == "Master":
                        ai_tags.append(tag["what"])

            y_true.append(human_tag)
            if human_tag not in labels:
                labels.append(human_tag)
            if len(ai_tags) == 0:
                y_pred.append("None")
            else:
                y_pred.append(ai_tags[0])
                if ai_tags[0] not in labels:
                    labels.append(ai_tags[0])

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(confusion_file, format="png")
    # cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    # cm = np.nan_to_num(cm)
    model_score(cm, labels)


EXCLUDED_TAGS = ["poor tracking", "part", "untagged", "unidentified"]
worker_model = None
after_date = None


def init_worker(model, date):
    global worker_model, after_date
    worker_model = model
    after_date = date


def load_clip_data(cptv_file):
    # for clip in dataset.clips:
    reason = {}
    clip_db = RawDatabase(cptv_file)
    clip = clip_db.get_clip_tracks(BuildConfig.DEFAULT_GROUPS)
    if clip is None:
        logging.warn("No clip for %s", cptv_file)
        return None

    if filter_clip(clip, None, None, reason, after_date=after_date):
        # logging.info("Filtering %s", cptv_file)
        return None
    clip.tracks = [
        track for track in clip.tracks if not filter_track(track, EXCLUDED_TAGS, reason)
    ]
    if len(clip.tracks) == 0:
        logging.info("No tracks after filtering %s", cptv_file)
        return None
    clip_db.load_frames()
    segment_frame_spacing = int(round(clip.frames_per_second))
    thermal_medians = []
    for f in clip_db.frames:
        thermal_medians.append(np.median(f.thermal))
    thermal_medians = np.uint16(thermal_medians)
    data = []
    for track in clip.tracks:
        try:
            frames, preprocessed, masses = worker_model.preprocess(
                clip_db, track, frames_per_classify=25, dont_filter=True, min_segments=1
            )
            data.append(
                (
                    f"{track.clip_id}-{track.get_id()}",
                    track.label,
                    frames,
                    preprocessed,
                    masses,
                )
            )
        except:
            logging.error("Could not load %s", clip.clip_id, exc_info=True)
    return data


def load_split_file(split_file):
    with open(split_file, "r") as f:
        split = json.load(f)
    return split


def evaluate_dir(
    model,
    dir,
    config,
    confusion_file,
    split_file=None,
    split_dataset="test",
    threshold=0.5,
    after_date=None,
):
    logging.info("Evaluating cptv files in %s with threshold %s", dir, threshold)

    with open("label_paths.json", "r") as f:
        label_paths = json.load(f)
    label_mapping = get_mappings(label_paths)
    reason = {}
    y_true = []
    y_pred = []
    if split_file is not None:
        split_json = load_split_file(split_file)
        files = split_json.get(split_dataset)
        files = [dir / f["source"] for f in files]
        logging.info(
            "Splitting on %s dataset %s files %s ...",
            split_file,
            split_dataset,
            files[:2],
        )
    else:
        files = list(dir.glob(f"**/*cptv"))
    files.sort()
    # files = files[:8]
    start = time.time()
    # quite faster with just one process for loading and using main process for predicting
    with Pool(
        processes=1,
        initializer=init_worker,
        initargs=(
            model,
            after_date,
        ),
    ) as pool:
        for clip_data in pool.imap_unordered(load_clip_data, files):
            if clip_data is None:
                continue
            for data in clip_data:
                label = data[1]
                preprocessed = data[3]
                if len(preprocessed) == 0:
                    logging.info("No data found for %s", data[0])
                    y_true.append(label_mapping.get(label, label))
                    y_pred.append("None")
                    continue
                output = model.predict(preprocessed)

                prediction = TrackPrediction(data[0], model.labels)
                masses = np.array(data[4])
                masses = masses[:, None]
                top_score = None
                if model.params.multi_label is True:
                    #     # every label could be 1 for each prediction
                    top_score = np.sum(masses)
                #     smoothed = output
                # else:
                smoothed = output * masses
                prediction.classified_clip(
                    output, smoothed, data[2], masses, top_score=top_score
                )
                y_true.append(label_mapping.get(label, label))
                predicted_labels = [prediction.predicted_tag()]
                confidence = prediction.max_score
                predicted_tag = "None"
                if confidence < threshold:
                    y_pred.append("unidentified")
                elif len(predicted_labels) == 0:
                    y_pred.append("None")
                else:
                    logging.info("Predicted  %s", predicted_labels)
                    predicted_tag = ",".join(predicted_labels)
                    y_pred.append(predicted_tag)
                if y_pred[-1] != y_true[-1]:
                    logging.info(
                        "%s predicted %s but should be %s with confidence %s",
                        data[0],
                        y_pred[-1],
                        label,
                        np.round(100 * prediction.class_best_score),
                    )
    model.labels.append("None")
    model.labels.append("unidentified")
    cm = confusion_matrix(y_true, y_pred, labels=model.labels)
    npy_file = Path(confusion_file).with_suffix(".npy")
    logging.info("Saving %s", npy_file)
    np.save(str(npy_file), cm)

    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=model.labels)
    plt.savefig(confusion_file, format="png")
    logging.info("Saving %s", Path(confusion_file).with_suffix(".png"))

    model_score(cm, model.labels)


min_tag_clarity = 0.2
min_tag_confidence = 0.8


# was used to save model architecture differently so that the predictions were the same
def re_save(model_file, weights, config):
    tf.random.set_seed(1)
    model = KerasModel(train_config=config.train)
    model.load_model(model_file.parent)
    model.model.load_weights(weights).expect_partial()
    model = model.model

    output = model.layers[1].output
    for new_l in model.layers[2:]:
        print(new_l.name)
        output = new_l(output)
    new_model = tf.keras.models.Model(model.layers[1].input, outputs=output)

    print("Saving", model_file.parent / "re_save")
    new_model.summary()
    new_model.save(model_file.parent / "re_save")


def main():
    args = load_args()
    init_logging()
    config = Config.load_from_file(args.config_file)
    print("Loading config", args.config_file)
    weights = None
    if args.model_file:
        model_file = Path(args.model_file)
    if args.weights:
        weights = model_file / args.weights
    base_dir = Path(config.base_folder) / "training-data"
    if args.evaluate_dir and args.confusion_from_meta:
        metadata_confusion(
            Path(args.evaluate_dir), args.confusion, args.date, args.model_metadata
        )
    else:

        model = KerasModel(train_config=config.train)
        model.load_model(model_file, training=False, weights=weights)
        if args.evaluate_dir:
            evaluate_dir(
                model,
                Path(args.evaluate_dir),
                config,
                args.confusion,
                args.split_file,
                args.dataset,
                threshold=args.threshold,
                after_date=args.date,
            )
        elif args.dataset:
            model_labels = model.labels.copy()
            model.load_training_meta(base_dir)
            # model.labels = model_labels
            if model.params.multi_label:
                model.labels.append("land-bird")
            excluded, remapped = get_excluded(model.data_type)

            if model.params.excluded_labels is not None:
                excluded = model.params.excluded_labels

            if model.params.remapped_labels is not None:
                remapped = model.params.remapped_labels

            files = base_dir / args.dataset
            dataset, _, new_labels, _ = get_dataset(
                files,
                model.data_type,
                model.labels,
                model_labels=model_labels,
                batch_size=64,
                image_size=model.params.output_dim[:2],
                preprocess_fn=model.preprocess_fn,
                augment=False,
                resample=False,
                include_features=model.params.mvm,
                one_hot=True,
                deterministic=True,
                shuffle=False,
                excluded_labels=excluded,
                remapped_labels=remapped,
                multi_label=model.params.multi_label,
                include_track=True,
                cache=True,
                channels=model.params.channels,
                num_frames=model.params.square_width**2,
            )
            model.labels = new_labels
            logging.info(
                "Dataset loaded %s, using labels %s",
                args.dataset,
                model.labels,
            )
            model.confusion_tracks(dataset, args.confusion, threshold=args.threshold)


if __name__ == "__main__":
    main()
