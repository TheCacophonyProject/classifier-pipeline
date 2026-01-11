"""

Author Giampaolo Ferraro

Date December 2020

Some tools to evaluate a model

"""

import argparse
import logging
import sys
import time
import matplotlib.ticker as mtick
from config.config import Config
import os

import json

# from config.config import Config
# from ml_tools.kerasmodel import (
#     KerasModel,
#     plot_confusion_matrix,
#     get_dataset,
#     get_excluded,
# )
from classify.trackprediction import TrackPrediction

from ml_tools import tools
from ml_tools.rawdb import RawDatabase
from ml_tools.dataset import filter_clip, filter_track
import pytz
from dateutil.parser import parse as parse_date
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from config.buildconfig import BuildConfig
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool
import multiprocessing
from dateutil.parser import parse as parse_date
from ml_tools.interpreter import get_interpreter_from_path, ModelMeta

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
    labels = labels.copy()
    if "None" not in labels:
        labels.append("None")

    # if "unidentified" not in labels:
    #     labels.append("unidentified")
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
    total_score = 0
    for l_i, l in enumerate(labels):
        # if l in ["static", "animal", "deer", "sheep"]:
        # continue
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
        other_animals = 1 - (fp_acc + none_acc + unid_acc + accuracy)
        if np.sum(cm[l_i]) == 0:
            other_animals = 0
        if l == "bird":
            # other_animals = 1 - (fp_acc + none_acc + unid_acc + accuracy)
            print(
                "Bird score is ",
                accuracy,
                " fp ",
                fp_acc,
                " none",
                none_acc,
                " unid ",
                unid_acc,
                " other animals",
                other_animals,
            )
            score = accuracy * 1.2 - other_animals
        elif l in ["vehicle", "wallaby"]:
            score = accuracy * 0.8
        elif l in ["mustelid", "human"]:
            score = accuracy * 0.9
        elif l not in ["None", "unidentified"]:
            score = accuracy * 1

        print(
            f"score for {l} is {score} unid {unid_acc} other animasl {round(other_animals,2)}"
        )
        total_score += score
    logging.info("Model accuracy score is %s", total_score)


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
        action="store_true",
        help="Use metadata to produce a confusion matrix",
    )

    parser.add_argument(
        "confusion",
        help="Confusion matrix filename, used if you want to save confusion matrix image",
    )
    parser.add_argument(
        "--threshold",
        default=0.8,
        type=float,
        help="Prediction threshold default 0.5",
    )

    parser.add_argument(
        "--model-score",
        help="Model score calculation for this numpy file",
    )

    parser.add_argument(
        "--best-threshold",
        action="store_true",
        help="calculate best threshold for model",
    )

    parser.add_argument(
        "--prediction-results",
        help="Results npy file to calculate from",
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
    from ml_tools.kerasmodel import (
        plot_confusion_matrix,
        get_excluded,
    )
    from track.region import Region

    confusion_file = Path(confusion_file)
    with open("label_paths.json", "r") as f:
        label_paths = json.load(f)
    label_mapping = get_mappings(label_paths)
    if model_metadata is not None and Path(model_metadata).exists():
        with open(model_metadata, "r") as t:
            # add in some metadata stats
            model_meta = json.load(t)
        logging.info("Loaded metadata from %s", model_metadata)
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
            "false-positive",
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
    remapped_labels["rat"] = "rodent"
    remapped_labels["mouse"] = "rodent"
    remapped_labels["bird/kiwi"] = "kiwi"

    logging.info(
        "Labels are %s excluded %s remapped %s",
        labels,
        excluded_labels,
        remapped_labels,
    )

    y_true = []
    y_pred = []
    median_areas = []
    dir = Path(dir)
    for cptv_file in dir.glob(f"**/*cptv"):
        meta_f = cptv_file.with_suffix(".txt")
        if not meta_f.exists():
            continue
        meta_data = None
        try:
            with open(meta_f, "r") as t:
                # add in some metadata stats
                meta_data = json.load(t)
        except:
            logging.error("Couldnt load %s", cptv_file, exc_info=True)
            continue
        if after_date is not None:
            rec_time = parse_date(meta_data["recordingDateTime"])
            if rec_time <= after_date:
                continue
        tracks_meta = meta_data.get("Tracks")
        if tracks_meta is None:
            tracks_meta = meta_data.get("tracks")
        if tracks_meta is None:
            continue
        for track in tracks_meta:
            tags = track.get("tags", [])
            human_tags = [
                tag.get("what") for tag in tags if tag.get("automatic") == False
            ]
            human_tags = set(human_tags)
            if len(human_tags) > 1:
                logging.info("Conflicting tags for %s %s", track.get("id"), cptv_file)
            if len(human_tags) == 0:
                # print("No humans in ", meta_f)
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
                    if "model" in tag:
                        data = tag["model"]
                    else:
                        data = tag.get("data", {})
                    if isinstance(data, str):
                        if data == "Master":
                            ai_tags.append(tag["what"])
                    elif model.get("name") == "Master":
                        ai_tags.append(tag["what"])

            positions = [
                Region.region_from_json(pos).area for pos in track["positions"]
            ]
            median_area = np.median(positions)
            median_areas.append(median_area)
            y_true.append(human_tag)

            if human_tag not in labels:
                labels.append(human_tag)
            if len(ai_tags) == 0:
                y_pred.append("None")
            else:
                ai_tag = ai_tags[0]
                if ai_tag in ["rat", "mouse"]:
                    ai_tag = "rodent"
                y_pred.append(ai_tag)
                if ai_tag not in labels:
                    labels.append(ai_tag)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    median_areas = np.array(median_areas)
    median = None
    prev_median = 0
    all_labels = LabelGraph()
    # indices =y_true == y_pred

    # print("True vs pred",y_true,y_pred, y_true == y_pred, median_areas[indices],y_true[indices])

    label_graphs = {}
    for l in labels:
        label_graphs[l] = LabelGraph()
    unid_index = labels.index("unidentified")
    for width in range(4, 41):
        # if width == 40:
        #     median = 160 * 120
        # else:
        median = width * width
        print("doing median ", median)
        indices = (median_areas > prev_median) & (median_areas <= median)

        med_y_true = y_true[indices]
        if len(med_y_true) == 0:
            # all_labels.blank(median)
            prev_median = median
            # for i, l in enumerate(labels):
            #     label_graphs[l].blank(median)
            continue

        med_y_pred = y_pred[indices]
        cm = confusion_matrix(med_y_true, med_y_pred, labels=labels)

        all_total = 0
        all_correct = 0
        all_unid = 0
        all_incorrect = 0
        for i, l in enumerate(labels):
            total = np.sum(cm[i])
            correct = cm[i][i]
            if total == 0:
                continue
            print("Adding correct for ", l, correct, total, median)
            unided = cm[i][unid_index]
            incorrect = total - correct - unided
            label_graphs[l].add(median, correct, incorrect, unided, total)

            all_total += total
            all_correct += correct
            all_unid += unided
            all_incorrect += incorrect
        all_labels.add(median, all_correct, all_incorrect, all_unid, all_total)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(
            cm, class_names=labels, title=f"{prev_median} - {median} Median Area"
        )

        med_file = confusion_file.parent / f"{confusion_file.stem}-{median}"
        plt.savefig(med_file.with_suffix(".png"), format="png")
        np.save(med_file.with_suffix(".npy"), cm)
        prev_median = median

    for lbl, lbl_graph in label_graphs.items():

        graph_file = (
            confusion_file.parent / f"{confusion_file.stem}-{lbl.replace('/','-')}"
        )
        lbl_graph.plot(f"{lbl} Median vs Accuracy", graph_file)

    graph_file = confusion_file.parent / f"{confusion_file.stem}-all"
    all_labels.plot(f"All Median vs Accuracy", graph_file)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(confusion_file, format="png")
    np.save(confusion_file.with_suffix(".npy"), cm)

    # cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    # cm = np.nan_to_num(cm)
    model_score(cm, labels)


EXCLUDED_TAGS = ["poor tracking", "part", "untagged", "unidentified"]
worker_model = None
after_date = None


def init_worker(model_file, weights, date):
    global worker_model, after_date
    import tensorflow as tf

    try:
        worker_model = get_interpreter_from_path(model_file)
        if weights is not None:
            worker_model.model.load_weights(weights)
        after_date = date
    except:
        logging.error("init_worker error", exc_info=True)


def load_clip_data(cptv_file):
    # for clip in dataset.clips:
    reason = {}
    clip_db = RawDatabase(cptv_file)
    clip = clip_db.get_clip_tracks(BuildConfig.DEFAULT_GROUPS)
    if clip is None:
        logging.warn("No clip for %s", cptv_file)
        return None

    if (
        filter_clip(clip, None, None, reason, after_date=after_date)
        or len(clip.tracks) == 0
    ):
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
    preprocess_data = []
    for track in clip.tracks:
        try:
            samples = worker_model.frames_for_prediction(
                clip, track, frames_per_classify=25, dont_filter=True, min_segments=1
            )

            frames, preprocessed, masses = worker_model.preprocess(
                clip_db,
                track,
                samples,
                frames_per_classify=25,
                dont_filter=True,
                min_segments=1,
            )
            output = None
            if len(preprocessed) > 0:
                preprocess_data.extend(preprocessed)

            data.append(
                [
                    f"{track.clip_id}-{track.get_id()}",
                    track.label,
                    frames,
                    len(preprocessed),
                    masses,
                ]
            )
        except:
            logging.error("Could not load %s", clip.clip_id, exc_info=True)
    if len(preprocess_data) > 0:
        preprocess_data = np.array(preprocess_data)
        output = worker_model.predict(preprocess_data)
        pred_pos = 0
        for i in range(len(data)):
            num_preds = data[i][3]
            if num_preds == 0:
                continue
            preds = output[pred_pos : pred_pos + num_preds]
            assert len(preds) == num_preds
            data[i][3] = preds
            # print(len(preds),"Setting data preds ",pred_pos,"-", num_preds+pred_pos, " total preds are ", len(output))
            pred_pos += num_preds

    return data


def load_split_file(split_file):
    with open(split_file, "r") as f:
        split = json.load(f)
    return split


def evaluate_dir(
    model_file,
    model_weights,
    dir,
    config,
    confusion_file,
    split_file=None,
    split_dataset="test",
    threshold=0.5,
    after_date=None,
):
    # is faster to run multiple models on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    from ml_tools.kerasmodel import plot_confusion_matrix

    model = ModelMeta(model_file)
    confusion_file = Path(confusion_file)
    if model.params.excluded_labels is not None:
        excluded_labels = model.params.excluded_labels

    if model.params.remapped_labels is not None:
        remapped_labels = model.params.remapped_labels
    for k, v in remapped_labels.items():
        if v == "land-bird":
            remapped_labels[k] = "bird"
    remapped_labels["rat"] = "rodent"
    remapped_labels["mouse"] = "rodent"
    remapped_labels["bird/kiwi"] = "kiwi"
    print("remapped is ", remapped_labels, " excluded", excluded_labels)

    # with open(model_file.with_suffix(".txt"), "r") as t:
    #     # add in some metadata stats
    #     model_meta = json.load(t)
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
    # files = files[:1]
    start = time.time()
    processed = 0
    # quite faster with just one process for loading and using main process for predicting

    pool = Pool(
        processes=8,
        initializer=init_worker,
        initargs=(
            model_file,
            model_weights,
            after_date,
        ),
        maxtasksperchild=500,
    )
    raw_preds = []
    raw_confs = []
    raw_class_confidences = []
    try:

        stats = {"correct": [], "incorrect": [], "low-confidence": []}
        for clip_data in pool.imap_unordered(load_clip_data, files, chunksize=20):
            if processed % 100 == 0:
                logging.info("Procesed %s / %s", processed, len(files))
            processed += 1
            if clip_data is None:
                continue
            for data in clip_data:
                label = data[1]
                output = data[3]
                if output is None:
                    logging.info("No data found for %s", data[0])
                    raw_preds.append("None")
                    raw_confs.append(0)
                    raw_class_confidences.append(np.zeros(len(model.labels)))
                    y_true.append(label_mapping.get(label, label))
                    y_pred.append("None")
                    continue

                prediction = TrackPrediction(data[0], model.labels, smooth_preds=False)
                masses = np.array(data[4])
                masses = masses[:, None]
                top_score = None
                if model.params.multi_label is True:
                    #     # every label could be 1 for each prediction
                    top_score = np.sum(masses)
                #     smoothed = output
                # else:
                prediction.classified_track(output, data[2], masses)
                y_true.append(label_mapping.get(label, label))
                predicted_labels = [prediction.predicted_tag()]
                confidence = prediction.max_score
                raw_preds.append(prediction.predicted_tag())
                raw_confs.append(confidence)
                raw_class_confidences.append(prediction.class_best_score)

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
                    if predicted_labels[0] == y_true[-1]:
                        stats["low-confidence"].append(data[0])
                    else:
                        stats["incorrect"].append(data[0])
                    logging.info(
                        "%s predicted %s but should be %s with confidence %s",
                        data[0],
                        y_pred[-1],
                        label,
                        np.round(100 * prediction.class_best_score),
                    )
                else:
                    stats["correct"].append(data[0])

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)
    finally:
        pool.close()  # Ensure resources are released
        pool.join()

    stats_f = confusion_file.parent / "stats.json"
    logging.info("Stats saved in %s", stats_f)

    with stats_f.open("w") as f:
        json.dump(stats, f)

    model.labels.append("None")
    filename = confusion_file
    raw_preds_i = [model.labels.index(pred) for pred in raw_preds]
    y_true_i = [
        model.labels.index(y_t) if y_t in model.labels else -1 for y_t in y_true
    ]

    results = np.array(raw_preds)
    confidences = np.array(raw_confs)
    raw_preds_i = np.uint8(raw_preds_i)
    raw_class_confidences = np.array(raw_class_confidences)
    y_true_i = np.array(y_true_i)
    npy_file = filename.parent / f"{filename.stem}-raw.npy"
    logging.info("Saving %s", npy_file)
    with npy_file.open("wb") as f:
        np.save(f, y_true_i)
        np.save(f, raw_preds_i)
        np.save(f, raw_class_confidences)
    print("Y true ", y_true_i)
    print(raw_preds_i)
    # thresholds found from best_score
    thresholds_per_label = [
        0.46797615,
        0.70631117,
        0.2496017,
        0.96398157,
        0.33895272,
        0.9697655,
        0.35740834,
        0.60906386,
        0.88741493,
        0.02124451,
        0.9998618,
        0.6102594,
        0.5604206,
        0.9881419,
        0.98753905,
        0.987157,
    ]
    thresholds_per_label = np.array(thresholds_per_label)
    thresholds_per_label[thresholds_per_label < 0.5] = 0.5
    preds = results.copy()
    for i, threshold in enumerate(thresholds_per_label):
        pred_mask = preds == model.labels[i]
        conf_mask = confidences < threshold
        preds[pred_mask & conf_mask] = "None"

    print("Y true is", y_true, preds)
    cm = confusion_matrix(y_true, preds, labels=model.labels)

    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=model.labels)
    smoothing_file = filename.parent / f"{filename.stem}-fscore"
    plt.savefig(smoothing_file.with_suffix(".png"), format="png")
    np.save(smoothing_file.with_suffix(".npy"), cm)

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    for threshold in thresholds:
        preds = results.copy()
        # set these to None
        preds[confidences < threshold] = "None"
        cm = confusion_matrix(y_true, preds, labels=model.labels)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=model.labels)
        smoothing_file = filename.parent / f"{filename.stem}-{round(100*threshold)}%"
        plt.savefig(smoothing_file.with_suffix(".png"), format="png")
        np.save(smoothing_file.with_suffix(".npy"), cm)

    # model.labels.append("None")
    model.labels.append("unidentified")
    cm = confusion_matrix(y_true, y_pred, labels=model.labels)
    npy_file = confusion_file.with_suffix(".npy")
    logging.info("Saving %s", npy_file)
    np.save(str(npy_file), cm)

    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=model.labels)
    plt.savefig(confusion_file.with_suffix(".png"), format="png")
    logging.info("Saving %s", confusion_file.with_suffix(".png"))

    model_score(cm, model.labels)


min_tag_clarity = 0.2
min_tag_confidence = 0.8


# # was used to save model architecture differently so that the predictions were the same
# def re_save(model_file, weights, config):
#     tf.random.set_seed(1)
#     model = KerasModel(train_config=config.train)
#     model.load_model(model_file.parent)
#     model.model.load_weights(weights).expect_partial()
#     model = model.model

#     output = model.layers[1].output
#     for new_l in model.layers[2:]:
#         print(new_l.name)
#         output = new_l(output)
#     new_model = tf.keras.models.Model(model.layers[1].input, outputs=output)

#     print("Saving", model_file.parent / "re_save")
#     new_model.summary()
#     new_model.save(model_file.parent / "re_save")


def main():
    args = load_args()
    init_logging()
    config = Config.load_from_file(args.config_file)
    print("Loading config", args.config_file)
    if args.model_score is not None:
        logging.info(
            "Running model score on %s and metadata %s",
            args.model_score,
            args.model_metadata,
        )
        with open(args.model_metadata, "r") as t:
            # add in some metadata stats
            model_meta = json.load(t)
        cm = np.load(args.model_score)
        model_score(cm, model_meta["labels"])
        return
    weights = None
    if args.model_file:
        model_file = Path(args.model_file)
    if args.weights:
        weights = model_file / args.weights
    base_dir = Path(config.base_folder) / "training-data"
    # shredhold from res
    threshold_from_res = True
    # add command line params
    if args.prediction_results:
        model = ModelMeta(model_file)
        results_f = Path(args.prediction_results)
        print("Loading results from ", results_f)
        with results_f.open("rb") as f:
            y_true_i = np.load(f)
            raw_preds_i = np.load(f)
            confidences = np.load(f)

        thresholds = best_threshold(
            model.labels, y_true_i, raw_preds_i, confidences, Path(args.confusion)
        )
        confusion_for_thresholds(
            thresholds,
            model.labels,
            y_true_i,
            raw_preds_i,
            confidences,
            Path(args.confusion),
        )
        return

    if args.evaluate_dir and args.confusion_from_meta:
        metadata_confusion(
            Path(args.evaluate_dir),
            Path(args.confusion),
            args.date,
            args.model_metadata,
        )
    else:
        if args.evaluate_dir:

            # logging.info("Loading weights %s", weights)
            # if weights is not None:
            #     model.model.load_weights(weights)

            evaluate_dir(
                model_file,
                weights,
                Path(args.evaluate_dir),
                config,
                args.confusion,
                args.split_file,
                args.dataset,
                threshold=args.threshold,
                after_date=args.date,
            )
        elif args.dataset:
            model = get_interpreter_from_path(model_file)

            if weights is None:
                acc = (
                    "val_acc.weights.h5"
                    if model.params.multi_label
                    else "val_acc.weights.h5"
                )
                weights = [
                    "final",
                    model_file.parent / "val_loss.weights.h5",
                    model_file.parent / acc,
                ]
            else:
                weights = [weights]
            model_labels = model.labels.copy()
            model.load_training_meta(base_dir)
            # # model.labels = model_labels
            # if model.params.multi_label:
            #     model.labels.append("land-bird")

            # tf complains if you import tf before tflite
            from ml_tools.kerasmodel import (
                get_dataset,
                get_excluded,
            )

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
            base_confusion_file = Path(args.confusion)
            base_confusion_file = base_confusion_file.parent / base_confusion_file.stem
            for weight in weights:
                if weight != "final":
                    logging.info("Loading weights %s", weight)
                    model.model.load_weights(weight)
                    weight_name = weight.stem
                    suffix_start = weight_name.index(".weights")
                    weight_name = weight_name[:suffix_start]
                    confusion_final = (
                        base_confusion_file.parent
                        / f"{base_confusion_file.stem}-{weight_name}"
                    )
                else:
                    logging.info("Using final weights")
                    confusion_final = (
                        base_confusion_file.parent / f"{base_confusion_file.stem}-final"
                    )
                if args.best_threshold:
                    best_threshold_for_ds(
                        model.model, model.labels, dataset, confusion_final
                    )
                else:
                    model.confusion_tracks(
                        dataset, confusion_final, threshold=args.threshold
                    )


class LabelGraph:
    def __init__(self):
        self.correct = []
        self.incorrect = []
        self.unid = []
        self.x_ticks = []
        self.counts = []

    def blank(self, tick):
        self.x_ticks.append(tick)
        self.correct.append(0)
        self.incorrect.append(0)
        self.unid.append(0)

    def add(self, tick, c, i, u, total):
        self.counts.append(total)
        self.x_ticks.append(tick)
        # change to percent
        c = c / total
        i = i / total
        u = u / total
        self.correct.append(c)
        self.incorrect.append(i)
        self.unid.append(u)

    def plot(self, title, out_file):

        self.x_ticks = np.array(self.x_ticks)
        # print("Plotting for ", title, self.correct,self.incorrect, self.unid)
        plt.clf()
        plt.close("all")
        fig, ax = plt.subplots(figsize=(20, 20))

        ax.plot(self.x_ticks, self.correct, label="Correct", color="g", marker="o")
        ax.plot(
            self.x_ticks,
            self.incorrect,
            label="In correct",
            color="r",
            marker="o",
            alpha=0.5,
        )
        ax.plot(
            self.x_ticks,
            self.unid,
            label="Unidentified",
            color="b",
            marker="o",
            alpha=0.5,
        )

        total = np.sum(self.counts)
        count_percent = np.array(self.counts) / total
        ax.plot(
            self.x_ticks,
            count_percent,
            label="Percentage of data",
            color="black",
            alpha=0.5,
        )

        x_labels = []
        for count, tick in zip(self.counts, self.x_ticks):
            x_labels.append(f"{tick} ({count})")
        ax.set_xticks(self.x_ticks, x_labels, rotation=90)
        plt.subplots_adjust(bottom=0.2)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        plt.legend()
        plt.xlabel("Median area of track bounding boxes and ( # records)")
        plt.ylabel("Percent")
        plt.savefig(out_file.with_suffix(".png"), format="png")


def best_threshold_for_ds(model, labels, dataset, filename):
    import tensorflow as tf

    # sklearn.metrics.auc(
    y_pred = model.predict(dataset)

    # true_categories = [y[0] for x, y in dataset]
    # logging.info("Shape is %s", true_categories.shape)
    # true_categories = tf.concat(true_categories, axis=0)
    # logging.info("Shape is %s", true_categories.shape)

    true_categories = []
    track_ids = []
    avg_mass = []
    for x, y in dataset:
        true_categories.extend(y[0].numpy())
        # dataset_y[0]
        track_ids.extend(y[1].numpy())
        avg_mass.extend(y[2].numpy())
    true_categories = np.array(true_categories)
    true_categories = np.int64(tf.argmax(true_categories, axis=1))

    # make per track
    pred_per_track = {}

    flat_y = []
    for y, track_id, mass, p in zip(true_categories, track_ids, avg_mass, y_pred):
        y_max = y
        track_pred = pred_per_track.setdefault(
            track_id, (y_max, TrackPrediction(track_id, labels))
        )
        track_pred[1].classified_frame(None, p, mass)

    confidences = []
    y_pred = []
    for y, pred in pred_per_track.values():
        pred.normalize_score()
        y_pred.append(pred.class_best_score)
        flat_y.append(y)
        y_pred.append(pred.best_label_index)
    flat_y = np.array(flat_y)
    y_pred = np.array(y_pred)

    confidences = np.array(confidences)
    true_categories = np.array(flat_y)
    best_threshold(labels, true_categories, y_pred, confidences, filename)


def confusion_for_thresholds(
    thresholds_per_label, model_labels, y_true, y_pred, confidences, filename
):
    from ml_tools.kerasmodel import plot_confusion_matrix

    logging.info("Running confusion for thresholds %s", thresholds_per_label)
    max_conf = []

    if len(confidences.shape) > 1:
        # multi class confidence
        max_conf = np.max(confidences, axis=1)
        assert len(max_conf) == len(y_pred)
    else:
        max_conf = confidences
    thresholds_per_label = np.array(thresholds_per_label)
    thresholds_per_label[thresholds_per_label < 0.5] = 0.5

    thresholds_per_label[thresholds_per_label > 0.9] = 0.9
    thresholds_per_label = np.round(thresholds_per_label, 3)
    # thresholds_per_label[:]=0.5
    print(thresholds_per_label)
    labels = model_labels.copy()
    preds = y_pred.copy()
    if "None" not in labels:
        labels.append("None")
    none_index = labels.index("None")
    for i, threshold in enumerate(thresholds_per_label):
        pred_mask = preds == i
        conf_mask = max_conf < threshold
        preds[pred_mask & conf_mask] = none_index

    print("Y true is", y_true, preds)
    cm = confusion_matrix(y_true, preds, labels=np.arange(len(labels)))

    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=labels)
    smoothing_file = filename.parent / f"{filename.stem}-fscorev3"
    plt.savefig(smoothing_file.with_suffix(".png"), format="png")
    np.save(smoothing_file.with_suffix(".npy"), cm)


def best_threshold(labels, y_true, y_pred, confidences, filename):
    from sklearn.metrics import precision_recall_curve, RocCurveDisplay

    from sklearn.preprocessing import LabelBinarizer

    print("Y_true is ", y_true.shape)
    print("Y_pred is ", y_pred.shape)
    print("Confidences ", confidences.shape)

    label_binarizer = LabelBinarizer().fit(y_true)
    y_onehot_test = label_binarizer.transform(y_true)
    thresholds_best = []
    for i, class_of_interest in enumerate(labels):
        print("Class ", class_of_interest)
        lbl_mask = y_true == i
        if len(y_true[lbl_mask]) == 0:
            thresholds_best.append(0)
            continue
        binary_true = np.uint8(lbl_mask)

        if len(confidences.shape) == 1:
            # just best lbl confidence
            lbl_pred = confidences.copy()
            lbl_pred[~lbl_mask] = 0
        else:
            lbl_pred = confidences[:, i]
            # print("CHooisng all of this labl", lbl_pred)
        print("plt show for", class_of_interest)

        precision, recall, thresholds = precision_recall_curve(binary_true, lbl_pred)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)

        scatters = []
        for t_i, th in enumerate(thresholds):
            if th >= 0.6 and len(scatters) == 0:
                scatters.append((t_i, th))
            if th >= 0.7 and len(scatters) == 1:
                scatters.append((t_i, th))
            if th >= 0.8 and len(scatters) == 2:
                scatters.append((t_i, th))
                break
        no_skill = len(binary_true[lbl_mask]) / len(binary_true)

        plt.plot(recall, precision, marker=".", label="Logistic")
        plt.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
        plt.axis("square")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Recall vs Precision - {labels[i]}")
        plt.legend()
        plt.scatter(recall[ix], precision[ix], marker="o", color="black", label="Best")

        colours = ["red", "yellow", "green"]
        for point, colour in zip(scatters, colours):
            plt.scatter(
                recall[point[0]],
                precision[point[0]],
                marker="o",
                color=colour,
                label=f"TX {point[1]}",
            )
            print("plotted ", point, " with colour ", colour)
        label_f = filename.parent / f"{filename.stem}-{labels[i]}.png"
        plt.savefig(label_f, format="png")
        plt.clf()
        print("Best Threshold=%f, F-Score=%.3f" % (thresholds[ix], fscore[ix]))
        thresholds_best.append(thresholds[ix])

    thresholds = np.array(thresholds_best)
    logging.info(
        "ALl thresholds are %s mean %s median %s",
        thresholds,
        np.mean(thresholds),
        np.median(thresholds),
    )
    return thresholds


if __name__ == "__main__":
    # this makes tensorflow work under processes
    multiprocessing.set_start_method("spawn")

    main()
