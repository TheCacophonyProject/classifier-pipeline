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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import pickle
from config.config import Config
from ml_tools.kerasmodel import (
    KerasModel,
    plot_confusion_matrix,
    get_dataset,
    get_excluded,
)
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
from ml_tools.preprocess import preprocess_ir, preprocess_frame
from ml_tools.frame import Frame
from ml_tools import imageprocessing
import cv2
from config.loadconfig import LoadConfig
from sklearn.metrics import confusion_matrix

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

    parser.add_argument("-c", "--config-file", help="Path to config file to use")

    parser.add_argument("-d", "--date", help="Use clips after this")

    parser.add_argument(
        "confusion",
        help="Confusion matrix filename, used if you want to save confusion matrix image",
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


def evalute_prod_confusion(dir, confusion_file):
    with open("label_paths.json", "r") as f:
        label_paths = json.load(f)
    label_mapping = get_mappings(label_paths)

    labels = [
        "bird",
        "bird/kiwi",
        "cat",
        "false-positive",
        "hedgehog",
        "human",
        "insect",
        "leporidae",
        "mustelid",
        "penguin",
        "possum",
        "rodent",
        "sheep",
        "vehicle",
        "wallaby",
        "None",
        "unidentified",
    ]
    y_true = []
    y_pred = []
    dir = Path(dir)
    for cptv_file in dir.glob(f"**/*cptv"):
        meta_f = cptv_file.with_suffix(".txt")
        meta_data = None
        with open(meta_f, "r") as t:
            # add in some metadata stats
            meta_data = json.load(t)

        for track in meta_data.get("Tracks", []):
            tags = track.get("tags", [])
            human_tags = [
                tag.get("what")
                for tag in tags
                if tag.get("automatic") == False
                # and tag.get("what", "") not in LoadConfig.EXCLUDED_TAGS
            ]
            human_tags = set(human_tags)
            if len(human_tags) > 1:
                print("Conflicting tags for ", track.get("id"), cptv_file)
            if len(human_tags) == 0:
                print("No humans in ", tags)
                continue
            human_tag = human_tags.pop()
            human_tag = label_mapping.get(human_tag, human_tag)
            ai_tag = [
                tag.get("what")
                for tag in tags
                if tag.get("automatic") is True
                and tag.get("data", {}).get("name") == "Inc3 RF"
            ]
            y_true.append(human_tag)
            if len(ai_tag) == 0:
                y_pred.append("None")
            else:
                y_pred.append(ai_tag[0])

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(confusion_file, format="png")


def evaluate_dir(
    model,
    dir,
    config,
    confusion_file,
):
    with open("label_paths.json", "r") as f:
        label_paths = json.load(f)
    label_mapping = get_mappings(label_paths)
    # dataset = Dataset(
    #     dir,
    #     "dataset",
    #     config,
    #     consecutive_segments=False,
    #     label_mapping=label_mapping,
    #     raw=True,
    #     ext=".cptv",
    # )
    #
    # tracks_loaded, total_tracks = dataset.load_clips(dont_filter_segment=True)
    reason = {}
    y_true = []
    y_pred = []
    files = list(dir.glob(f"**/*cptv"))
    files.sort()
    for cptv_file in files:
        # for clip in dataset.clips:
        clip_db = RawDatabase(cptv_file)
        clip = clip_db.get_clip_tracks(LoadConfig.DEFAULT_GROUPS)
        if clip is None:
            logging.warn("No clip for %s", cptv_file)
            continue

        if filter_clip(clip, reason):
            logging.info("Filtering %s", cptv_file)
            continue
        clip.tracks = [
            track
            for track in clip.tracks
            if not filter_track(track, config.load.excluded_tags, reason)
        ]
        if len(clip.tracks) == 0:
            logging.info("No tracks after filtering %s", cptv_file)
            continue
        clip_db.load_frames()
        segment_frame_spacing = int(round(clip.frames_per_second))
        thermal_medians = []
        for f in clip_db.frames:
            thermal_medians.append(np.median(f.thermal))
        thermal_medians = np.uint16(thermal_medians)

        for track in clip.tracks:
            track.calculate_segments(
                segment_frame_spacing * 2,
                model.params.square_width**2,
                segment_min_mass=10,
                segment_type=SegmentType.ALL_SECTIONS,
                ffc_frames=clip_db.ffc_frames,
            )
            frame_indices = set()
            for sample in track.samples:
                frame_indices.update(set(sample.frame_indices))
                sample.remapped_label = label_mapping.get(
                    sample.original_label, sample.original_label
                )
                track.label = sample.remapped_label
            frame_indices = list(frame_indices)
            frame_indices.sort()
            track_frames = {}
            # frames = []
            for i in frame_indices:
                f = clip_db.frames[i]
                f.region = track.regions_by_frame[f.frame_number]
                pre_f = preprocess_frame(
                    f, (32, 32), f.region, clip_db.background, crop_rectangle
                )
                track_frames[i] = pre_f

            # for f in frames:
            #     region = track.regions_by_frame[f.frame_number]
            #     track_frame = f.crop_by_region(region)
            #     track_frame.region = region
            #     track_frames[region.frame_number] = track_frame
            #
            # min_diff, max_diff = filter_diffs(track_frames.values(), clip_db.background)
            # for f in track_frames.values():
            #     f.float_arrays()
            #     f.filtered = f.thermal - f.region.subimage(clip_db.background)
            #     f.filtered, stats = imageprocessing.normalize(
            #         f.filtered, min=min_diff, max=max_diff, new_max=255
            #     )
            #     f.thermal -= thermal_medians[f.frame_number]
            #     np.clip(f.thermal, a_min=0, a_max=None, out=f.thermal)
            #     f.thermal, stats = imageprocessing.normalize(f.thermal, new_max=255)
            #     if f.thermal.size > 0:
            #         f.resize_with_aspect(
            #             (32, 32),
            #             crop_rectangle,
            #             True,
            #         )

            prediction = model.classify_track_data(
                track.track_id, track_frames, track.samples
            )
            y_true.append(track.label)
            if prediction.predicted_tag() is None:
                y_pred.append("None")
            else:
                # to compare to prod add conf rules
                if (
                    prediction.clarity > min_tag_clarity
                    and prediction.max_score > min_tag_confidence
                ):
                    y_pred.append(prediction.predicted_tag())
                else:
                    y_pred.append("unidentified")
            print(
                track,
                "Got a prediction of",
                y_pred[-1],
                " should be ",
                track.label,
                np.round(100 * prediction.predictions),
            )
    model.labels.append("None")
    model.labels.append("unidentified")

    cm = confusion_matrix(y_true, y_pred, labels=model.labels)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=model.labels)
    plt.savefig(confusion_file, format="png")


min_tag_clarity = 0.2
min_tag_confidence = 0.8


# trying to figre out differing predictions
def test_model(model_file, weights, config):
    import tensorflow.keras.backend as K

    model = None
    tf.random.set_seed(1)
    model = KerasModel(train_config=config.train)
    model.load_model(model_file.parent)
    # model.model.save_weights(model_file.parent / "final_weights")
    # model.build_model(dropout=0.3)

    # model.model.load_weights(str(weights))
    # print("Loading weigfhts", weights)
    model.model.load_weights(weights)
    model = model.model

    output = model.layers[1].output
    for new_l in model.layers[2:]:
        print(new_l.name)
        output = new_l(output)
    new_model = tf.keras.models.Model(model.layers[1].input, outputs=output)

    print("Saving", model_file.parent / "re_save")
    new_model.summary()
    new_model.save(model_file.parent / "re_save")
    return
    for _ in range(2):
        test = np.ones((1, 160, 160, 3), dtype=np.float32)
        test = test / 2.0
        print("RUN")
        print("RUN")
        print("RUN")
        print("RUN")

        # for l, o in zip(model.layers[1].layers, layer_outs):
        #     print(l.name)
        #     for x in np.array(o).ravel():
        #         print(x)
        print(model.predict(test))
        # print(layer_outs)
    return


def re_save(model_file, weights, config):
    tf.random.set_seed(1)
    model = KerasModel(train_config=config.train)
    model.load_model(model_file.parent)
    # model.model.save_weights(model_file.parent / "final_weights")
    # model.build_model(dropout=0.3)

    # model.model.load_weights(str(weights))
    # print("Loading weigfhts", weights)
    # model.model.save_weight(model_file.parent / "final_weights")
    model.model.load_weights(weights)
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
    # evalute_prod_confusion(args.evaluate_dir, args.confusion)
    print("LOading config", args.config_file)
    weights = None
    if args.model_file:
        model_file = Path(args.model_file)
    if args.weights:
        weights = model_file / args.weights
    # test_model(model_file, Path(weights), config)
    # return
    base_dir = config.tracks_folder
    tf.random.set_seed(1)

    model = KerasModel(train_config=config.train)
    model.load_model(model_file, training=False, weights=weights)

    if args.evaluate_dir:
        evaluate_dir(model, Path(args.evaluate_dir), config, args.confusion)
    elif args.dataset:
        model.load_training_meta(base_dir)
        if model.params.multi_label:
            model.labels.append("land-bird")
        excluded, remapped = get_excluded(model.type)
        files = base_dir + f"/training-data/{args.dataset}"
        dataset, _, new_labels, _ = get_dataset(
            files,
            model.type,
            model.labels,
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
        )
        model.labels = new_labels
        logging.info(
            "Dataset loaded %s, using labels %s",
            args.dataset,
            model.labels,
        )
        model.confusion_tfrecords(dataset, args.confusion)


def evaluate_db_clips(model, config, after_date, confusion_file="tracks-confusion"):
    type = model.type
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
        if type == "IR":
            for frame in track_data:
                preprocessed = preprocess_ir(
                    frame.copy(),
                    (
                        model.params.frame_size,
                        model.params.frame_size,
                    ),
                    False,
                    frame.region,
                    model.preprocess_fn,
                    save_info=f"{frame.region.frame_number} - {frame.region}",
                )
                predictions = model.model.predict(preprocessed[np.newaxis, :])
                best_res = np.argmax(predictions[0])
                prediction = model.labels[best_res]
                if prediction != s.label:
                    if prediction == "false-positive":
                        out_dir = "animal-fp"
                    else:
                        out_dir = "diff-animal"

                    cv2.imwrite(
                        f"{out_dir}/{s.clip_id}-{s.track_id}-{frame.frame_number}-{s.label}-pred-{prediction}.png",
                        frame.thermal,
                    )
                # 1 / 0

        else:
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


if __name__ == "__main__":
    main()
