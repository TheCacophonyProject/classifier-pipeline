# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.

import argparse
import os
import random
from functools import partial
import datetime
import logging
import pickle
import pytz
import json
from dateutil.parser import parse as parse_date
from ml_tools.logs import init_logging
from config.config import Config
from ml_tools.dataset import Dataset
from ml_tools.datasetstructures import Camera
from ml_tools.tfwriter import create_tf_records
from ml_tools.irwriter import save_data as save_ir_data
from ml_tools.thermalwriter import save_data as save_thermal_data
from ml_tools.tools import CustomJSONEncoder
import attrs
import numpy as np

from pathlib import Path


from build import get_mappings


class Stat:
    def __init__(self, label):
        self.label = label
        self.mins = []
        self.lq = []
        self.median = []
        self.uq = []
        self.maxs = []
        self.timestamps = []
        self.clip_ids = []
        self.track_ids = []
        self.background_median = []
        self.clip_id = None
        self.track_id = None

    def add(
        self, a_min, lq, median, uq, a_max, timestamp, clip_id, track_id, back_median
    ):
        self.mins.append(a_min)
        self.lq.append(lq)
        self.median.append(median)
        self.uq.append(uq)
        self.maxs.append(a_max)
        self.timestamps.append(timestamp)
        self.clip_ids.append(clip_id)
        self.track_ids.append(track_id)
        self.background_median.append(back_median)

    def get_median_stat(self):
        if len(self.mins) == 0:
            return None
        stat = Stat(self.label)
        stat.mins = [np.amin(self.mins)]
        stat.lq = [np.median(self.lq)]
        stat.median = [np.median(self.median)]
        stat.uq = [np.median(self.uq)]
        stat.maxs = [np.amax(self.maxs)]
        stat.background_median = (
            [np.median(self.background_median)] if self.background_median else []
        )
        stat.clip_ids = [self.clip_ids[0]]
        stat.track_ids = [self.track_ids[0]]
        stat.timestamps = [self.timestamps[0]]

        return stat

    def min_min(self):
        return np.amin(self.mins) if self.mins else None

    def max_max(self):
        return np.amax(self.maxs) if self.maxs else None

    def matches(self, other, threshold=120):
        a_min, a_max = self.min_min(), self.max_max()
        other_min, other_max = other.min_min(), other.max_max()
        if None in (a_min, a_max, other_min, other_max):
            return False
        return (
            abs(a_min - other_min) <= threshold and abs(a_max - other_max) <= threshold
        )

    def overlaps(self, other):
        a_min, a_max = self.min_min(), self.max_max()
        other_min, other_max = other.min_min(), other.max_max()
        if None in (a_min, a_max, other_min, other_max):
            return False
        return a_min <= other_max and other_min <= a_max

    def merge(self, other):
        self.mins.extend(other.mins)
        self.lq.extend(other.lq)
        self.median.extend(other.median)
        self.uq.extend(other.uq)
        self.maxs.extend(other.maxs)
        self.timestamps.extend(other.timestamps)
        self.clip_ids.extend(other.clip_ids)
        self.track_ids.extend(other.track_ids)
        self.background_median.extend(other.background_median)

    @classmethod
    def load(cls, file_name):
        file_name = Path(file_name)
        data = np.load(file_name, allow_pickle=True)
        stat = cls(file_name.stem)
        stat.mins = data["mins"].tolist()
        stat.lq = data["lq"].tolist()
        stat.median = data["median"].tolist()
        stat.uq = data["uq"].tolist()
        stat.maxs = data["maxs"].tolist()
        stat.timestamps = data["timestamps"].tolist()
        stat.clip_ids = data["clip_ids"].tolist()
        stat.track_ids = data["track_ids"].tolist()
        stat.background_median = data["background_median"].tolist()
        return stat

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / self.label,
            mins=self.mins,
            lq=self.lq,
            median=self.median,
            uq=self.uq,
            maxs=self.maxs,
            timestamps=self.timestamps,
            clip_ids=self.clip_ids,
            track_ids=self.track_ids,
            background_median=self.background_median,
        )


def load_config(config_file):
    return Config.load_from_file(config_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, help="Config file")

    parser.add_argument("-s", "--seed", type=int, help="Seed to use")

    parser.add_argument("data_dir", type=Path, help="Directory of hdf5 files")
    parser.add_argument("output_dir", type=Path, help="Directory to save npz files")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load stats from output_dir instead of recomputing",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit stats from output_dir instead of recomputing",
    )
    args = parser.parse_args()
    return args


def main():
    from config.buildconfig import BuildConfig

    init_logging()
    args = parse_args()
    if args.fit:

        fit_data(args.output_dir)
    elif args.load:
        print("Loading stats from ", args.output_dir)
        all_stats = {
            p.stem: Stat.load(p, p.stem) for p in sorted(args.output_dir.glob("*.npz"))
        }
        plot_median(all_stats)

    else:
        config = load_config(args.config)
        label_mapping = get_mappings()
        logging.info("Using mappings %s", label_mapping)
        master_dataset = Dataset(
            args.data_dir,
            "dataset",
            config,
            label_mapping=label_mapping,
        )
        master_dataset.load_clips(dont_filter_segment=False, seed=args.seed)
        source_files = filtered_source(master_dataset)
        excluded_tags = BuildConfig.EXCLUDED_TAGS
        all_stats = {}
        from multiprocessing import Pool

        worker = partial(track_overlap_check, excluded_tags=excluded_tags)
        all_track_stats = []
        with Pool() as pool:
            for clip_stats in pool.imap_unordered(worker, source_files):
                all_track_stats.extend(clip_stats)
                # for label, stat in clip_stats.items():
                #     if label not in all_stats:
                #         all_stats[label] = stat
                #     else:
                #         all_stats[label].merge(stat)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        # for k, stat in all_stats.items():
        #     stat.save(args.output_dir)
        with open(args.output_dir / "all_track_stats.pkl", "wb") as f:
            pickle.dump(all_track_stats, f)
        # plot_median(all_stats)


def fit_data(output_dir):
    with open(output_dir / "all_track_stats.pkl", "rb") as f:
        all_track_stats = pickle.load(f)
    by_clip = {}
    labels = set()
    for stat in all_track_stats:
        labels.add(stat.label)
        by_clip.setdefault(stat.clip_id, []).append(stat)
    labels = list(labels)
    labels.sort()
    labels.append("None")
    if output_dir.is_file():

        confusion_file = output_dir.with_stem("confusion.png")
    else:
        confusion_file = output_dir / "confusion.png"
    evaluate_matches(by_clip, labels, confusion_file)
    return by_clip


def evaluate_matches(by_clip, labels, output_file, threshold=120):
    from ml_tools.kerasmodel import plot_confusion_matrix
    import matplotlib.pyplot as plt

    label_index = {label: i for i, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    tp = fp = tn = fn = 0
    for clip_id, stats in by_clip.items():
        for stat in stats:
            for other in stats:
                if other == stat:
                    continue
                should_match = stat.label == other.label
                does_match = stat.matches(other, threshold=threshold)
                j = label_index.get(other.label)

                if should_match and does_match:
                    tp += 1
                elif should_match and not does_match:
                    j = label_index.get("None")

                    fn += 1
                elif not should_match and does_match:
                    fp += 1
                else:
                    tn += 1
                i = label_index.get(stat.label)
                if i is not None and j is not None:
                    confusion[i][j] += 1

    figure = plot_confusion_matrix(
        confusion, class_names=labels, title=f"Temp threshold is {threshold}"
    )
    plt.savefig(output_file, format="png")
    print("Writing to ", output_file)
    # total = tp + fp + tn + fn
    # precision = tp / (tp + fp) if (tp + fp) else 0.0
    # recall = tp / (tp + fn) if (tp + fn) else 0.0
    # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # print(f"threshold={threshold}  pairs={total}  tp={tp} fp={fp} tn={tn} fn={fn}")
    # print(f"  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")
    # print(f"Confusion matrix (rows=true label, cols=matched label):")
    # header = "".join(f"{l:>12}" for l in labels)
    # print(f"{'':>12}{header}")
    # for i, row_label in enumerate(labels):
    #     row = "".join(f"{confusion[i][j]:>12}" for j in range(len(labels)))
    #     print(f"{row_label:>12}{row}")
    # return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall, f1=f1, confusion=confusion)


def filtered_source(dataset):
    source_files = []
    for clip in dataset.clips:
        labels = [
            track.label for track in clip.tracks if track.label != "false-positive"
        ]
        if len(labels) > 1:
            # print("Adding because labels are ", labels)
            source_files.append(clip.source_file)
    return source_files


def plot_median(all_stats):
    print(all_stats)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, stat in all_stats.items():
        medians = [cv_to_celcius(median) for median in stat.median]
        ax.plot(range(len(medians)), medians, label=label)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Median temperature (K)")
    ax.set_title("Median thermal temperature per label")
    ax.legend()
    plt.tight_layout()
    plt.savefig("median_temps.png")
    # plt.show()


def track_overlap_check(source_file, excluded_tags):
    from ml_tools.rawdb import RawDatabase
    from config.buildconfig import BuildConfig
    from ml_tools.dataset import filter_track

    THRESHOLD = 50  # .5Celcius
    db = RawDatabase(source_file)
    db.load_frames()
    clip = db.get_clip_meta(BuildConfig.DEFAULT_GROUPS)
    clip.tracks = [
        track for track in clip.tracks if not filter_track(track, excluded_tags)
    ]
    background = db.get_clip_background()
    back_median = np.median(background)
    all_stats = {}
    logging.info("Loading %s", db.clip_id)

    all_track_stats = []
    for track in clip.tracks:
        if track.label in all_stats:
            label_stats = all_stats[track.label]
        else:
            label_stats = Stat(track.label)

            all_stats[track.label] = label_stats
        track_stats = Stat(track.label)

        for frame_i in range(track.start_frame, track.start_frame + track.num_frames):
            f = db.frames[frame_i]
            region = track.regions_by_frame[frame_i]
            f.crop_by_region(region)
            # f.filtered[f.filtered < THRESHOLD] = 0
            np.clip(f.filtered, a_min=0, a_max=None, out=f.filtered)

            masked_thermal = f.thermal[f.filtered > THRESHOLD]
            if len(masked_thermal) == 0:
                # print(
                #     "Skipping frame ",
                #     frame_i,
                #     cv_to_celcius(27315 + np.amax(f.filtered)),
                # )
                # show_image(f.thermal)
                continue
            img = f.filtered.copy()
            img[f.filtered <= THRESHOLD] = 0
            # show_image(img)
            a_min, q1, median, q3, a_max = np.percentile(
                masked_thermal, [0, 25, 50, 75, 100]
            )
            track_stats.add(
                a_min,
                q1,
                median,
                q3,
                a_max,
                clip.rec_time.timestamp(),
                clip.clip_id,
                track.track_id,
                back_median,
            )

        median_stat = track_stats.get_median_stat()
        if median_stat:
            median_stat.track_id = track.id
            median_stat.clip_id = clip.clip_id
            label_stats.merge(median_stat)
            all_track_stats.append(median_stat)
    all_labels = [track.label for track in clip.tracks]
    label_counts = {label: all_labels.count(label) for label in set(all_labels)}
    # print("All labels ", all_labels)
    # matches = 0

    for stat in all_track_stats:
        matches = 0
        expected_matches = label_counts[stat.label] - 1
        t_id = stat.track_id

        print(
            f"source_file {stat.label} {clip.clip_id}:{t_id}: {cv_to_celcius(stat.min_min())}-{cv_to_celcius(stat.max_max())}"
        )
        for other in all_track_stats:
            if stat == other:
                continue
            if stat.matches(other):
                if stat.label != other.label:
                    print(
                        f"Wrong {stat.label} {clip.clip_id}:{t_id}: {cv_to_celcius(stat.min_min())}-{cv_to_celcius(stat.max_max())} is over laping with {other.label} {other.track_id}: {cv_to_celcius(other.min_min())}-{cv_to_celcius(other.max_max())} "
                    )
                else:
                    matches += 1
                    print(
                        f"match {stat.label} {clip.clip_id}:{t_id}: {cv_to_celcius(stat.min_min())}-{cv_to_celcius(stat.max_max())} is over laping with {other.label} {other.track_id}: {cv_to_celcius(other.min_min())}-{cv_to_celcius(other.max_max())} "
                    )
        if matches != expected_matches:
            print(f"Expect {expected_matches} and got {matches}")
    return all_track_stats


def clip_temp_data(source_file, excluded_tags):
    from ml_tools.rawdb import RawDatabase
    from config.buildconfig import BuildConfig
    from ml_tools.dataset import filter_track

    THRESHOLD = 50  # .5Celcius
    db = RawDatabase(source_file)
    db.load_frames()
    clip = db.get_clip_meta(BuildConfig.DEFAULT_GROUPS)
    clip.tracks = [
        track for track in clip.tracks if not filter_track(track, excluded_tags)
    ]
    background = db.get_clip_background()
    back_median = np.median(background)
    all_stats = {}
    logging.info("Loading %s", db.clip_id)
    for track in clip.tracks:
        if track.label in all_stats:
            label_stats = all_stats[track.label]
        else:
            label_stats = Stat(track.label)

            all_stats[track.label] = label_stats
        track_stats = Stat(track.label)

        for frame_i in range(track.start_frame, track.start_frame + track.num_frames):
            f = db.frames[frame_i]
            region = track.regions_by_frame[frame_i]
            f.crop_by_region(region)
            # f.filtered[f.filtered < THRESHOLD] = 0
            np.clip(f.filtered, a_min=0, a_max=None, out=f.filtered)

            masked_thermal = f.thermal[f.filtered > THRESHOLD]
            if len(masked_thermal) == 0:
                print(
                    "Skipping frame ",
                    frame_i,
                    cv_to_celcius(27315 + np.amax(f.filtered)),
                )
                # show_image(f.thermal)
                continue
            img = f.filtered.copy()
            img[f.filtered <= THRESHOLD] = 0
            # show_image(img)
            a_min, q1, median, q3, a_max = np.percentile(
                masked_thermal, [0, 25, 50, 75, 100]
            )
            track_stats.add(
                a_min,
                q1,
                median,
                q3,
                a_max,
                clip.rec_time.timestamp(),
                clip.clip_id,
                track.track_id,
                back_median,
            )

        median_stat = track_stats.get_median_stat()
        if median_stat:
            label_stats.merge(median_stat)
    return all_stats


def cv_to_celcius(cv):
    return np.float32(cv) / 100 - 273.15


def show_image(img):
    import cv2
    from ml_tools.imageprocessing import normalize

    norm_f, _ = normalize(img, new_max=255)
    norm_f = np.uint8(norm_f)
    cv2.imshow("norm", norm_f)
    cv2.waitKey()


if __name__ == "__main__":
    main()
