# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.

import argparse
import datetime
import os
import pickle
import random

import dateutil
import numpy as np

from ml_tools.logs import init_logging
from ml_tools.trackdatabase import TrackDatabase
from ml_tools.config import Config
from ml_tools.dataset import Dataset, dataset_db_path

# uses split from previous run
USE_PREVIOUS_SPLIT = True

# note: these should really be in a text file or something, or excluded during extraction
BANNED_CLIPS = {
    "20171130-113732-akaroa10.cptv",  # this is a trapped clip, but metadata hasn't been updated.
    "20180103-075552-akaroa09.cptv",
    "20180103-075552-akaroa09.cptv",
    "20171025-020827-akaroa03.cptv",
    "20171025-020827-akaroa03.cptv",
    "20171207-114424-akaroa09.cptv",
    "20171207-114424-akaroa09.cptv",
    "20171207-114424-akaroa09.cptv",
    "20171219-102910-akaroa12.cptv",
    "20171219-105919-akaroa12.cptv",
}

INCLUDED_LABELS = ["bird", "false-positive", "hedgehog", "possum", "rat", "stoat"]

# if true removes any trapped animal footage from dataset.
# trapped footage can be a problem as there tends to be lots of it and the animals do not move in a normal way.
# however, bin weighting will generally take care of the excessive footage problem.
EXCLUDE_TRAPPED = True

# sets a maximum number of segments per bin, where the cap is this many standard deviations above the norm.
# bins with more than this number of segments will be weighted lower so that their segments are not lost, but
# will be sampled less frequently.
CAP_BIN_WEIGHT = 1.5

# adjusts the weight for each animal class.  Setting this lower for animals that are less represented can help
# with training, otherwise the few examples we have will be used excessively.  This also helps build a prior for
# the class suggesting that the class is more or less likely.  For example bumping up the human weighting will cause
# the classifier lean towards guessing human when it is not sure.

# xxx this doesn't actually work, and should be removed.
LABEL_WEIGHTS = {"bird-kiwi": 0.1}

# clips after this date will be ignored.
# note: this is based on the UTC date.
END_DATE = datetime.datetime(2019, 12, 31)

# minimum average mass for test segment
TEST_MIN_MASS = 30

TRAIN_MIN_MASS = 20

# any day with a track this number of second or longer will be excluded from the validation set.
# this is because it would be more useful to train on the long track, and we don't want the track to dominate the
# validation set (otherwise a single track could end up being 50% of the data)
MAX_VALIDATION_SET_TRACK_DURATION = 120

# number of segments to include in test set for each class (multiplied by label weights)
TEST_SET_COUNT = 300

# minimum number of bins used for test set
TEST_SET_BINS = 10

filtered_stats = {"confidence": 0, "trap": 0, "banned": 0, "date": 0}


def track_filter(clip_meta, track_meta):
    # some clips are banned for various reasons
    source = os.path.basename(clip_meta["filename"])
    if source in BANNED_CLIPS:
        filtered_stats["banned"] += 1
        return True

    if track_meta["tag"] not in INCLUDED_LABELS:
        return True

    # filter by date
    if dateutil.parser.parse(clip_meta["start_time"]).date() > END_DATE.date():
        filtered_stats["date"] += 1
        return True

    # always let the false-positives through as we need them even though they would normally
    # be filtered out.
    if track_meta["tag"] == "false-positive":
        return False

    # for some reason we get some records with a None confidence?
    if clip_meta.get("confidence", 0.0) <= 0.6:
        filtered_stats["confidence"] += 1
        return True

    # remove tracks of trapped animals
    if (
        "trap" in clip_meta.get("event", "").lower()
        or "trap" in clip_meta.get("trap", "").lower()
    ):
        filtered_stats["trap"] += 1
        return True

    return False


def show_tracks_breakdown():
    print("Tracks breakdown:")
    for label in dataset.labels:
        count = len([track for track in dataset.tracks_by_label[label]])
        print("  {:<20} {} tracks".format(label, count))


def show_cameras_breakdown():
    print("Cameras breakdown")
    tracks_by_camera = {}
    for track in dataset.tracks:
        if track.camera not in tracks_by_camera:
            tracks_by_camera[track.camera] = []
        tracks_by_camera[track.camera].append(track)

    for camera, tracks in tracks_by_camera.items():
        print("{:<20} {}".format(camera, len(tracks)))


def show_segments_breakdown():
    print("Segments breakdown:")
    for label in dataset.labels:
        count = sum([len(track.segments) for track in dataset.tracks_by_label[label]])
        print("  {:<20} {} segments".format(label, count))


def split_dataset_predefined():
    """Splits dataset into train / test / validation using a predefined set of camera-label allocations.

    This method puts all tracks into 'camera-label' bins and then assigns certain bins to the validation set.
    In this case the 'test' set is simply a subsample of the validation set. (as there are not enough unique cameras
    at this point for 3 sets).

    The advantages of this method are

    1/ The datasets are fixed, even as more footage comes through
    2/ Learning an animal on one camera / environment and predicting it on another is a very good indicator that that
        algorthim has generalised
    3/ Some adjustment can be made to make sure that the cameras used in the test / validation sets contain 'reasonable'
        footage. (i.e. footage that is close to what we want to classify.

    The disadvantages are that it requires seeing animals on multiple cameras, which we don't always have data for.
    It can also be a bit wasteful as we sometimes dedicate a substantial portion of the data to some animal types.

    A possible solution would be to note when (or if) the model over-fits then run on the entire dataset.

    Another option would be k-fold validation.
    """

    validation_bins = [
        ("bird", "akaroa03"),
        ("hedgehog", "akaroa09"),
        ("hedgehog", "akaroa03"),
        ("hedgehog", "brent01"),
        ("possum", "brent01"),
        ("possum", "akaroa13"),
        ("cat", "akaroa03"),
        ("cat", "akaroa13"),
        ("cat", "akaroa04"),
        ("cat", "zip02"),
        ("stoat", "akaroa09"),
        ("stoat", "zip03"),
        ("human", "akaroa09"),
        ("rat", "akaroa04"),
        ("rat", "akaroa10"),
        ("rat", "zip01"),
        ("false-positive", "akaroa09"),
    ]

    raise Exception("not implemented yet.")


def is_heavy_bin(bin_id, max_bin_segments):
    bin_segments = sum(len(track.segments) for track in dataset.tracks_by_bin[bin_id])
    max_track_duration = max(track.duration for track in dataset.tracks_by_bin[bin_id])
    return (
        bin_segments > max_bin_segments
        or max_track_duration > MAX_VALIDATION_SET_TRACK_DURATION
    )


def split_dataset_days(prefill_bins=None):
    """
    Randomly selects tracks to be used as the train, validation, and test sets
    :param prefill_bins: if given will use these bins for the test set
    :return: tuple containing train, validation, and test datasets.

    This method assigns tracks into 'label-camera-day' bins and splits the bins across datasets.
    """

    # pick out groups to use for the various sets
    bins_by_label = {}
    for label in dataset.labels:
        bins_by_label[label] = []

    for bin_id, tracks in dataset.tracks_by_bin.items():
        label = dataset.track_by_id[tracks[0].track_id].label
        bins_by_label[label].append(bin_id)

    train = Dataset(db, "train")
    validation = Dataset(db, "validation")
    test = Dataset(db, "test")

    train.labels = dataset.labels.copy()
    validation.labels = dataset.labels.copy()
    test.labels = dataset.labels.copy()

    # check bins distribution
    bin_segments = []
    for bin, tracks in dataset.tracks_by_bin.items():
        count = sum(len(track.segments) for track in tracks)
        bin_segments.append((count, bin))
    bin_segments.sort()

    counts = [count for count, bin in bin_segments]
    bin_segment_mean = np.mean(counts)
    bin_segment_std = np.std(counts)
    max_bin_segments = bin_segment_mean + bin_segment_std * CAP_BIN_WEIGHT

    print()
    print(
        "Bin segment mean:{:.1f} std:{:.1f} auto max segments:{:.1f}".format(
            bin_segment_mean, bin_segment_std, max_bin_segments
        )
    )
    print()

    used_bins = {}

    for label in dataset.labels:
        used_bins[label] = []

    if prefill_bins is not None:
        print("Reusing bins from previous split:")
        for label in dataset.labels:
            available_bins = set(bins_by_label[label])
            if label not in prefill_bins:
                continue
            for sample in prefill_bins[label]:
                # this happens if we have banned/deleted the clip, but it was previously used.
                if sample not in dataset.tracks_by_bin:
                    continue
                # this happens if we changed what a 'heavy' bin is.
                if is_heavy_bin(sample, max_bin_segments):
                    continue

                validation.add_tracks(dataset.tracks_by_bin[sample])
                test.add_tracks(dataset.tracks_by_bin[sample])
                validation.filter_segments(
                    TEST_MIN_MASS, ignore_labels=["false-positive"]
                )
                test.filter_segments(TEST_MIN_MASS, ignore_labels=["false-positive"])

                available_bins.remove(sample)
                used_bins[label].append(sample)

            for bin_id in available_bins:
                train.add_tracks(dataset.tracks_by_bin[bin_id])
                train.filter_segments(TRAIN_MIN_MASS, ignore_labels=["false-positive"])

    # assign bins to test and validation sets
    # if we previously added bins from another dataset we are simply filling in the gaps here.
    for label in dataset.labels:

        available_bins = set(bins_by_label[label])

        # heavy bins are bins with an unsually high number of examples on a day.  We exclude these from the test/validation
        # set as they will be subfiltered down and there is no need to waste that much data.
        heavy_bins = set()
        for bin_id in available_bins:
            if is_heavy_bin(bin_id, max_bin_segments):
                heavy_bins.add(bin_id)

        available_bins -= heavy_bins
        available_bins -= set(used_bins[label])

        # print bin statistics
        print(
            "{}: normal {} heavy {} pre-filled {}".format(
                label, len(available_bins), len(heavy_bins), len(used_bins[label])
            )
        )

        required_samples = TEST_SET_COUNT * LABEL_WEIGHTS.get(label, 1.0)
        required_bins = TEST_SET_BINS * LABEL_WEIGHTS.get(
            label, 1.0
        )  # make sure there is some diversity
        required_bins = max(4, required_bins)

        # we assign bins to the test and validation sets randomly until we have enough segments + bins
        # the remaining bins can be used for training
        while len(available_bins) > 0 and (
            validation.get_class_segments_count(label) < required_samples
            or len(used_bins[label]) < required_bins
        ):

            sample = random.sample(available_bins, 1)[0]

            validation.add_tracks(dataset.tracks_by_bin[sample])
            test.add_tracks(dataset.tracks_by_bin[sample])

            validation.filter_segments(TEST_MIN_MASS, ignore_labels=["false-positive"])
            test.filter_segments(TEST_MIN_MASS, ignore_labels=["false-positive"])

            available_bins.remove(sample)
            used_bins[label].append(sample)

            if prefill_bins is not None:
                print(" - required added adddtional sample ", sample)

        available_bins.update(heavy_bins)

        for bin_id in available_bins:
            train.add_tracks(dataset.tracks_by_bin[bin_id])
            train.filter_segments(TRAIN_MIN_MASS, ignore_labels=["false-positive"])

    print("Segments per class:")
    print("-" * 90)
    print("{:<20} {:<21} {:<21} {:<21}".format("Class", "Train", "Validation", "Test"))
    print("-" * 90)

    # if we have lots of segments on a single day, reduce the weight so we don't overtrain on this specific
    # example.
    train.balance_bins(max_bin_segments)
    validation.balance_bins(bin_segment_mean + bin_segment_std * CAP_BIN_WEIGHT)

    # balance out the classes
    train.balance_weights(weight_modifiers=LABEL_WEIGHTS)
    validation.balance_weights(weight_modifiers=LABEL_WEIGHTS)
    test.balance_resample(
        weight_modifiers=LABEL_WEIGHTS, required_samples=TEST_SET_COUNT
    )

    # display the dataset summary
    for label in dataset.labels:
        print(
            "{:<20} {:<20} {:<20} {:<20}".format(
                label,
                "{}/{}/{}/{:.1f}".format(*train.get_counts(label)),
                "{}/{}/{}/{:.1f}".format(*validation.get_counts(label)),
                "{}/{}/{}/{:.1f}".format(*test.get_counts(label)),
            )
        )
    print()

    return train, validation, test


def get_bin_split(filename):
    """Loads bin splits from previous database. """
    train, validation, text = pickle.load(open(filename, "rb"))
    test_bins = {}
    for label in validation.labels:
        test_bins[label] = set()
        for track in validation.tracks_by_label[label]:
            test_bins[label].add(track.bin_id)
        test_bins[label] = list(test_bins[label])
    return test_bins


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    args = parser.parse_args()
    return Config.load_from_file(args.config_file)


def main():
    init_logging()

    global dataset
    global db

    config = load_config()

    db = TrackDatabase(os.path.join(config.tracks_folder, "dataset.hdf5"))
    dataset = Dataset(db, "dataset")

    total_tracks = len(db.get_all_track_ids())

    tracks_loaded = dataset.load_tracks(track_filter)

    print(
        "Loaded {}/{} tracks, found {:.1f}k segments".format(
            tracks_loaded, total_tracks, len(dataset.segments) / 1000
        )
    )
    for key, value in filtered_stats.items():
        if value != 0:
            print("  {} filtered {}".format(key, value))
    print()

    labels = sorted(list(set(dataset.tracks_by_label.keys())))
    dataset.labels = labels

    show_tracks_breakdown()
    print()
    show_segments_breakdown()
    print()
    show_cameras_breakdown()
    print()

    print("Splitting data set into train / validation")
    if USE_PREVIOUS_SPLIT:
        split = get_bin_split("template.dat")
        datasets = split_dataset_days(split)
    else:
        datasets = split_dataset_days()

    pickle.dump(datasets, open(dataset_db_path(config), "wb"))


if __name__ == "__main__":
    main()
