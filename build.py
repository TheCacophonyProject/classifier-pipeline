# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.

import argparse
import os
import pickle
import random
import numpy as np

from ml_tools.logs import init_logging
from ml_tools.trackdatabase import TrackDatabase
from config.config import Config
from ml_tools.dataset import Dataset, dataset_db_path


MIN_BINS = 4


def show_tracks_breakdown(dataset):
    print("Tracks breakdown:")
    for label in dataset.labels:
        count = len([track for track in dataset.tracks_by_label[label]])
        print("  {:<20} {} tracks".format(label, count))


def show_cameras_breakdown(dataset):
    print("Cameras breakdown")
    tracks_by_camera = {}
    for track in dataset.tracks:
        if track.camera not in tracks_by_camera:
            tracks_by_camera[track.camera] = []
        tracks_by_camera[track.camera].append(track)

    for camera, tracks in tracks_by_camera.items():
        print("{:<20} {}".format(camera, len(tracks)))


def show_segments_breakdown(dataset):
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


def prefill_bins(
    dataset,
    fill_datasets,
    prefill_bins,
    used_bins,
    max_bin_segments,
    max_validation_track_duration,
):
    print("Reusing bins from previous split:")

    for label in dataset.labels:
        if label not in prefill_bins:
            continue
        normal_bins, heavy_bins = dataset.split_heavy_bins(
            prefill_bins[label], max_bin_segments, max_validation_track_duration
        )

        for sample in normal_bins:
            tracks = dataset.tracks_by_bin[sample]
            for ds in fill_datasets:
                ds.add_tracks(tracks)
        used_bins[label].extend(normal_bins)


def split_dataset(db, dataset, build_config, prefill_dataset=None):
    """
    Randomly selects tracks to be used as the train, validation, and test sets
    :param prefill_bins: if given will use these bins for the test set
    :return: tuple containing train, validation, and test datasets.

    This method assigns tracks into 'label-camera-day' bins and splits the bins across datasets.
    """

    # pick out groups to use for the various sets
    bins_by_label = {}
    used_bins = {}

    for label in dataset.labels:
        bins_by_label[label] = []
        used_bins[label] = []

    counts = []
    for bin_id, tracks in dataset.tracks_by_bin.items():
        label = tracks[0].label
        bins_by_label[tracks[0].label].append(bin_id)
        counts.append(sum(len(track.segments) for track in tracks))

    train = Dataset(db, "train")
    validation = Dataset(db, "validation")
    test = Dataset(db, "test")

    bin_segment_mean = np.mean(counts)
    bin_segment_std = np.std(counts)
    max_bin_segments = bin_segment_mean + bin_segment_std * build_config.cap_bin_weight

    print_bin_segment_stats(bin_segment_mean, bin_segment_std, max_bin_segments)

    max_track_duration = build_config.max_validation_set_track_duration
    if prefill_dataset is not None:
        prefill_bins(
            dataset,
            [validation, test],
            prefill_dataset,
            used_bins,
            max_bin_segments,
            max_track_duration,
        )

    required_samples = build_config.test_set_count
    required_bins = max(MIN_BINS, build_config.test_set_bins)

    # assign bins to test and validation sets
    # if we previously added bins from another dataset we are simply filling in the gaps here.
    for label in dataset.labels:
        available_bins = set(bins_by_label[label]) - set(used_bins[label])

        normal_bins, heavy_bins = dataset.split_heavy_bins(
            available_bins, max_bin_segments, max_track_duration
        )

        print_bin_stats(label, normal_bins, heavy_bins, used_bins)

        add_random_samples(
            dataset,
            [validation, test],
            normal_bins,
            used_bins[label],
            label,
            required_samples,
            required_bins,
        )

        normal_bins.extend(heavy_bins)
        for bin_id in normal_bins:
            train.add_tracks(dataset.tracks_by_bin[bin_id])

    # if we have lots of segments on a single day, reduce the weight
    # so we don't overtrain on this specific example.
    train.balance_bins(max_bin_segments)
    validation.balance_bins(max_bin_segments)
    # balance out the classes
    train.balance_weights()
    validation.balance_weights()

    test.balance_resample(required_samples=build_config.test_set_count)

    print_segments(dataset, train, validation, test)

    return train, validation, test


def add_random_samples(
    dataset,
    fill_datasets,
    sample_set,
    used_bins,
    label,
    required_samples,
    required_bins,
):
    """
        add random samples from the sample_set to every dataset in 
        fill_datasets until the bin requirements are met
        Updates the bins in sample_set and used_bins
        """
    while sample_set and needs_more_bins(
        fill_datasets[0], label, used_bins, required_samples, required_bins
    ):

        bin_id = random.sample(sample_set, 1)[0]
        tracks = dataset.tracks_by_bin[bin_id]
        for ds in fill_datasets:
            ds.add_tracks(tracks)

        sample_set.remove(bin_id)
        used_bins.append(bin_id)


def needs_more_bins(dataset, label, used_bins, required_samples, required_bins):
    return (
        dataset.get_label_segments_count(label) < required_samples
        or len(used_bins) < required_bins
    )


def print_bin_segment_stats(bin_segment_mean, bin_segment_std, max_bin_segments):
    print()
    print(
        "Bin segment mean:{:.1f} std:{:.1f} auto max segments:{:.1f}".format(
            bin_segment_mean, bin_segment_std, max_bin_segments
        )
    )
    print()


def print_bin_stats(label, available_bins, heavy_bins, used_bins):
    print(
        "{}: normal {} heavy {} pre-filled {}".format(
            label, len(available_bins), len(heavy_bins), len(used_bins[label])
        )
    )


def print_segments(dataset, train, validation, test):

    print("Segments per class:")
    print("-" * 90)
    print("{:<20} {:<21} {:<21} {:<21}".format("Class", "Train", "Validation", "Test"))
    print("-" * 90)
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


def get_previous_validation_bins(filename):
    """Loads bin from previous database. """
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
    config = Config.load_from_file(args.config_file)
    return config


def print_data(dataset):
    for camera, data in enumerate(dataset.camera_bins):
        for loc, l_data in enumerate(data):
            for tag, t_data in enumerate(l_data):
                for date, d_data in enumerate(t_data):
                    print(
                        "{},{},{},{},{}".format(camera, loc, tag, data, len(d_data))
                    )


def main():
    init_logging()

    config = load_config()
    build_config = config.build
    db = TrackDatabase(os.path.join(config.tracks_folder, "dataset.hdf5"))
    dataset = Dataset(db, "dataset", config)
    tracks_loaded, total_tracks = dataset.load_tracks()
    print_data(dataset)
    print(
        "Loaded {}/{} tracks, found {:.1f}k segments".format(
            tracks_loaded, total_tracks, len(dataset.segments) / 1000
        )
    )
    for key, value in dataset.filtered_stats.items():
        if value != 0:
            print("  {} filtered {}".format(key, value))
    print()

    show_tracks_breakdown(dataset)
    print()
    show_segments_breakdown(dataset)
    print()
    show_cameras_breakdown(dataset)
    print()

    print("Splitting data set into train / validation")
    if build_config.use_previous_split:
        split = get_previous_validation_bins(build_config.previous_split)
        datasets = split_dataset(db, dataset, build_config, split)
    else:
        datasets = split_dataset(db, dataset, build_config)

    pickle.dump(datasets, open(dataset_db_path(config), "wb"))


if __name__ == "__main__":
    main()
