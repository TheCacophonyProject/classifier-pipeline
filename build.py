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
    # 10 cameras
    # 5 tests
    # 5 train
    # then change the names
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


def print_cameras(train, validation, test):

    print("Cameras per set:")
    print("-" * 90)
    print("Train")
    print(train.cameras)
    print("Validation")
    print(validation.cameras)
    print("Test")
    print(test.cameras)


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
                    print("{},{},{},{},{}".format(camera, loc, tag, data, len(d_data)))


def split_dataset_by_cameras(db, dataset, build_config):

    train_percent = 0.7
    validation_percent = 0.3
    test_cameras = 1
    # 10 cameras
    # 5 tests
    # 5 train
    # then change the names
    train = Dataset(db, "train")
    validation = Dataset(db, "validation")
    test = Dataset(db, "test")
    camera_count = len(dataset.camera_bins)
    assert camera_count >= 3, "Atleast 3 cameras are required"
    remaining_cameras = camera_count - test_cameras
    validation_cameras = max(1, round(remaining_cameras * validation_percent))
    remaining_cameras -= validation_cameras
    train_cameras = remaining_cameras

    # use hubble 6 for testing
    camera_data = dataset.camera_bins
    # want a test set that covers all labels

    cameras = list(camera_data.values())
    # randomize order
    cameras.sort(key=lambda x: np.random.random_sample())
    test_i = -1
    test_data = []
    for i, camera in enumerate(cameras):
        if len(camera.label_to_bins.keys()) == len(dataset.labels):
            test_data.append(camera)
            test_i = i
            break
    assert len(test_data) >= 0, "No test camera found with all labels"
    del cameras[test_i]

    train_data = cameras[:train_cameras]

    required_samples = build_config.test_set_count
    required_bins = max(MIN_BINS, build_config.test_set_bins)

    add_camera_data(
        dataset.labels,
        train,
        train_data,
        required_samples * 4,
        required_bins,
        build_config.cap_bin_weight,
        build_config.max_segments_per_track,
    )

    validate_data = cameras[train_cameras : train_cameras + validation_cameras]
    add_camera_data(
        dataset.labels,
        validation,
        validate_data,
        required_samples,
        required_bins,
        build_config.cap_bin_weight,
        build_config.max_segments_per_track,
    )

    # validation.add_cameras(validate_data)
    add_camera_data(
        dataset.labels,
        test,
        test_data,
        required_samples,
        required_bins,
        build_config.max_segments_per_track,
    )

    # balance out the classes
    train.balance_weights()
    validation.balance_weights()

    test.balance_resample(required_samples=build_config.test_set_count)

    print_segments(dataset, train, validation, test)
    print_cameras(train, validation, test)
    return train, validation, test


def add_random_camera_samples(
    dataset,
    camera_data,
    sample_set,
    label,
    required_samples,
    required_bins,
    segments_per_track,
):
    """
        add random samples from the sample_set to every dataset in 
        fill_datasets until the bin requirements are met
        Updates the bins in sample_set and used_bins
        """
    used_bins = []
    num_cameras = len(sample_set)
    cur_camera = 0
    # 1 from each camera
    while num_cameras > 0 and needs_more_bins(
        dataset, label, used_bins, required_samples, required_bins
    ):
        camera_i, cam_bins = sample_set[cur_camera]
        # camera_i = cam_bins[0]
        # cam_bins = cam_bins[1]

        bin_id = random.sample(cam_bins, 1)[0]
        tracks = camera_data[camera_i].bins[bin_id]
        dataset.add_tracks(tracks, segments_per_track)
        dataset.cameras.add(camera_data[camera_i].camera)

        cam_bins.remove(bin_id)
        used_bins.append(bin_id)

        if len(cam_bins) == 0:
            num_cameras -= 1
            del sample_set[cur_camera]
            if num_cameras > 0:
                cur_camera %= num_cameras
            continue

        cur_camera += 1
        cur_camera %= num_cameras


def add_camera_data(
    labels,
    dataset,
    cameras,
    required_samples,
    required_bins,
    cap_bin_weight=None,
    max_segments_per_track=None,
):

    counts = []

    # assign bins to test and validation sets
    # if we previously added bins from another dataset we are simply filling in the gaps here.
    for camera_data in cameras:
        counts.extend((camera_data.bin_segment_sum.values()))

    for label in labels:
        bins = []

        for i, camera_data in enumerate(cameras):

            if label not in camera_data.label_to_bins:
                continue
            bins.append((i, camera_data.label_to_bins[label]))

        add_random_camera_samples(
            dataset,
            cameras,
            bins,
            label,
            required_samples,
            required_bins,
            max_segments_per_track,
        )

    if cap_bin_weight is not None:
        bin_segment_mean = np.mean(counts)
        bin_segment_std = np.std(counts)
        max_bin_segments = bin_segment_mean + bin_segment_std * cap_bin_weight

        dataset.balance_bins(max_bin_segments)


def main():
    init_logging()

    config = load_config()
    build_config = config.build
    db = TrackDatabase(os.path.join(config.tracks_folder, "dataset.hdf5"))
    dataset = Dataset(db, "dataset", config)
    tracks_loaded, total_tracks = dataset.load_tracks()
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
    datasets = split_dataset_by_cameras(db, dataset, build_config)
    # if build_config.use_previous_split:
    #     split = get_previous_validation_bins(build_config.previous_split)
    #     datasets = split_dataset(db, dataset, build_config, split)
    # else:
    #     datasets = split_dataset(db, dataset, build_config)

    pickle.dump(datasets, open(dataset_db_path(config), "wb"))


if __name__ == "__main__":
    main()
