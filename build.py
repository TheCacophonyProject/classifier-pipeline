# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.

import argparse
import os
import random
import datetime
import logging
import pickle
import pytz

from dateutil.parser import parse as parse_date
from ml_tools.logs import init_logging
from config.config import Config
from ml_tools.dataset import Dataset
from ml_tools.datasetstructures import Camera
from ml_tools.recordwriter import create_tf_records

MIN_TRACKS = 1
use_clips = True


def load_config(config_file):
    return Config.load_from_file(config_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild-important", action="count", help="Rebuild important frames"
    )

    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument("-d", "--date", help="Use clips after this")
    parser.add_argument("--dont-cap", action="count", help="Dont cap numbers")
    parser.add_argument(
        "--consecutive-segments",
        action="count",
        default=False,
        help="Use consecutive frames for segments",
    )
    parser.add_argument(
        "--bb",
        "--balance-bins",
        action="count",
        help="Balance bins so each track has even percentage of being picked",
    )
    parser.add_argument(
        "--bl",
        "--balance-labels",
        action="count",
        help="Balance labels so that they have are distributed as defined in config",
    )

    args = parser.parse_args()
    if args.date:
        if args.date == "None":
            args.date = None
        else:
            args.date = parse_date(args.date)
    else:
        if args.date is None:
            args.date = datetime.datetime.now(pytz.utc) - datetime.timedelta(days=30)
            args.date = datetime.datetime.now() - datetime.timedelta(days=30)

    logging.info("Loading training set up to %s", args.date)
    return args


def show_tracks_breakdown(dataset):
    print("Tracks breakdown:")
    for label in dataset.labels:
        print("labels are", label)
        count = len([track for track in dataset.tracks_by_label.get(label, [])])
        print("  {:<20} {} tracks".format(label, count))


def show_cameras_tracks(dataset):
    for id, camera in dataset.cameras_by_id.items():
        count = "Tracks:"
        for label in dataset.labels:
            count = "{} {}: {}".format(count, label, camera.label_track_count(label))
        print("Camera", id)
        print(count)


def show_cameras_breakdown(dataset):
    print("Cameras breakdown")
    samples_by_camera = {}
    for sample in dataset.samples:
        if sample.camera not in samples_by_camera:
            samples_by_camera[sample.camera] = []
        samples_by_camera[sample.camera].append(sample)

    for camera, samples in samples_by_camera.items():
        print("{:<20} {}".format(camera, len(samples)))


def show_segments_breakdown(dataset):
    print("Segments breakdown:")
    for label in dataset.labels:
        count = sum(
            [len(track.segments) for track in dataset.tracks_by_label.get(label, [])]
        )
        print("  {:<20} {} segments".format(label, count))


def show_samples_breakdown(dataset):
    print("Samples breakdown:")
    for label in dataset.labels:
        count = len(dataset.samples_by_label.get(label, []))
        print("  {:<20} {} Samples".format(label, count))


def print_cameras(train, validation, test):

    print("Cameras per set:")
    print("-" * 90)
    print("Train")
    print(train.camera_names)
    print("Validation")
    print(validation.camera_names)
    print("Test")
    print(test.camera_names)

    print()


def print_counts(dataset, train, validation, test):
    print("Counts per class:")
    print("-" * 90)
    print("{:<20} {:<21} {:<21} {:<21}".format("Class", "Train", "Validation", "Test"))
    print("-" * 90)
    print("Samples / Tracks/ Bins/ weight")
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


def split_label(dataset, label, existing_test_count=0):
    # split a label from dataset such that vlaidation is 15% or MIN_TRACKS
    samples = dataset.samples_by_label.get(label, [])
    sample_bins = [sample.bin_id for sample in samples]
    if len(sample_bins) == 0:
        return None, None, None

    # sample_bins duplicates
    sample_bins = list(set(sample_bins))

    random.shuffle(sample_bins)
    train_c = Camera("{}-Train".format(label))
    validate_c = Camera("{}-Val".format(label))
    test_c = Camera("{}-Test".format(label))

    camera_type = "validate"
    add_to = validate_c
    last_index = 0
    label_count = 0
    total = len(sample_bins)
    min_t = MIN_TRACKS
    if label in ["vehicle", "human"]:
        min_t = 10
    num_validate_samples = max(total * 0.15, min_t)
    # num_test_tracks = max(total * 0.05, min_t) - existing_test_count
    # should have test covered by test set
    num_test_samples = 0
    for i, sample_bin in enumerate(sample_bins):
        samples = dataset.samples_by_bin[sample_bin]
        for sample in samples:
            if sample.label == label:
                label_count += 1

            sample.camera = "{}-{}".format(sample.camera, camera_type)
            add_to.add_sample(sample)
        dataset.samples_by_bin[sample_bin] = []
        last_index = i
        if label_count >= num_validate_samples:
            # 100 more for test
            if add_to == validate_c:
                add_to = test_c
                camera_type = "test"
                if num_test_samples <= 0:
                    break
                num_validate_samples += num_test_samples
            else:
                break

    sample_bins = sample_bins[last_index + 1 :]
    camera_type = "train"
    for i, sample_bin in enumerate(sample_bins):
        samples = dataset.samples_by_bin[sample_bin]
        for sample in samples:
            sample.camera = "{}-{}".format(sample.camera, camera_type)
            train_c.add_sample(sample)
        dataset.samples_by_bin[sample_bin] = []

    return train_c, validate_c, test_c


def get_test_set_camera(dataset, test_clips, after_date):
    # load test set camera from tst_clip ids and all clips after a date
    test_c = Camera("Test-Set-Camera")
    test_samples = [
        sample
        for sample in dataset.samples
        if sample.clip_id in test_clips
        or after_date is not None
        and sample.start_time > after_date
    ]
    for sample in test_samples:
        dataset.remove_sample(sample)
        test_c.add_sample(sample)
    return test_c


def split_randomly(db_file, dataset, config, args, test_clips=[], balance_bins=True):
    # split data randomly such that a clip is only in one dataset
    # have tried many ways to split i.e. location and cameras found this is simplest
    # and the results are the same
    train = Dataset(db_file, "train", config)
    train.enable_augmentation = True
    validation = Dataset(db_file, "validation", config)
    test = Dataset(db_file, "test", config)
    test_c = get_test_set_camera(dataset, test_clips, args.date)
    test_cameras = [test_c]
    validate_cameras = []
    train_cameras = []
    for label in dataset.labels:
        existing_test_count = len(test.samples_by_label.get(label, []))
        train_c, validate_c, test_c = split_label(
            dataset, label, existing_test_count=existing_test_count
        )
        if train_c is not None:
            train_cameras.append(train_c)
        if validate_c is not None:
            validate_cameras.append(validate_c)
        if test_c is not None:
            test_cameras.append(test_c)

    add_camera_samples(dataset.labels, train, train_cameras, balance_bins)
    add_camera_samples(dataset.labels, validation, validate_cameras, balance_bins)
    add_camera_samples(dataset.labels, test, test_cameras, balance_bins)
    return train, validation, test


def add_camera_samples(
    labels,
    dataset,
    cameras,
    balance_bins=None,
):
    # add camera tracks to the daaset and calculate segments and bins
    all_samples = []
    for label in labels:
        for camera in cameras:
            samples = camera.label_to_samples.get(label, {}).values()
            all_samples.extend(list(samples))
    dataset.add_samples(all_samples)
    dataset.recalculate_segments()
    dataset.balance_bins()


def main():
    init_logging()
    args = parse_args()
    config = load_config(args.config_file)
    # return
    # import yaml
    #
    # with open("defualtstest.yml", "w") as f:
    #     yaml.dump(config.as_dict(), f)
    test_clips = config.build.test_clips()
    if test_clips is None:
        test_clips = []
    logging.info("# of test clips are %s", len(test_clips))
    db_file = os.path.join(config.tracks_folder, "dataset.hdf5")
    dataset = Dataset(
        db_file, "dataset", config, consecutive_segments=args.consecutive_segments
    )

    tracks_loaded, total_tracks = dataset.load_clips()
    # return
    dataset.labels.sort()
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
    show_samples_breakdown(dataset)
    print()
    show_cameras_breakdown(dataset)
    print()
    print("Splitting data set into train / validation")
    datasets = split_randomly(db_file, dataset, config, args, test_clips)
    validate_datasets(datasets, test_clips, args.date)

    print_counts(dataset, *datasets)
    print("split data")

    base_dir = config.tracks_folder
    record_dir = os.path.join(base_dir, "training-data/")
    for dataset in datasets:
        dir = os.path.join(record_dir, dataset.name)
        create_tf_records(dataset, dir, num_shards=10)

        # dataset.saveto_numpy(os.path.join(base_dir))

    for dataset in datasets:
        dataset.clear_samples()
        dataset.db = None
        logging.info("saving to %s", f"{os.path.join(base_dir, dataset.name)}.dat")
        pickle.dump(dataset, open(f"{os.path.join(base_dir, dataset.name)}.dat", "wb"))


def validate_datasets(datasets, test_clips, date):
    # check that clips are only in one dataset
    # that only test set has clips after date
    # that test set is the only dataset with test_clips
    for dataset in datasets[:2]:
        for track in dataset.tracks:
            assert track.start_time < date

    for dataset in datasets:
        clips = set([track.clip_id for track in dataset.tracks])
        tracks = set([track.track_id for track in dataset.tracks])
        if test_clips is not None and dataset.name != "test":
            assert (
                len(clips.intersection(set(test_clips))) == 0
            ), "test clips should only be in test set"
        if len(clips) == 0:
            continue
        if len(tracks) == 0:
            continue
        for other in datasets:
            if dataset.name == other.name:
                continue
            other_clips = set([track.clip_id for track in other.tracks])
            other_tracks = set([track.track_id for track in other.tracks])
            assert clips != other_clips, "clips should only be in one set"
            assert tracks != other_tracks, "tracks should only be in one set"


if __name__ == "__main__":
    main()
