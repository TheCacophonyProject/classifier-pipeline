# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.

import argparse
import os
import pickle
import numpy as np
import random
import datetime
from dateutil.parser import parse as parse_date
import logging
from ml_tools.logs import init_logging
from ml_tools.trackdatabase import TrackDatabase
from config.config import Config
from ml_tools.dataset import Dataset, dataset_db_path
from ml_tools.datasetstructures import Camera
from track.track import TrackChannels
from ml_tools.kerasmodel import KerasModel

import pytz

LOW_DATA_LABELS = ["wallaby", "human", "dog", "vehicle"]
MIN_TRACKS = 100
MIN_SIZE = 4
CAP_DATA = True
MIN_VALIDATE_CAMERAS = 5


def show_tracks_breakdown(dataset):
    print("Tracks breakdown:")
    for label in dataset.labels:
        count = len([track for track in dataset.tracks_by_label[label]])
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


def show_important_frames_breakdown(dataset):
    print("important frames breakdown:")
    for label in dataset.labels:
        frame_count = len(dataset.frames_by_label[label])
        print("  {:<20} {} frames".format(label, frame_count))


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
    print("Segments / Frames / Tracks/ Bins/ weight")
    # display the dataset summary
    for label in dataset.labels:
        print(
            "{:<20} {:<20} {:<20} {:<20}".format(
                label,
                "{}/{}/{}/{}/{:.1f}".format(*train.get_counts(label)),
                "{}/{}/{}/{}/{:.1f}".format(*validation.get_counts(label)),
                "{}/{}/{}/{}/{:.1f}".format(*test.get_counts(label)),
            )
        )
    print()


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
        args.date = parse_date(args.date)
    return args


def print_data(dataset):
    for camera, data in enumerate(dataset.camera_bins):
        for loc, l_data in enumerate(data):
            for tag, t_data in enumerate(l_data):
                for date, d_data in enumerate(t_data):
                    print("{},{},{},{},{}".format(camera, loc, tag, data, len(d_data)))


def diverse_validation(cameras, labels, max_cameras):
    """
    idea is to try get a bit of all labels in validation set
    take 1 of the top 4 diverse CameraSegments
    then pick a missing label and try to find a camera
    until we either have reached max cameras, or there aren't any labels with less
    than MIN_TRACKS tracks
    """
    val_cameras = []

    all_labels = labels.copy()
    # dont try add low data labels or we'll have none in train set
    for tag in LOW_DATA_LABELS:
        if tag in all_labels:
            all_labels.remove(tag)
    lbl_counts = {}
    for label in all_labels:
        lbl_counts[label] = 0
    missing_labels = all_labels.copy()
    # randomize camera order so we dont always pick the same cameras
    np.random.shuffle(cameras)

    missing = len(missing_labels) / len(all_labels)

    # always try find min label first
    label = min(lbl_counts.keys(), key=(lambda k: lbl_counts[k]))
    while len(val_cameras) <= max_cameras and missing != 0:
        found = False
        for i, camera in enumerate(cameras):
            if label in camera.label_to_bins:
                val_cameras.append(camera)
                for label, tracks in camera.label_to_tracks.items():
                    count = len(tracks)
                    if label in lbl_counts:
                        lbl_counts[label] += count
                    else:
                        lbl_counts[label] = count

                    if label in missing_labels and lbl_counts[label] > MIN_TRACKS:
                        missing_labels.remove(label)
                del cameras[i]
                missing = len(missing_labels) / len(all_labels)
                found = True
                break
        if not found:
            break
        if len(missing_labels) == 0:
            break
        # always add to min label
        label = min(lbl_counts.keys(), key=(lambda k: lbl_counts[k]))

    return val_cameras, cameras


# only have one wallaby camera so just take MIN_TRACKS from wallaby and make a validation camera
def split_label(dataset, label, holdout_cameras, existing_test_count=0):
    tracks = dataset.tracks_by_label.get(label, [])
    track_bins = [
        track.bin_id
        for track in tracks
        if track.camera_id not in holdout_cameras and len(track.segments) > 0
    ]

    if len(track_bins) == 0:
        return None, None, None

    # remove duplicates
    track_bins = list(set(track_bins))

    random.shuffle(track_bins)
    train_c = Camera("{}-Train".format(label))
    validate_c = Camera("{}-Val".format(label))
    test_c = Camera("{}-Test".format(label))

    camera_type = "validate"
    add_to = validate_c
    last_index = 0
    label_count = 0
    total = len(tracks)
    min_t = MIN_TRACKS
    if label in ["vehicle", "human"]:
        min_t = 10
    num_validate_tracks = max(total * 0.15, min_t)
    num_test_tracks = max(total * 0.05, min_t) - existing_test_count
    cameras_to_remove = set()
    for i, track_bin in enumerate(track_bins):
        tracks = dataset.tracks_by_bin[track_bin]
        cameras_to_remove.add("{}-{}".format(tracks[0].camera, tracks[0].location))
        for track in tracks:
            if track.label == label:
                label_count += 1

            track.camera = "{}-{}".format(track.camera, camera_type)
            add_to.add_track(track)
        last_index = i
        if label_count >= num_validate_tracks:
            # 100 more for test
            if add_to == validate_c:
                add_to = test_c
                camera_type = "test"
                if num_test_tracks < 0:
                    break
                num_validate_tracks += num_test_tracks
            else:
                break

    track_bins = track_bins[last_index + 1 :]
    camera_type = "train"
    for i, track_bin in enumerate(track_bins):
        tracks = dataset.tracks_by_bin[track_bin]
        for track in tracks:
            track.camera = "{}-{}".format(track.camera, camera_type)
            train_c.add_track(track)
    for camera_name in cameras_to_remove:
        camera = dataset.cameras_by_id[camera_name]
        camera.remove_label(label)

    return train_c, validate_c, test_c


def get_test_set_camera(dataset, test_clips):
    test_c = Camera("Test-Set-Camera")

    test_tracks = [track for track in dataset.tracks if track.clip_id in test_clips]
    for track in test_tracks:
        dataset.remove_track(track)
        test_c.add_track(track)
    return test_c


def split_randomly(db, dataset, config, args, test_clips=None, balance_bins=True):
    # shuffle tracks and then add X% to validate, and 100 to test and rest to train
    # holdout_cameras = [
    #     "TrapCam01-None",
    #     "ospri11-None",
    #     "TrapCam03-[-43.65495 172.63125]",
    #     "TrapCam01-[-43.65495 172.63125]",
    #     "A_S4_C1-[-43.65315 172.63215]",
    #     "TrapCam01-[-43.65585 172.63125]",
    #     "TrapCam03-[-43.65585 172.63125]",
    #     "A_S4_C3-[-43.65405 172.62765]",
    #     "A_S4_C2-[-43.65405 172.62945]",
    #     "Wallaby2-[-44.76285 170.56f395]",
    # ]
    holdout_cameras = []
    test_cameras_only = []
    for camera_name in holdout_cameras:
        if camera_name in dataset.cameras_by_id:
            print("holding out", camera_name)
            test_cameras_only.append(dataset.cameras_by_id[camera_name])
            del dataset.cameras_by_id[camera_name]
    train = Dataset(db, "train", config)
    train.enable_augmentation = True
    validation = Dataset(db, "validation", config)
    test = Dataset(db, "test", config)
    test_cameras = test_cameras_only
    if test_clips is not None:
        test_c = get_test_set_camera(dataset, test_clips)
        add_camera_tracks(dataset.labels, test, [test_c], balance_bins)
        # test_cameras.append(test_c)
    validate_cameras = []
    train_cameras = []
    for label in dataset.labels:
        existing_test_count = len(test.tracks_by_label.get(label, []))
        train_c, validate_c, test_c = split_label(
            dataset, label, holdout_cameras, existing_test_count=existing_test_count
        )
        if train_c is not None:
            train_cameras.append(train_c)
        if validate_c is not None:
            validate_cameras.append(validate_c)
        if test_c is not None:
            test_cameras.append(test_c)

    add_camera_tracks(dataset.labels, train, train_cameras, balance_bins)
    add_camera_tracks(dataset.labels, validation, validate_cameras, balance_bins)
    add_camera_tracks(dataset.labels, test, test_cameras, balance_bins)
    return train, validation, test


def split_dataset_by_cameras(db, dataset, config, args, balance_bins=True):
    validation_percent = 0.3
    train = Dataset(db, "train", config, labels=dataset.labels)
    train.enable_augmentation = True
    validation = Dataset(db, "validation", config, labels=dataset.labels)
    test = Dataset(db, "test", config, labels=dataset.labels)

    # holdout_cameras = [
    #     "TrapCam01-None",
    #     "ospri11-None",
    #     "TrapCam03-[-43.65495 172.63125]",
    #     "TrapCam01-[-43.65495 172.63125]",
    #     "A_S4_C1-[-43.65315 172.63215]",
    #     "TrapCam01-[-43.65585 172.63125]",
    #     "TrapCam03-[-43.65585 172.63125]",
    #     "A_S4_C3-[-43.65405 172.62765]",
    #     "A_S4_C2-[-43.65405 172.62945]",
    #     "Wallaby2-[-44.76285 170.56f395]",
    # ]
    holdout_cameras = []
    test_cameras_only = []
    for camera_name in holdout_cameras:
        if camera_name in dataset.cameras_by_id:
            print("holding out", camera_name)
            test_cameras_only.append(dataset.cameras_by_id[camera_name])
            del dataset.cameras_by_id[camera_name]

    train_data = []
    validate_data = []
    test_data = test_cameras_only
    cameras = list(dataset.cameras_by_id.values())
    camera_count = len(cameras)
    validation_cameras = max(
        MIN_VALIDATE_CAMERAS, round(camera_count * validation_percent)
    )

    wallaby_train, wallaby_validate, wallaby_test = split_label(
        dataset, "wallaby", holdout_cameras
    )
    if wallaby_train is not None:
        train_data.append(wallaby_train)
    if wallaby_validate is not None:
        validate_data.append(wallaby_validate)
    if wallaby_test is not None:
        test_data.append(wallaby_test)

    vehicle_train, vehicle_validate, vehicle_test = split_label(
        dataset, "vehicle", holdout_cameras
    )
    if vehicle_train is not None:
        train_data.append(vehicle_train)
    if vehicle_validate is not None:
        validate_data.append(vehicle_validate)
    if vehicle_test is not None:
        test_data.append(vehicle_test)

    # has all the rabbits so put in training
    rabbits = dataset.cameras_by_id.get("ruru19w44a-[-36.03915 174.51675]")
    if rabbits:
        cameras.remove(rabbits)
        train_data.append(rabbits)
    diverse_validate, cameras = diverse_validation(
        cameras, dataset.labels, validation_cameras
    )
    validate_data.extend(diverse_validate)
    train_data.extend(cameras)

    add_camera_tracks(dataset.labels, train, train_data, balance_bins)
    add_camera_tracks(dataset.labels, validation, validate_data, balance_bins)
    add_camera_tracks(dataset.labels, test, test_data, balance_bins)

    return train, validation, test


def add_camera_tracks(
    labels,
    dataset,
    cameras,
    balance_bins=None,
):
    all_tracks = []
    for label in labels:
        for camera in cameras:
            tracks = camera.label_to_tracks.get(label, {}).values()
            all_tracks.extend(list(tracks))
    dataset.add_tracks(all_tracks, None)
    dataset.recalculate_segments(scale=1.0)
    dataset.balance_bins()


def test_dataset(db, test, config, args, date):
    dataset = Dataset(
        db, "dataset", config, consecutive_segments=args.consecutive_segments
    )
    tracks_loaded, total_tracks = dataset.load_tracks(after_date=args.date)

    print("Test Loaded {}/{} tracks".format(tracks_loaded, total_tracks))
    for key, value in test.filtered_stats.items():
        if value != 0:
            print("Test  {} filtered {}".format(key, value))
    test.add_tracks(dataset.tracks)
    return test


def recalc_important(dataset_filename, db, model_file=None):
    datasets = pickle.load(open(dataset_filename, "rb"))
    print_counts(datasets[0], *datasets)
    datasets[0].enable_augmentation = True
    model = KerasModel()
    model.load_model(model_file, training=False)
    for dataset in datasets:
        # if model:
        # dataset.frame_model = model
        dataset.frames_by_id = {}
        dataset.frame_samples = []
        dataset.frames_by_label = {}
        for track in dataset.tracks:
            if track.label == "false-positive":
                continue

            pre = len(track.important_frames)
            track_data = db.get_track(track.clip_id, track.track_id)
            track.important_frames = []
            if track_data is None:
                continue
            track.set_important_frames(20, track_data, model)
            print(
                "recalculated",
                track,
                "was",
                pre,
                "now",
                len(track.important_frames),
                track.label,
                dataset.name,
            )

        dataset.random_segments(require_movement=False)
        dataset.rebuild_cdf()
    print("after recalculating")
    print_counts(datasets[0], *datasets)
    dataset.frame_model = None
    return datasets


def add_overlay(dataset_filename, db):
    datasets = pickle.load(open(dataset_filename, "rb"))
    print_counts(datasets[0], *datasets)
    datasets[0].enable_augmentation = True

    for dataset in datasets:
        dataset.add_overlay()


def set_important(dataset, model_file):

    model = KerasModel()
    model.load_model(model_file, training=False)
    dataset.frame_model = model
    tracks_loaded, total_tracks = dataset.load_tracks()
    dataset.add_important()


def redo_important(dataset, db):
    # if model:
    # dataset.frame_model = model
    dataset.frames_by_id = {}
    dataset.frame_samples = []
    dataset.frames_by_label = {}
    for track in dataset.tracks:
        if track.label == "false-positive":
            continue

        pre = len(track.important_frames)
        track_data = db.get_track(
            track.clip_id, track.track_id, channels=TrackChannels.thermal
        )
        if track_data is None:
            continue
        keep_indices = []
        for i, frame in enumerate(track.important_frames):

            thermal = track_data[frame.frame_num].thermal
            height, width = thermal.shape
            if height < MIN_SIZE or width < MIN_SIZE:
                continue
            keep_indices.append(i)
        track.important_frames = np.array(track.important_frames)[keep_indices]
        dataset.db.set_important_frames(
            track.clip_id,
            track.track_id,
            [sample.frame_num for sample in track.important_frames],
        )
        print(
            "recalculated",
            track,
            "was",
            pre,
            "now",
            len(track.important_frames),
            track.label,
            dataset.name,
        )


def main():
    init_logging()
    args = parse_args()
    config = load_config(args.config_file)
    test_clips = config.build.test_clips()
    print("# of test clips are", len(test_clips))
    datasets_filename = dataset_db_path(config)
    db = TrackDatabase(os.path.join(config.tracks_folder, "dataset.hdf5"))
    dataset = Dataset(
        db, "dataset", config, consecutive_segments=args.consecutive_segments
    )
    tracks_loaded, total_tracks = dataset.load_tracks(before_date=args.date)
    print(
        "Loaded {}/{} tracks, found {:.1f}k segments".format(
            tracks_loaded, total_tracks, len(dataset.segments) / 1000
        )
    )
    for key, value in dataset.filtered_stats.items():
        if value != 0:
            print("  {} filtered {}".format(key, value))
    # print()
    # show_cameras_tracks(dataset)
    print()
    show_tracks_breakdown(dataset)
    print()
    show_segments_breakdown(dataset)
    print()
    show_important_frames_breakdown(dataset)
    print()
    show_cameras_breakdown(dataset)
    print()

    print("Splitting data set into train / validation")
    # datasets = split_dataset_by_cameras(db, dataset, config, args)
    datasets = split_randomly(db, dataset, config, args, test_clips)
    # if args.date is None:
    #     args.date = datetime.datetime.now(pytz.utc) - datetime.timedelta(days=7)
    test_dataset(db, datasets[2], config, args, args.date)
    validate_datasets(datasets, test_clips)
    print_counts(dataset, *datasets)
    print_cameras(*datasets)
    pickle.dump(datasets, open(dataset_db_path(config), "wb"))


def validate_datasets(datasets, test_clips):
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
