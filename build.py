# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.

import argparse
import os
import pickle
import numpy as np
import datetime
from dateutil.parser import parse as parse_date
import logging
from ml_tools.logs import init_logging
from ml_tools.trackdatabase import TrackDatabase
from config.config import Config
from ml_tools.dataset import Dataset, dataset_db_path
from ml_tools.datasetstructures import Camera
import pytz

LOW_DATA_LABELS = ["wallaby", "human", "dog"]
# minimum number of tracks for each label in validation dataset
MIN_TRACKS = 100
MIN_VALIDATE_CAMERAS = 5


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


def show_important_frames_breakdown(dataset):
    print("important frames breakdown:")
    for label in dataset.labels:
        frame_count = len(dataset.frames_by_label[label])
        print("  {:<20} {} frames".format(label, frame_count))


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
    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument("-d", "--date", help="Use clips after this")
    parser.add_argument(
        "--consecutive-segments",
        action="count",
        default=False,
        help="Use consecutive frames for segments",
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
    if len(labels) == 0:
        return [], cameras

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
    label = np.random.choice(missing_labels, 1)[0]
    while len(val_cameras) <= max_cameras and len(missing_labels) > 0:
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
                found = True
                break
        if not found:
            missing_labels.remove(label)
        # always add to min label
        if len(missing_labels) == 0:
            break
        label = min(missing_labels, key=(lambda k: lbl_counts[k]))

    return val_cameras, cameras


# only have one wallaby camera so just take MIN_TRACKS from wallaby and make a validation camera
def split_wallaby_cameras(dataset, cameras):
    if "Wallaby-None" not in dataset.cameras_by_id:
        return None, None
    wallaby = dataset.cameras_by_id["Wallaby-None"]
    cameras.remove(wallaby)
    wallaby_validate = Camera("Wallaby-2")
    remove = []
    last_index = 0
    wallaby_count = 0
    total = len(dataset.tracks_by_label.get("wallaby", []))
    wallaby_validate_tracks = max(total * 0.2, MIN_TRACKS)
    for i, bin_id in enumerate(wallaby.label_to_bins["wallaby"]):
        bin = wallaby.bins[bin_id]
        for track in bin:
            wallaby_count += 1
            track.camera = "Wallaby-2"
            wallaby_validate.add_track(track)
            wallaby.remove_track(track)
        remove.append(bin_id)
        last_index = i
        if wallaby_count > wallaby_validate_tracks:
            break
    wallaby.label_to_bins["wallaby"] = wallaby.label_to_bins["wallaby"][
        last_index + 1 :
    ]
    for bin in remove:
        del wallaby.bins[bin]
    return wallaby, wallaby_validate


def split_dataset_by_cameras(db, dataset, config, args):
    validation_percent = 0.3
    train = Dataset(db, "train", config)
    train.enable_augmentation = True
    validation = Dataset(db, "validation", config)

    train_cameras = []
    cameras = list(dataset.cameras_by_id.values())
    camera_count = len(cameras)
    num_validate_cameras = max(
        MIN_VALIDATE_CAMERAS, round(camera_count * validation_percent)
    )

    wallaby, wallaby_validate = split_wallaby_cameras(dataset, cameras)
    if wallaby:
        train_cameras.append(wallaby)
    # has all the rabbits so put in training
    rabbits = dataset.cameras_by_id.get("ruru19w44a-[-36.03915 174.51675]")
    if rabbits:
        cameras.remove(rabbits)
        train_cameras.append(rabbits)

    validate_cameras, cameras = diverse_validation(
        cameras, dataset.labels, num_validate_cameras
    )
    if wallaby_validate:
        validate_cameras.append(wallaby_validate)
    train_cameras.extend(cameras)

    add_camera_tracks(dataset.labels, train, train_cameras)
    add_camera_tracks(dataset.labels, validation, validate_cameras)

    return train, validation


def add_camera_tracks(
    labels,
    dataset,
    cameras,
):
    for label in labels:
        for camera in cameras:
            tracks = camera.label_to_tracks.get(label, {}).values()
            dataset.add_tracks(tracks, None)

    dataset.balance_bins()


def test_dataset(db, config, date):
    test = Dataset(db, "test", config)
    tracks_loaded, total_tracks = test.load_tracks(shuffle=True, after_date=date)
    print("Test Loaded {}/{} tracks".format(tracks_loaded, total_tracks))
    for key, value in test.filtered_stats.items():
        if value != 0:
            print("Test  {} filtered {}".format(key, value))

    return test


def main():
    init_logging()
    args = parse_args()
    config = load_config(args.config_file)
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
    datasets = split_dataset_by_cameras(db, dataset, config, args)
    if args.date is None:
        args.date = datetime.datetime.now(pytz.utc) - datetime.timedelta(days=7)
    test = test_dataset(db, config, args.date)
    datasets = (*datasets, test)
    print_counts(dataset, *datasets)
    print_cameras(*datasets)
    pickle.dump(datasets, open(dataset_db_path(config), "wb"))


if __name__ == "__main__":
    main()
