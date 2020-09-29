# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.

import argparse
import os
import pickle
import random
import numpy as np
from dateutil.parser import parse as parse_date

from ml_tools.logs import init_logging
from ml_tools.trackdatabase import TrackDatabase
from config.config import Config
from ml_tools.dataset import Dataset, dataset_db_path, Camera

LOW_DATA_LABELS = ["wallaby", "human", "dog"]
MIN_FRAMES = 1000
MIN_TRACKS = 100

CAP_DATA = True
MIN_VALIDATE_CAMERAS = 5
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
        "--bw",
        "--balance-weights",
        action="count",
        help="Balance weights for each label",
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


# idea is to try get a bit of all labels in validation set
# take 1 of the top 4 diverse CameraSegments
# then pick a missing label and try to find a camera
# until we either have reached max cameras, or there aren't any labels with less
# than MIN_FRAMES frames
def diverse_validation(cameras, labels, max_cameras):
    val_cameras = []

    # cameras = sorted(
    #     cameras, key=lambda camera: len(camera.label_to_bins.keys()), reverse=True
    # )
    # most_diverse_i = np.random.randint(0, 4)
    # most_diverse = cameras[most_diverse_i]
    # del cameras[most_diverse_i]
    # val_cameras.append(most_diverse)

    all_labels = labels.copy()
    # dont try add low data labels explicitly or we'll have none in train set
    for tag in LOW_DATA_LABELS:
        if tag in all_labels:
            all_labels.remove(tag)
    lbl_counts = {}
    for label in all_labels:
        lbl_counts[label] = 0
    missing_labels = all_labels.copy()
    # for label, count in most_diverse.label_frames.items():
    #     lbl_counts[label] = count
    #     if count >= MIN_FRAMES:
    #         missing_labels.remove(label)

    # randomize camera order so we dont always pick the same cameras
    np.random.shuffle(cameras)

    missing = len(missing_labels) / len(all_labels)
    missing_i = 0
    # always try find min label first
    label = min(lbl_counts.keys(), key=(lambda k: lbl_counts[k]))
    while len(val_cameras) <= max_cameras and missing != 0:
        found = False
        for i, camera in enumerate(cameras):
            if label in camera.label_to_bins:
                val_cameras.append(camera)

                # update validation counts
                # by segments or frames or both?
                # probably by # tracks
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


# only have one wallaby camera so just take MIN_FRAMES from wallaby and make a validation camera
def split_wallaby_cameras(dataset, cameras):
    if "Wallaby-None" not in dataset.cameras_by_id:
        return None, None
    wallaby = dataset.cameras_by_id["Wallaby-None"]
    cameras.remove(wallaby)
    print("wallaby bin", len(wallaby.bins))
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


def split_dataset_by_cameras(db, dataset, config, args, balance_bins=True):
    build_config = config.build
    validation_percent = 0.3
    train = Dataset(db, "train", config)
    train.enable_augmentation = True
    validation = Dataset(db, "validation", config)

    train_data = []
    cameras = list(dataset.cameras_by_id.values())
    camera_count = len(cameras)
    validation_cameras = max(
        MIN_VALIDATE_CAMERAS, round(camera_count * validation_percent)
    )

    wallaby, wallaby_validate = split_wallaby_cameras(dataset, cameras)
    if wallaby:
        train_data.append(wallaby)
    # has all the rabbits so put in training
    rabbits = dataset.cameras_by_id.get("ruru19w44a-[-36.03915 174.51675]")
    if rabbits:
        cameras.remove(rabbits)
        train_data.append(rabbits)
    validate_data, cameras = diverse_validation(
        cameras, dataset.labels, validation_cameras
    )
    if wallaby_validate:
        validate_data.append(wallaby_validate)
    train_data.extend(cameras)

    add_camera_segments(dataset.labels, train, train_data, balance_bins)
    add_camera_segments(dataset.labels, validation, validate_data, balance_bins)
    # balance out the classes

    if args.bl:
        train.balance_labels_and_remove(high_tracks_first=True)
        validation.balance_labels_and_remove()
        train.balance_labels_and_remove(high_tracks_first=True, use_segments=False)
        validation.balance_labels_and_remove(use_segments=False)
    # if we have lots of segments on a single day, reduce the weight
    # so we don't overtrain on this specific example.
    if args.bb:
        train.balance_bins()
        validation.balance_bins()
    # balance out the classes
    if args.bw:
        train.balance_weights()
        validation.balance_weights()
    return train, validation


def add_camera_segments(
    labels, dataset, cameras, balance_bins=None,
):
    all_tracks = []
    for label in labels:
        for camera in cameras:
            tracks = camera.label_to_tracks.get(label, {}).values()
            all_tracks.extend(list(tracks))

    dataset.add_tracks(all_tracks, None)
    dataset.balance_bins()


def add_random_camera_frames(
    dataset, cameras, label, max_frames,
):
    """
    add random samples from the sample_set to every dataset in
    fill_datasets until the bin requirements are met
    Updates the bins in sample_set and used_bins
    """
    used_bins = []
    num_cameras = len(cameras)
    cur_camera = 0
    print("max frames for ", label, max_frames)
    # 1 from each camera, until nothing left
    while num_cameras > 0 and (
        max_frames is None or dataset.samples_for(label) < max_frames
    ):
        camera = cameras[cur_camera]
        # bin_id = random.sample(cam_bins, 1)[0]
        # tracks = camera_data[camera_i].bins[bin_id]
        track, f = camera.sample_frame(label)
        if f is not None:
            dataset.add_track_header_frame(track, f)

        if camera.label_tracks(label) == 0:
            num_cameras -= 1
            del cameras[cur_camera]
            if num_cameras > 0:
                cur_camera %= num_cameras
            # print("removed a camera", num_cameras, cur_camera)
            continue

        cur_camera += 1
        cur_camera %= num_cameras
    if num_cameras == 0:
        print("use all data for {} dataset {}".format(label, dataset.name))


def add_camera_frames(
    labels,
    dataset,
    cameras,
    required_samples,
    required_bins,
    cap_bin_weight=None,
    max_segments_per_track=None,
    max_frames_per_track=None,
    dont_limit=["bird"],
):
    label_cap, label_data = get_distribution(labels, cameras, max_frames_per_track)

    for label, data in label_data.items():
        limit = data["max_frames"]
        if label in dont_limit or not CAP_DATA:
            limit = None
            print("dont limit", label)
        cameras = data["cameras"]
        add_random_camera_frames(
            dataset, cameras, label, limit,
        )


def test_dataset(db, config, date):
    test = Dataset(db, "test", config)
    tracks_loaded, total_tracks = test.load_tracks(shuffle=True, after_date=date)
    print("Test Loaded {}/{} tracks".format(tracks_loaded, total_tracks))
    for key, value in test.filtered_stats.items():
        if value != 0:
            print("Test  {} filtered {}".format(key, value))

    return test


def recalc_important(dataset_filename, db):
    datasets = pickle.load(open(dataset_filename, "rb"))
    print_counts(datasets[0], *datasets)

    for dataset in datasets:

        for track in dataset.tracks:
            if track.label == "false-positive":
                continue

            pre = len(track.important_frames)
            track_data = db.get_track(track.clip_id, track.track_id)
            track.important_frames = []
            if track_data is None:
                continue
            track.set_important_frames([], 16, False, frame_data=track_data)
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
        dataset.rebuild_cdf()
    print("after recalculating")
    print_counts(datasets[0], *datasets)

    return datasets


def main():
    init_logging()
    args = parse_args()
    config = load_config(args.config_file)
    build_config = config.build
    db = TrackDatabase(os.path.join(config.tracks_folder, "dataset.hdf5"))
    if args.rebuild_important:
        print("rebuild important")
        datasets = recalc_important(dataset_db_path(config), db)
        pickle.dump(datasets, open("important.dat", "wb"))
        return
    dataset = Dataset(db, "dataset", config)
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
    test = test_dataset(db, config, args.date)
    datasets = (*datasets, test)
    print_counts(dataset, *datasets)
    print_cameras(*datasets)
    # if build_config.use_previous_split:
    #     split = get_previous_validation_bins(build_config.previous_split)
    #     datasets = split_dataset(db, dataset, build_config, split)
    # else:
    #     datasets = split_dataset(db, dataset, build_config)
    pickle.dump(datasets, open(dataset_db_path(config), "wb"))


if __name__ == "__main__":
    main()
