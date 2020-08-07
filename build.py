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
from ml_tools.dataset import Dataset, dataset_db_path, Camera

LOW_DATA_LABELS = ["wallaby", "human", "dog"]
MIN_FRAMES = 1000
CAP_DATA = True
MIN_VALIDATE_CAMERAS = 1
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
        tracks = dataset.tracks_by_label[label]
        count = sum(
            [
                len(dataset.tracks_by_id[track_id].important_frames)
                for track_id in tracks
            ]
        )
        print("  {:<20} {} frames".format(label, count))


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


def split_dataset(db, dataset, build_config, args):
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

    if args.balance_labels:
        train.balance_labels()
        validation.balance_labels()

    # if we have lots of segments on a single day, reduce the weight
    # so we don't overtrain on this specific example.
    if args.balance_weights:
        train.balance_bins(max_bin_segments)
        validation.balance_bins(max_bin_segments)
    # balance out the classes
    if args.balance_weights:
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
    if required_bins is None and required_samples is None:
        return True

    needs_samples = (
        required_samples is None
        or dataset.get_label_segments_count(label) < required_samples
    )
    needs_bins = required_bins is None or len(used_bins) < required_bins
    if required_bins is None or required_samples is None:
        return needs_samples and needs_bins
    return needs_samples or needs_bins


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


def load_config(config_file):
    return Config.load_from_file(config_file)


def parse_args():
    parser = argparse.ArgumentParser()
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
        args.date = parse(args.date)
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
    cameras = sorted(
        cameras, key=lambda camera: len(camera.label_to_bins.keys()), reverse=True
    )
    most_diverse_i = np.random.randint(0, 4)
    most_diverse = cameras[most_diverse_i]
    del cameras[most_diverse_i]
    val_cameras.append(most_diverse)

    all_labels = labels.copy()
    for tag in LOW_DATA_LABELS:
        if tag in all_labels:
            all_labels.remove(tag)
    lbl_counts = {}
    for label in all_labels:
        lbl_counts[label] = 0
    missing_labels = all_labels.copy()
    for label, count in most_diverse.label_frames.items():
        lbl_counts[label] = count
        if count >= MIN_FRAMES:
            missing_labels.remove(label)

    # randomize camera order so we dont always pick the same cameras
    np.random.shuffle(cameras)

    missing = len(missing_labels) / len(all_labels)
    missing_i = 0

    # min label
    label = min(lbl_counts.keys(), key=(lambda k: lbl_counts[k]))
    while len(val_cameras) <= max_cameras and missing != 0:
        found = False
        for i, camera in enumerate(cameras):
            if label in camera.label_to_bins:
                val_cameras.append(camera)

                # update validation counts
                for label, count in camera.label_frames.items():
                    if label in lbl_counts:
                        lbl_counts[label] += count
                    else:
                        lbl_counts[label] = count

                    if label in missing_labels and lbl_counts[label] > MIN_FRAMES:
                        missing_labels.remove(label)
                print(lbl_counts)
                del cameras[i]
                missing = len(missing_labels) / len(all_labels)
                print("missing percent", missing)
                found = True
                break
        if not found:
            break
        if len(missing_labels) == 0:
            break
        # always add to min label
        label = min(lbl_counts.keys(), key=(lambda k: lbl_counts[k]))
    print(missing_labels, missing)
    print(lbl_counts)

    return val_cameras, cameras


def split_wallaby_cameras(dataset, cameras):
    if "Wallaby-None" not in dataset.cameras_by_id:
        return None, None
    wallaby = dataset.cameras_by_id["Wallaby-None"]
    cameras.remove(wallaby)

    wallaby_validate = Camera("Wallaby-2")
    remove = []
    last_index = 0
    wallaby_count = 0
    total = wallaby.label_frame_count("wallaby")
    wallaby_validate_segments = max(total * 0.2, MIN_FRAMES)
    print(
        "wallaby bins",
        len(wallaby.label_to_bins["wallaby"]),
        total,
        wallaby_validate_segments,
    )
    for i, bin_id in enumerate(wallaby.label_to_bins["wallaby"]):
        bin = wallaby.bins[bin_id]
        for track in bin:
            wallaby_count += len(track.segments)
            track.camera = "Wallaby-2"
            wallaby_validate.add_track(track)
            wallaby.segments -= 1
            wallaby.segment_sum -= len(track.segments)
        remove.append(bin_id)
        last_index = i
        if wallaby_count > wallaby_validate_segments:
            break
    wallaby.label_to_bins["wallaby"] = wallaby.label_to_bins["wallaby"][
        last_index + 1 :
    ]
    print("wallaby length is now", len(wallaby.label_to_bins["wallaby"]))
    for bin in remove:
        del wallaby.bins[bin]
    return wallaby, wallaby_validate


def split_dataset_by_cameras(db, dataset, build_config, args, balance_bins=True):
    validation_percent = 0.3
    train = Dataset(db, "train")
    validation = Dataset(db, "validation")
    test = Dataset(db, "test")
    train_data = []
    cameras = list(dataset.cameras_by_id.values())
    camera_count = len(cameras)
    validation_cameras = min(
        MIN_VALIDATE_CAMERAS, round(camera_count * validation_percent)
    )
    print("total cameras", camera_count)

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
        train.balance_labels(high_tracks_first=True)
        validation.balance_labels()

    # if we have lots of segments on a single day, reduce the weight
    # so we don't overtrain on this specific example.
    if args.bb:
        train.balance_bins()
        validation.balance_bins()
    # balance out the classes
    if args.bw:
        train.balance_weights()
        validation.balance_weights()

    print_segments(dataset, train, validation, test)
    print_cameras(train, validation, test)
    return train, validation, test


def add_camera_segments(
    labels, dataset, cameras, balance_bins=None,
):
    for label in labels:
        for camera in cameras:
            tracks = camera.label_to_tracks.get(label, {}).values()
            dataset.add_tracks(tracks, None)
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
    test = FrameDataset(db, "test", config, important_frames=False)
    tracks_loaded, total_tracks = test.load_tracks(shuffle=True, after_date=date)
    test.add_tracks()
    print("Test Loaded {}/{} tracks".format(tracks_loaded, total_tracks,))
    for key, value in test.filtered_stats.items():
        if value != 0:
            print("Test  {} filtered {}".format(key, value))

    return test


def main():
    init_logging()
    args = parse_args()
    config = load_config(args.config_file)
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
    datasets = split_dataset_by_cameras(db, dataset, build_config, args)
    # if build_config.use_previous_split:
    #     split = get_previous_validation_bins(build_config.previous_split)
    #     datasets = split_dataset(db, dataset, build_config, split)
    # else:
    #     datasets = split_dataset(db, dataset, build_config)

    pickle.dump(datasets, open(dataset_db_path(config), "wb"))


if __name__ == "__main__":
    main()
