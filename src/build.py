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
import json
from dateutil.parser import parse as parse_date
from ml_tools.logs import init_logging
from config.config import Config
from ml_tools.dataset import Dataset
from ml_tools.datasetstructures import Camera
from ml_tools.tfwriter import create_tf_records
from ml_tools.irwriter import save_data as save_ir_data
from ml_tools.thermalwriter import save_data as save_thermal_data


import numpy as np

from pathlib import Path

MAX_TEST_TRACKS = None
MAX_TEST_SAMPLES = None

MIN_SAMPLES = 1
MIN_TRACKS = 1
# LOW_SAMPLES_LABELS = ["bird", "cat", "possum"]
LOW_SAMPLES_LABELS = []


def load_config(config_file):
    return Config.load_from_file(config_file)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--min-samples",
        default=None,
        type=int,
        help="Min tracks per dataset (Default 100)",
    )
    parser.add_argument(
        "-a",
        "--aug_percent",
        default=None,
        type=float,
        help="Percentage of training set to add extra augmentations of",
    )
    parser.add_argument("--split-file", help="Json file defining a split")
    parser.add_argument(
        "--ext", default=".hdf5", help="Extension of files to load .mp4,.cptv,.hdf5"
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
    parser.add_argument("data_dir", help="Directory of hdf5 files")
    args = parser.parse_args()
    if args.date:
        # if args.date == "None":
        #     args.date = None
        # else:
        args.date = parse_date(args.date)
        args.date = args.date.replace(tzinfo=pytz.utc)
    # else:
    #     if args.date is None:
    #         args.date = datetime.datetime.now(pytz.utc) - datetime.timedelta(days=30)

    if args.min_samples is not None:
        global MAX_TEST_TRACKS, MAX_TEST_SAMPLES, MIN_SAMPLES, MIN_TRACKS
        MAX_TEST_TRACKS = args.min_samples
        MAX_TEST_SAMPLES = args.min_samples

        MIN_SAMPLES = args.min_samples
        MIN_TRACKS = args.min_samples
        # args.date = datetime.datetime.now() - datetime.timedelta(days=30)
    if args.data_dir is not None:
        args.data_dir = Path(args.data_dir)
    logging.info("Loading training set up to %s", args.date)
    return args


def show_clips_breakdown(dataset):
    print("Clips breakdown:")

    for label in dataset.labels:
        print(label)
        lbl_samples = dataset.samples_by_label.get(label, [])
        if dataset.label_mapping:
            actual_labels = {}
            for s in lbl_samples:
                actual_labels.setdefault(s.original_label, set()).add(s.clip_id)
            for actual, clips in actual_labels.items():
                print("  {:<20} {} clips".format(actual, len(clips)))

        else:
            clips = [s.clip_id for s in lbl_samples]
            clips = set(clips)
            print("  {:<20} {} clips".format(label, len(clips)))


def show_tracks_breakdown(dataset):
    print("Tracks breakdown:")

    for label in dataset.labels:
        lbl_samples = dataset.samples_by_label.get(label, [])
        tracks = [s.unique_track_id for s in lbl_samples]
        tracks = set(tracks)
        print("  {:<20} {} tracks".format(label, len(tracks)))


def show_cameras_tracks(dataset):
    for id, camera in dataset.cameras_by_id.items():
        count = "Tracks:"
        for label in dataset.labels:
            count = "{} {}: {}".format(count, label, camera.label_track_count(label))
        print("Camera", id, count)


def show_cameras_breakdown(dataset):
    print("Cameras breakdown")
    samples_by_camera = {}
    for clip in dataset.clips:
        if clip.camera not in samples_by_camera:
            samples_by_camera[clip.camera] = []

        for track in clip.tracks:
            samples_by_camera[clip.camera].extend(track.samples)

    for camera, samples in samples_by_camera.items():
        print(
            "Camera: {:<20} {}".format(
                "None" if camera is None else camera, len(samples)
            )
        )


def show_bins_breakdown(dataset):
    print("Bins breakdown")
    samples_by_bins = {}
    for clip in dataset.clips:
        for t in clip.tracks:
            for sample in t.samples:
                if sample.bin_id not in samples_by_bins:
                    samples_by_bins[sample.bin_id] = []
                samples_by_bins[sample.bin_id].append(sample)

    for bin_id, samples in samples_by_bins.items():
        print(
            "Bin Id: {:<20} {}".format(
                "None" if bin_id is None else bin_id, len(samples)
            )
        )


def show_stations_breakdown(dataset):
    print("Stations breakdown")
    samples_by_station = {}
    for clip in dataset.clips:
        if clip.station_id not in samples_by_station:
            samples_by_station[clip.station_id] = []

        for track in clip.tracks:
            samples_by_station[clip.station_id].extend(track.samples)

    for station, samples in samples_by_station.items():
        print(
            "StationId: {:<20} {}".format(
                "None" if station is None else station, len(samples)
            )
        )


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


def print_counts(train, validation, test):
    logging.info("Counts per class:")
    logging.info("-" * 90)
    logging.info(
        "{:<20} {:<21} {:<21} {:<21}".format("Class", "Train", "Validation", "Test")
    )
    logging.info("-" * 90)
    logging.info("Samples / Tracks/ Bins/ weight")
    # display the dataset summary
    for label in train.labels:
        logging.info(
            "{:<20} {:<20} {:<20} {:<20}".format(
                label,
                "{}/{}/{}/{:.1f}".format(*train.get_counts(label)),
                "{}/{}/{}/{:.1f}".format(*validation.get_counts(label)),
                "{}/{}/{}/{:.1f}".format(*test.get_counts(label)),
            )
        )
    logging.info("")


# default split is by stationid, but some labels dont have many stations so best to just split by clip
split_by_clip = ["vehicle", "penguin", "wallaby"]


def split_label(
    dataset,
    label,
    counts,
    train_count,
    validation_count,
    test_count,
    existing_test_count=0,
    max_samples=None,
    use_test=True,
):
    # split a label from dataset such that vlaidation is 15% or MIN_TRACKS
    # dont split these by location and camera
    logging.info(
        "Splitting %s have counts (Tracks/Samples/Bins) %s already have (Tracks/Samples) in train %s validation %s and in test %s",
        label,
        counts,
        train_count,
        validation_count,
        test_count,
    )

    samples = dataset.samples_by_label.get(label, [])
    sample_bins = set([sample.bin_id for sample in samples])

    if counts[2] < 4 or counts[0] < 100:
        global split_by_clip
        split_by_clip.append(label)

    if label in split_by_clip:
        sample_bins = set([sample.clip_id for sample in samples])
        logging.info("%s Splitting by clip", label)
        samples_by_bin = {}
        for clip in dataset.clips:
            if clip.clip_id in sample_bins:
                samples = clip.get_samples()
                by_id = {}
                for s in samples:
                    by_id[s.id] = s
                samples_by_bin[clip.clip_id] = by_id
    else:
        samples_by_bin = dataset.samples_by_bin
    if len(sample_bins) == 0:
        return None, None, None

    if max_samples is not None:
        sample_bins = np.random.choice(
            sample_bins, min(len(sample_bins), max_samples), replace=False
        )

    sample_count = counts[1]
    total_tracks = counts[0]

    sample_bins = list(sample_bins)

    random.shuffle(sample_bins)
    train_c = []
    validate_c = []
    test_c = [] if use_test else None

    camera_type = "validate"
    add_to = validate_c
    last_index = 0
    label_count = 0
    min_t = MIN_SAMPLES

    if label in LOW_SAMPLES_LABELS:
        min_t = 10
    num_validate_samples = max(sample_count * 0.15, min_t) - validation_count[1]
    num_test_samples = max(sample_count * 0.05, min_t)
    if MAX_TEST_SAMPLES is not None:
        num_test_samples = min(MAX_TEST_SAMPLES, num_test_samples)
    num_test_samples -= test_count[1]

    min_t = MIN_TRACKS

    if label in LOW_SAMPLES_LABELS:
        min_t = 10

    num_validate_tracks = max(total_tracks * 0.15, min_t) - validation_count[0]
    num_test_tracks = max(total_tracks * 0.05, min_t)
    if MAX_TEST_TRACKS is not None:
        num_test_tracks = min(MAX_TEST_TRACKS, num_test_tracks)
    num_test_tracks -= test_count[0]

    track_limit = num_validate_tracks
    sample_limit = num_validate_samples
    tracks = set()
    logging.info(
        f"{label} - looking for val {num_validate_tracks} tracks out of {total_tracks} tracks and {num_validate_samples} samples from a total of {sample_count} samples  with {num_test_tracks} test tracks and {num_test_samples} test samples"
    )
    # if sample_limit < 0 and track_limit < 0:
    #     add_to = test_c
    #     camera_type = "test"
    #     sample_limit = num_test_samples
    #     track_limit = num_test_tracks
    #     label_count = 0
    #     tracks = set()
    if sample_limit > 0 and track_limit > 0:
        for i, sample_bin in enumerate(sample_bins):
            samples = samples_by_bin[sample_bin].values()
            for sample in samples:
                if sample.label == label:
                    tracks.add(sample.track_id)
                    label_count += 1

                # sample.camera = "{}-{}".format(sample.camera, camera_type)
                add_to.append(sample)

                if label in split_by_clip:
                    dataset.remove_sample(sample)
            if label not in split_by_clip:
                # while len(samples) > 0:
                sample_ids = [(s.id, s.bin_id) for s in samples]
                for id, bin_id in sample_ids:
                    dataset.remove_sample_by_id(id, bin_id)
            samples_by_bin[sample_bin] = {}
            last_index = i
            track_count = len(tracks)
            if label_count >= sample_limit and track_count >= track_limit:
                # 100 more for test
                if add_to == validate_c:
                    add_to = test_c
                    camera_type = "test"
                    if num_test_samples <= 0:
                        break
                    sample_limit = num_test_samples
                    track_limit = num_test_tracks
                    label_count = 0
                    tracks = set()
                    if use_test is False:
                        break
                else:
                    break

    # sample_bins = sample_bins[last_index + 1 :]
    # clearining anyway
    camera_type = "train"
    added = 0
    for i, sample_bin in enumerate(sample_bins):
        samples = samples_by_bin[sample_bin].values()
        for sample in samples:
            # sample.camera = "{}-{}".format(sample.camera, camera_type)
            train_c.append(sample)
            added += 1

            if label in split_by_clip:
                dataset.remove_sample(sample)
        if label not in split_by_clip:
            sample_ids = [(s.id, s.bin_id) for s in samples]
            for id, bin_id in sample_ids:
                dataset.remove_sample_by_id(id, bin_id)

        samples_by_bin[sample_bin] = []
    return train_c, validate_c, test_c


def get_test_set_camera(dataset, test_clips, after_date):
    # load test set camera from tst_clip ids and all clips after a date
    print("after date is", after_date)
    test_samples = []
    test_clips = [
        clip
        for clip in dataset.clips
        if clip.clip_id in test_clips
        or after_date is not None
        and clip.rec_time.replace(tzinfo=pytz.utc) > after_date
    ]
    for clip in test_clips:
        for track in clip.tracks:
            for sample in track.samples:
                dataset.remove_sample(sample)
                test_samples.append(sample)

    return test_samples


def split_by_file(dataset, config, split_file, base_dir, make_val=True):
    base_dir = Path(base_dir)
    with open(split_file, "r") as f:
        split = json.load(f)

    samples_by_source = {}
    for s in dataset.samples_by_id.values():
        samples_by_source.setdefault(s.source_file.name, []).append(s)
    datasets = []
    for name in ["train", "validation", "test"]:
        logging.info("Loading %s", name)
        split_dataset = Dataset(
            dataset.dataset_dir,
            name,
            config,
            label_mapping=dataset.label_mapping,
            ext=dataset.ext,
            raw=dataset.raw,
        )
        if name == "train":
            split_dataset.enable_augmentation = True
        elif name == "test":
            split_dataset.skip_ffc = False
        split_files = split.get(name, [])
        for f in split_files:
            # print("loading", base_dir / f["source"])
            # continue
            source_file = base_dir / f["source"]
            if source_file.exists():
                try:
                    # can filter crappy segments here, or can do at train stage to have some way of testing different thresholds
                    split_dataset.load_clip(source_file, dont_filter_segment=True)
                except:
                    logging.error("Could not load %s", source_file, exc_info=True)
            else:
                pass
                # logging.warn("No source file %s found for %s", f, name)
        datasets.append(split_dataset)

    print_counts(*datasets)
    if make_val:
        train, val, _ = split_randomly(datasets[0], config, None, use_test=False)
        datasets = [train, val, datasets[2]]
    return datasets


def split_randomly(dataset, config, date, test_clips=[], use_test=True):
    # split data randomly such that a clip is only in one dataset
    # have tried many ways to split i.e. location and cameras found this is simplest
    # and the results are the same
    train = Dataset(
        dataset.dataset_dir,
        "train",
        config,
        label_mapping=dataset.label_mapping,
        raw=dataset.raw,
        ext=dataset.ext,
    )
    train.enable_augmentation = True
    validation = Dataset(
        dataset.dataset_dir,
        "validation",
        config,
        label_mapping=dataset.label_mapping,
        raw=dataset.raw,
        ext=dataset.ext,
    )
    test = None
    test_counts = {}

    if use_test:
        test_c = get_test_set_camera(dataset, test_clips, date)
        test = Dataset(
            dataset.dataset_dir,
            "test",
            config,
            label_mapping=dataset.label_mapping,
            raw=dataset.raw,
            ext=dataset.ext,
        )
        add_samples(dataset.labels, test, test_c, test_counts)

    validate_cameras = []
    train_cameras = []
    min_label = None
    for label in dataset.labels:
        label_count = len(dataset.samples_by_label.get(label, []))
        if label not in ["insect", "false-positive"]:
            continue
        if min_label is None or label_count < min_label[1]:
            min_label = (label, label_count)
    lbl_order = sorted(
        dataset.labels,
        key=lambda lbl: len(dataset.samples_by_label.get(lbl, [])),
    )
    if "wallaby" in lbl_order:
        # make sure we do wallaby first so we get a good split
        lbl_order.remove("wallaby")
        lbl_order.insert(0, "wallaby")
    if "pest" in lbl_order:
        # make sure its last
        lbl_order.remove("pest")
        lbl_order.append("pest")

    lbl_counts = {}
    for lbl in dataset.labels:
        samples = dataset.samples_by_label.get(lbl, [])
        tracks = set([s.track_id for s in samples])
        bins = set([s.bin_id for s in samples])

        lbl_counts[lbl] = (len(tracks), len(samples), len(bins))

    logging.debug("lbl order is %s", lbl_order)
    train_counts = {}
    validation_counts = {}
    existing_test_count = 0
    for label in lbl_order:
        # existing_test_count = len(test.samples_by_label.get(label, []))
        train_c, validate_c, test_c = split_label(
            dataset,
            label,
            counts=lbl_counts[label],
            train_count=train_counts.get(label, (0, 0)),
            validation_count=validation_counts.get(label, (0, 0)),
            test_count=test_counts.get(label, (0, 0)),
            existing_test_count=existing_test_count,
            use_test=use_test,
            # max_samples=min_label[1],
        )
        if train_c is not None:
            add_samples(dataset.labels, train, train_c, train_counts)
        if validate_c is not None:
            add_samples(dataset.labels, validation, validate_c, validation_counts)
        if test_c is not None:
            add_samples(dataset.labels, test, test_c, test_counts)
        logging.debug("Train counts %s", train_counts)
        logging.debug("val counts %s", validation_counts)

    return train, validation, test


def add_samples(
    labels,
    dataset,
    samples,
    counts,
):
    by_labels = {}
    counts = {}
    for s in samples:
        if s.label not in by_labels:
            by_labels[s.label] = []
        by_labels[s.label].append(s)

    for label, lbl_samples in by_labels.items():
        track_count = len(set([s.track_id for s in lbl_samples]))
        counts[label] = (
            track_count,
            len(lbl_samples),
        )
    dataset.add_samples(samples)


def validate_datasets(datasets, test_bins, date):
    # check that clips are only in one dataset
    # that only test set has clips after date
    # that test set is the only dataset with test_clips

    # for dataset in datasets[:2]:
    #     for track in dataset.tracks:
    #         assert track.start_time < date

    for i, dataset in enumerate(datasets):
        dont_check = set(
            [
                sample.bin_id
                for sample in dataset.samples_by_id.values()
                if sample.label in split_by_clip
            ]
        )
        bins = set([sample.bin_id for sample in dataset.samples_by_id.values()])
        clips = set([sample.clip_id for sample in dataset.samples_by_id.values()])

        bins = bins - dont_check
        if test_bins is not None and dataset.name != "test":
            assert (
                len(bins.intersection(set(test_bins))) == 0
            ), "test bins should only be in test set"
        if len(bins) == 0:
            continue
        for other in datasets[(i + 1) :]:
            if dataset.name == other.name:
                continue
            dont_check = set(
                [
                    sample.bin_id
                    for sample in other.samples_by_id.values()
                    if sample.label in split_by_clip
                ]
            )
            other_bins = set([sample.bin_id for sample in other.samples_by_id.values()])
            other_bins = other_bins - dont_check
            other_clips = set(
                [sample.clip_id for sample in other.samples_by_id.values()]
            )

            intersection = bins.intersection(set(other_bins))

            assert (
                len(bins.intersection(set(other_bins))) == 0
            ), "bins should only be in one set"

            assert (
                len(clips.intersection(set(other_clips))) == 0
            ), "clips should only be in one set"


land_birds = [
    "pukeko",
    "california quail",
    "brown quail",
    "black swan",
    "quail",
    "pheasant",
    "penguin",
    "duck",
    "chicken",
    "rooster",
]

label_paths_dl = "https://raw.githubusercontent.com/TheCacophonyProject/cacophony-web/main/api/classifications/label_paths.json"


def dl_mappings():
    import requests

    logging.info("Downloading mappings file from %s ", label_paths_dl)
    response = requests.get(label_paths_dl)
    response.raise_for_status()

    mapping_content = response.content.decode()
    print(mapping_content)
    with open("label_paths.json", "w") as f:
        f.write(mapping_content)
    return mapping_content


def get_mappings():
    labels_path = Path("label_paths.json")
    if not labels_path.exists():
        label_paths = json.loads(dl_mappings())
    with open("label_paths.json", "r") as f:
        label_paths = json.load(f)
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


def dump_split_ids(datasets, out_file="datasplit.json"):
    splits = {}
    logging.info("Wrinting split ids to %s", out_file)
    for d in datasets:
        samples_by_source = d.get_samples_by_source()
        clips = []
        for source, samples in samples_by_source.items():
            tags = set([s.label for s in samples])
            clips.append(
                {
                    "clip_id": samples[0].clip_id,
                    "source": str(source),
                    "station_id": "{}".format(samples[0].station_id),
                    "tags": list(tags),
                }
            )
        # clips = set(clips)
        splits[d.name] = clips
    with open(out_file, "w") as f:
        json.dump(splits, f)
    return


def main():
    init_logging()
    args = parse_args()
    config = load_config(args.config_file)
    logging.info("Building for type %s", config.train.type)

    test_clips = config.build.test_clips()
    if test_clips is None:
        test_clips = []
    logging.info("# of test clips are %s", len(test_clips))
    label_mapping = get_mappings()
    logging.info("Using mappings %s", label_mapping)
    master_dataset = Dataset(
        args.data_dir,
        "dataset",
        config,
        consecutive_segments=args.consecutive_segments,
        label_mapping=label_mapping,
        raw=False if args.ext == ".hdf5" else True,
        ext=args.ext,
    )
    base_dir = Path(config.base_folder)
    record_dir = base_dir / "training-data"
    record_dir.mkdir(parents=True, exist_ok=True)

    if args.split_file:
        logging.info("Loading datasets from split file %s", args.split_file)
        datasets = split_by_file(master_dataset, config, args.split_file, args.data_dir)
        labels = set()
        for dataset in datasets:
            labels.update(dataset.labels)
        labels = list(labels)
        labels.sort()
        logging.info("Setting labels to %s", dataset.labels)
        for dataset in datasets:
            dataset.labels = labels
    else:
        tracks_loaded, total_tracks = master_dataset.load_clips(
            dont_filter_segment=True
        )

        master_dataset.labels.sort()
        print(
            "Loaded {}/{} tracks, found {:.1f}k samples".format(
                tracks_loaded, total_tracks, len(master_dataset.clips) / 1000
            )
        )
        for key, value in master_dataset.filtered_stats.items():
            if value != 0:
                print("  {} filtered {}".format(key, value))

        print()
        show_clips_breakdown(master_dataset)
        print()
        show_samples_breakdown(master_dataset)
        print()
        show_cameras_breakdown(master_dataset)
        print()
        show_stations_breakdown(master_dataset)
        print()
        show_bins_breakdown(master_dataset)
        print()
        print("Splitting data set into train / validation")

        datasets = split_randomly(master_dataset, config, args.date, test_clips)
        validate_datasets(datasets, test_clips, args.date)
        dump_split_ids(datasets, record_dir / "datasplit.json")

    print_counts(*datasets)
    print("split data")

    dataset_counts = {}
    # create_tf_records = create_thermal_records
    if config.train.type == "IR":
        threshold = (
            config.tracking[config.train.type]
            .motion.threshold_for_model(config.train.type)
            .background_thresh
        )
        # create_tf_records = create_ir_records
    else:
        threshold = None

    # create pre augmented records if samples are very low
    # not in use probably needs to be re tested if used
    aug_percent = args.aug_percent
    if aug_percent is not None:
        train_set = datasets[0]
        for l, samples in train_set.samples_by_label.items():
            track_dic = {}
            for s in samples:
                track_dic.setdefault(s.track_id, []).append(s)
            # track_dic = dict((x.track_id, x) for x in samples)

            for track, s in track_dic.items():
                augment_samples = int(aug_percent * len(s))
                # words = ['banana', 'pie', 'Washington', 'book']
                samples_by_mass = sorted(s, key=lambda s: s.region.mass, reverse=True)
                samples = np.random.choice(
                    samples_by_mass, augment_samples, replace=False
                )
                new_samples = []
                for s in samples:
                    new = s.copy()
                    new.augment = True
                    new_samples.append(new)
                train_set.add_samples(new_samples)
        print("Count post augmentation")
        print_counts(*datasets)

    for dataset in datasets:
        dir = os.path.join(record_dir, dataset.name)
        extra_args = {
            "use_segments": master_dataset.use_segments,
            "label_mapping": dataset.label_mapping,
        }
        if config.train.type == "IR":
            extra_args["back_thresh"] = threshold
            create_tf_records(
                dataset,
                dir,
                datasets[0].labels,
                save_ir_data,
                num_shards=100,
                back_thresh=threshold,
            )
        else:
            if args.ext != ".hdf5":
                extra_args.update(
                    {
                        "segment_frame_spacing": master_dataset.segment_spacing * 9,
                        "segment_width": master_dataset.segment_length,
                        "segment_type": master_dataset.segment_type,
                        "segment_min_avg_mass": master_dataset.segment_min_avg_mass,
                        "max_segments": master_dataset.max_segments,
                        "dont_filter_segment": True,
                        "skip_ffc": True,
                        "tag_precedence": config.load.tag_precedence,
                        "min_mass": master_dataset.min_frame_mass,
                    }
                )
            create_tf_records(
                dataset,
                dir,
                datasets[0].labels,
                save_thermal_data,
                num_shards=100,
                num_frames=dataset.segment_length,
                **extra_args,
            )
        counts = {}
        for label in dataset.labels:
            count = len(dataset.samples_by_label.get(label, []))
            counts[label] = count
        dataset_counts[dataset.name] = counts
    # dont need dataset anymore just need some meta
    meta_filename = f"{record_dir}/training-meta.json"
    meta_data = {
        "labels": datasets[0].labels,
        "type": config.train.type,
        "counts": dataset_counts,
        "by_label": False,
    }

    with open(meta_filename, "w") as f:
        json.dump(meta_data, f, indent=4)


if __name__ == "__main__":
    main()
