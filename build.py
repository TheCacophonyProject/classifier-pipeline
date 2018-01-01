"""
Author: Matthew Aitchison
Date: December 2017

Build a segment dataset for training.
Segment headers will be extracted from a track database and balanced according to class.
Some filtering occurs at this stage as well, for example tracks with low confidence are excluded.

"""

import os
import random
import math

import numpy as np

from ml_tools.trackdatabase import TrackDatabase
from ml_tools.dataset import Dataset

BANNED_CLIPS = set('20171207-114424-akaroa09.cptv')

EXCLUDED_LABELS = ['mouse','insect']

# if true removes any trapped animal footage from dataset.
# trapped footage can be a problem as there tends to be lots of it and the animals do not move in a normal way.
# however, bin weighting will generally take care of the excessive footage problem.
REMOVE_TRAPPED = True

# sets a maxmimum number of segments per bin, where the cap is this many standard deviations above the norm.
# bins with more than this number of segments will be weighted lower so that their segments are not lost, but
# will be sampled less frequently.
CAP_BIN_WEIGHT = 2.0

# adjusts the weight for each animal class.  Setting this lower for animals that are less represented can help
# with training, otherwise the few examples we have will be used excessively.  This also helps build a prior for
# the class suggesting that the class is more or less likely.  For example bumping up the human weighting will cause
# the classifier learn towards guessing human when it is not sure.
LABEL_WEIGHTS = {
    'human':0.8,
    'cat':0.2,
    'dog':0.2,
}

# minimum average mass for test segment
TEST_MIN_MASS = 20


filtered_stats = {'confidence':0,'trap':0,'banned':0}

def track_filter(clip_meta, track_meta):

    # some clips are banned for various reasons
    source = os.path.basename(clip_meta['filename'])
    if source in BANNED_CLIPS:
        filtered_stats['banned'] += 1
        return True

    if track_meta['tag'] in EXCLUDED_LABELS:
        return True

    # always let the false-positives through as we need them even though they would normally
    # be filtered out.
    if track_meta['tag'] == 'false-positive':
        return False

    # for some reason we get some records with a None confidence?
    if clip_meta.get('confidence', 0.0) <= 0.6:
        filtered_stats['confidence'] += 1
        return True

    # remove tracks of trapped animals
    if 'trap' in clip_meta.get('event', '').lower() or 'trap' in clip_meta.get('trap', '').lower():
        filtered_stats['trap'] += 1
        return True


    return False


def show_tracks_breakdown():
    print("Tracks breakdown:")
    for label in dataset.labels:
        count = len([track for track in dataset.tracks_by_label[label]])
        print("  {:<20} {} tracks".format(label, count))


def show_segments_breakdown():
    print("Segments breakdown:")
    for label in dataset.labels:
        count = sum([len(track.segments) for track in dataset.tracks_by_label[label]])
        print("  {:<20} {} segments".format(label, count))


def split_dataset():
    """
    Randomly selects tracks to be used as the train, validation, and test sets
    :return: tuple containing train, validation, and test datasets.
    """

    # pick out groups to use for the various sets
    bins_by_label = {}
    for label in dataset.labels:
        bins_by_label[label] = []

    for bin, tracks in dataset.tracks_by_bin.items():
        label = dataset.track_by_name[tracks[0].name].label
        bins_by_label[label].append(tracks)

    train = Dataset(db, 'train')
    validation = Dataset(db, 'validation')
    test = Dataset(db, 'test')

    train.labels = dataset.labels
    validation.labels = dataset.labels
    test.labels = dataset.labels

    for label in dataset.labels:
        bins = bins_by_label[label]

        random.shuffle(bins)

        val_bins = math.ceil(len(bins) / 10)
        test_bins = math.ceil(len(bins) / 10)

        val_start = len(bins) - (val_bins + test_bins)
        test_start = len(bins) - (test_bins)

        for bin in bins[:val_start]:
            train.add_tracks(bin)
        for bin in bins[val_start:test_start]:
            validation.add_tracks(bin)
        for bin in bins[test_start:]:
            test.add_tracks(bin)

    # check bins distribution
    bin_segments = []
    for bin, tracks in dataset.tracks_by_bin.items():
        count = sum(len(track.segments) for track in tracks)
        bin_segments.append((count, bin))
    bin_segments.sort()
    for count, bin in bin_segments:
        print("{:<20} {} segments".format(bin,count))
    counts = [count for count, bin in bin_segments]
    bin_segment_mean = np.mean(counts)
    bin_segment_std = np.std(counts)

    print()
    print("Bin segment mean:{:.1f} std:{:.1f}".format(bin_segment_mean, bin_segment_std))
    print()

    print("Segments per class:")
    print("-"*90)
    print("{:<20} {:<21} {:<21} {:<21}".format("Class","Train","Validation","Test"))
    print("-"*90)

    # if we have lots of segments on a single day, reduce the weight so we don't overtrain on this specific
    # example.
    train.balance_bins(bin_segment_mean+bin_segment_std * CAP_BIN_WEIGHT)
    validation.balance_bins(bin_segment_mean + bin_segment_std * CAP_BIN_WEIGHT)

    test.filter_segments(TEST_MIN_MASS)

    # balance out the classes
    train.balance_weights(weight_modifiers=LABEL_WEIGHTS)
    validation.balance_weights(weight_modifiers=LABEL_WEIGHTS)
    test.balance_resample(weight_modifiers=LABEL_WEIGHTS)

    for label in dataset.labels:

        train_segments = sum(len(track.segments) for track in train.tracks_by_label.get(label,[]))
        validation_segments = sum(len(track.segments) for track in validation.tracks_by_label.get(label,[]))
        test_segments = sum(len(track.segments) for track in test.tracks_by_label.get(label,[]))

        train_weight = train.get_class_weight(label)
        validation_weight = validation.get_class_weight(label)
        test_weight = test.get_class_weight(label)

        print("{:<20} {:<10} {:<10.1f} {:<10} {:<10.1f} {:<10} {:<10.1f}".format(
            label,
            train_segments, train_weight,
            validation_segments,  validation_weight,
            test_segments, test_weight
            ))

    return train, validation, test

def main():

    global dataset
    global db

    db = TrackDatabase('c:/cac/kea/dataset.hdf5')
    dataset = Dataset(db, 'dataset')

    total_tracks = len(db.get_all_track_ids())

    tracks_loaded = dataset.load_tracks(track_filter)

    print("Loaded {}/{} tracks, found {:.1f}k segments".format(tracks_loaded, total_tracks, len(dataset.segments)/1000))
    for key, value in filtered_stats.items():
        if value != 0:
            print("  {} filtered {}".format(key, value))
    print()

    labels = set(dataset.tracks_by_label.keys())
    dataset.labels = labels

    show_tracks_breakdown()
    print()
    show_segments_breakdown()
    print()

    print("Splitting data set into train / validation")
    train, validation, test = split_dataset()



if __name__ == "__main__":
    main()