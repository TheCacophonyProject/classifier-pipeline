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
import pickle
import time

import numpy as np

from ml_tools.trackdatabase import TrackDatabase
from ml_tools.dataset import Dataset

DATASET_FOLDER = 'c:/cac/robin/'

# todo: move into a text file
BANNED_CLIPS = {
    '20171025-020827-akaroa03.cptv',
    '20171025-020827-akaroa03.cptv',
    '20171207-114424-akaroa09.cptv',
    '20171207-114424-akaroa09.cptv',
    '20171207-114424-akaroa09.cptv',
    '20171219-102910-akaroa12.cptv',
    '20171219-105919-akaroa12.cptv'
}

EXCLUDED_LABELS = ['mouse','insect','rabbit','cat','dog','human','stoat']

# if true removes any trapped animal footage from dataset.
# trapped footage can be a problem as there tends to be lots of it and the animals do not move in a normal way.
# however, bin weighting will generally take care of the excessive footage problem.
EXCLUDE_TRAPPED = True

# sets a maxmimum number of segments per bin, where the cap is this many standard deviations above the norm.
# bins with more than this number of segments will be weighted lower so that their segments are not lost, but
# will be sampled less frequently.
CAP_BIN_WEIGHT = 1.5

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
TEST_MIN_MASS = 40

# number of segments to include in test set for each class (multiplied by label weights)
TEST_SET_COUNT = 300

# minimum number of bins used for test set
TEST_SET_BINS = 10

filtered_stats = {'confidence':0,'trap':0,'banned':0}

def track_filter(clip_meta, track_meta):

    # some clips are banned for various reasons
    source = os.path.basename(clip_meta['filename'])
    if source in BANNED_CLIPS:
        filtered_stats['banned'] += 1
        return True

    if track_meta['tag'] in EXCLUDED_LABELS:
        return True

    # this camera gets a lot of obscured footage so we filter out to just the best ones.

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
    """
    Splits dataset into train / test / validation using a predefined set of camerea-label allocations.

    This method puts all tracks into 'camera-label' bins and then assigns certian bins to the validation set.
    In this case the 'test' set is simply a subsample of the validation set. (as there are not enough unique cameras
    at this point for 3 sets).

    The advantages of this method are

    1/ The datasets are fixed, even as more footage comes through
    2/ Learning an animal on one camera / enviroment and predicting it on another is a very good indicator that that
        algorthim has generalised
    3/ Some adjustment can be made to make sure that the cameras used in the test / validation sets contain 'resonable'
        footage. (i.e. footage that is close to what we want to classify.

    The disadvantages are that it requires seeing animals on multiple cameras, which we don't always have data for.
    It can also be a bit wasteful as we sometimes dedicate a substantial portion of the data to some animal types.

    A posiable solution would be to note when (or if) the model over-fits then run on the entire dataset.

    Another option would be k-fold validation.

    """

    validation_bins = [
        ('bird', 'akaroa03'),
        ('hedgehog', 'akaroa09'),
        ('hedgehog', 'akaroa03'),
        ('hedgehog', 'brent01'),
        ('possum', 'brent01'),
        ('possum', 'akaroa13'),
        ('cat', 'akaroa03'),
        ('cat', 'akaroa13'),
        ('cat', 'akaroa04'),
        ('cat', 'zip02'),
        ('stoat', 'akaroa09'),
        ('stoat', 'zip03'),
        ('human', 'akaroa09'),
        ('rat', 'akaroa04'),
        ('rat', 'akaroa10'),
        ('rat', 'zip01'),
        ('false-positive', 'akaroa09')
    ]

    raise Exception('not implemented yet.')


def split_dataset_days(test_bins=None):
    """
    Randomly selects tracks to be used as the train, validation, and test sets
    :param test_bins: if given will use these bins for the test set
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

    train = Dataset(db, 'train')
    validation = Dataset(db, 'validation')
    test = Dataset(db, 'test')

    train.labels = dataset.labels
    validation.labels = dataset.labels
    test.labels = dataset.labels

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
    print("Bin segment mean:{:.1f} std:{:.1f} auto max segments:{:.1f}".format(bin_segment_mean, bin_segment_std, max_bin_segments))
    print()

    if test_bins is None:

        # assign bins to test and validation sets
        test_bins = {}
        for label in dataset.labels:

            available_bins = set(bins_by_label[label])

            # heavy bins are bins with an unsually high number of examples on a day.  We exclude these from the test/validation
            # set as they will be subfiltered down and there is no need to waste that much data.
            heavy_bins = set()
            for bin_id in available_bins:
                bin_segments = sum(len(track.segments) for track in dataset.tracks_by_bin[bin_id])
                if bin_segments > max_bin_segments:
                    heavy_bins.add(bin_id)

            available_bins -= heavy_bins

            # print bin statistics
            print("{}: normal {} heavy {}".format(label, len(available_bins), len(heavy_bins)))

            required_samples = TEST_SET_COUNT * LABEL_WEIGHTS.get(label, 1.0)
            required_bins = TEST_SET_BINS * LABEL_WEIGHTS.get(label, 1.0) # make sure there is some diversity

            # we assign bins to the test and validation sets randomly until we have enough segments + bins
            # the remaining bins can be used for training
            used_bins = []
            while len(available_bins) > 0 and \
                    (validation.get_class_segments_count(label) < required_samples or len(used_bins) < required_bins):

                sample = random.sample(available_bins, 1)[0]

                validation.add_tracks(dataset.tracks_by_bin[sample])
                test.add_tracks(dataset.tracks_by_bin[sample])

                validation.filter_segments(TEST_MIN_MASS, ['false-positive'])
                test.filter_segments(TEST_MIN_MASS, ['false-positive'])

                available_bins.remove(sample)
                used_bins.append(sample)

            available_bins.update(heavy_bins)

            for bin_id in available_bins:
                train.add_tracks(dataset.tracks_by_bin[bin_id])

            test_bins[label] = used_bins

        # write out bins used, just in case we want to repeat this split
        pickle.dump(test_bins, open('dataset_split-{}.dat'.format(str(time.time())),'wb'))
    else:
        print("Using existing split:")
        for label in dataset.labels:
            available_bins = set(bins_by_label[label])
            for sample in test_bins[label]:

                # this happens if we have banned/deleted the clip, but it was previously used.
                if sample not in dataset.tracks_by_bin:
                    continue

                validation.add_tracks(dataset.tracks_by_bin[sample])
                test.add_tracks(dataset.tracks_by_bin[sample])
                validation.filter_segments(TEST_MIN_MASS, ['false-positive'])
                test.filter_segments(TEST_MIN_MASS, ['false-positive'])

                available_bins.remove(sample)

            for bin_id in available_bins:
                train.add_tracks(dataset.tracks_by_bin[bin_id])

    print("Segments per class:")
    print("-"*90)
    print("{:<20} {:<21} {:<21} {:<21}".format("Class","Train","Validation","Test"))
    print("-"*90)

    # if we have lots of segments on a single day, reduce the weight so we don't overtrain on this specific
    # example.
    train.balance_bins(max_bin_segments)
    validation.balance_bins(bin_segment_mean + bin_segment_std * CAP_BIN_WEIGHT)

    # balance out the classes
    train.balance_weights(weight_modifiers=LABEL_WEIGHTS)
    validation.balance_weights(weight_modifiers=LABEL_WEIGHTS)
    test.balance_resample(weight_modifiers=LABEL_WEIGHTS, required_samples = TEST_SET_COUNT)

    # display the dataset summary
    for label in dataset.labels:

        print("{:<20} {:<20} {:<20} {:<20}".format(
            label,
            "{}/{}/{}/{:.1f}".format(*train.get_counts(label)),
            "{}/{}/{}/{:.1f}".format(*validation.get_counts(label)),
            "{}/{}/{}/{:.1f}".format(*test.get_counts(label)),
            ))
    print()

    # normalisation constants
    normalisation_constants = train.get_normalisation_constants(5000)
    print('Normalisation constants:')
    for i in range(len(normalisation_constants)):
        print("  {:.4f} {:.4f}".format(normalisation_constants[i][0], normalisation_constants[i][1]))

    train.normalisation_constants = validation.normalisation_constants = test.normalisation_constants = normalisation_constants

    return train, validation, test

def main():

    global dataset
    global db

    db = TrackDatabase(os.path.join(DATASET_FOLDER,'dataset.hdf5'))
    dataset = Dataset(db, 'dataset')

    total_tracks = len(db.get_all_track_ids())

    tracks_loaded = dataset.load_tracks(track_filter)

    print("Loaded {}/{} tracks, found {:.1f}k segments".format(tracks_loaded, total_tracks, len(dataset.segments)/1000))
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
    #split = pickle.load(open('test_split.dat','rb'))
    datasets = split_dataset_days()

    pickle.dump(datasets,open(os.path.join(DATASET_FOLDER,'datasets.dat'),'wb'))


if __name__ == "__main__":
    main()