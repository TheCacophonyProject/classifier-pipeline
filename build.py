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

import numpy as np

from ml_tools.trackdatabase import TrackDatabase
from ml_tools.dataset import Dataset

BANNED_CLIPS = {
    '20171207-114424-akaroa09.cptv',
    '20171123-040215-akaroa09.cptv',
    '20171130-193036-brent01.cptv',
    '20171020-032802-Akaroa01.cptv',
    '20171019-101525-Akaroa01.cptv',
    '20171114-094045-akaroa04.cptv'
}

EXCLUDED_LABELS = ['mouse','insect','rabbit','cat','dog','human','stoat']

# if true removes any trapped animal footage from dataset.
# trapped footage can be a problem as there tends to be lots of it and the animals do not move in a normal way.
# however, bin weighting will generally take care of the excessive footage problem.
EXCLUDE_TRAPPED = True

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

# number of segments to include in test set for each class (multiplied by label weights)
TEST_SET_COUNT = 200



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


def split_dataset_bins():
    """
    Randomly selects tracks to be used as the train, validation, and test sets
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

    # assign bins to test and validation sets
    for label in dataset.labels:

        available_bins = set(bins_by_label[label])

        # heavy bins are bins with an unsually high number of examples on a day.  We exclude these from the test/validation
        # set as a single one could dominate the set.
        heavy_bins = set()
        for bin_id in available_bins:
            bin_segments = sum(len(track.segments) for track in dataset.tracks_by_bin[bin_id])
            if bin_segments > max_bin_segments:
                heavy_bins.add(bin_id)

        available_bins -= heavy_bins

        required_samples = TEST_SET_COUNT * LABEL_WEIGHTS.get(label, 1.0)

        # we assign bins to the test and validation sets randomly until we have 100 segments in each
        # the remaining bins can be used for training

        # todo: this is a problem we might only pick out 1 day, with a single track using this method.
        # better to either take a few days or just assign cameras.  Probably best to assign cameras I think.
        # this is more consistant with before, and gives us a better idea of true performance.
        # will stick to just a single evaluation set though as new data can be the 'test' set.

        while validation.get_class_segments_count(label) < required_samples and len(available_bins) > 0:
            sample = random.sample(available_bins, 1)[0]
            validation.add_tracks(dataset.tracks_by_bin[sample])
            validation.filter_segments(TEST_MIN_MASS, ['false-positive'])
            available_bins.remove(sample)

        while test.get_class_segments_count(label) < required_samples and len(available_bins) > 0:
            sample = random.sample(available_bins, 1)[0]
            test.add_tracks(dataset.tracks_by_bin[sample])
            test.filter_segments(TEST_MIN_MASS, ['false-positive'])
            available_bins.remove(sample)

        for bin_id in available_bins | heavy_bins :
            train.add_tracks(dataset.tracks_by_bin[bin_id])
        train.filter_segments(TEST_MIN_MASS, ['false-positive'])


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
    print()

    # normalisation constants
    normalisation_constants = train.get_normalisation_constants(1000)
    print('Normalisation constants:')
    for i in range(len(normalisation_constants)):
        print("  {:.4f} {:.4f}".format(normalisation_constants[i][0], normalisation_constants[i][1]))

    train.normalisation_constants = validation.normalisation_constants = test.normalisation_constants = normalisation_constants

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

    labels = sorted(list(set(dataset.tracks_by_label.keys())))
    dataset.labels = labels

    show_tracks_breakdown()
    print()
    show_segments_breakdown()
    print()

    print("Splitting data set into train / validation")
    datasets = split_dataset()

    pickle.dump(datasets,open('c:/cac/kea/datasets.dat','wb'))


if __name__ == "__main__":
    main()