"""
Author: Matthew Aitchison
Date: December 2017

Build a segment dataset for training.
Segment headers will be extracted from a track database and balanced according to class.
Some filtering occurs at this stage as well, for example tracks with low confidence are excluded.

"""

import os

from ml_tools.trackdatabase import TrackDatabase
from ml_tools.dataset import Dataset

BANNED_CLIPS = set('20171207-114424-akaroa09.cptv')

filtered_stats = {'confidence':0,'trap':0,'banned':0}

def track_filter(clip_meta, track_meta):

    # some clips are banned for various reasons
    source = os.path.basename(clip_meta['filename'])
    if source in BANNED_CLIPS:
        filtered_stats['banned'] += 1
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

def main():

    global dataset

    db = TrackDatabase('c:/cac/kea/dataset.hdf5')
    dataset = Dataset(db, 'train')

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





if __name__ == "__main__":
    main()