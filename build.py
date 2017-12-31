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

def track_filter(clip_meta, track_meta):

    # some clips are banned for various reasons
    source = os.path.basename(clip_meta['filename'])
    if source in BANNED_CLIPS:
        return True

    # always let the false-positives through as we need them even though they would normally
    # be filtered out.
    if track_meta['tag'] == 'false-positive':
        return False

    # for some reason we get some records with a None confidence?
    if clip_meta.get('confidence', 0.0) < 0.5:
        return True

    # remove tracks of trapped animals
    if 'trap' in clip_meta.get('event', '').lower() or 'trap' in clip_meta.get('trap', '').lower():
        return True

    return False


def main():

    db = TrackDatabase('c:/cac/kea/dataset.hdf5')
    dataset = Dataset(db, 'train')

    total_tracks = len(db.get_all_track_ids())

    tracks_loaded = dataset.load_tracks(track_filter)

    print("Loaded {}/{} tracks, found {} segments".format(tracks_loaded, total_tracks, len(dataset.segments)))

    print(dataset.segments)


if __name__ == "__main__":
    main()