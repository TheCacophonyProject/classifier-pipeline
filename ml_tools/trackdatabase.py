"""

Author Matthew Aitchison

Date December 2017

Handles reading and writing tracks (or segments) to a large database.  Uses HDF5 as a backing store.

"""

import os
from multiprocessing import Lock
import h5py
import numpy as np

class TrackDatabase:

    def __init__(self, database_filename):
        """
        Initialises given database.  If database does not exist an empty one is created.
        :param database_filename: filename of database
        """

        self.database = database_filename

        if not os.path.exists(database_filename):
            print("Creating new database {}".format(database_filename))
            f = h5py.File(database_filename, 'w')
            f.create_group("clips")
            f.close()

    def has_clip(self, clip_id):
        """ 
        Returns if database contains track information for given clip
        :param clip_id: name of clip
        :return: If the database contains given clip
        """
        with hdf5_lock:
            f = h5py.File(self.database, 'r')
            clips = f['clips']
            has_record = clip_id in clips and 'finished' in clips[clip_id].attrs
            f.close()

        return has_record

    def create_clip(self, clip_id, tracker=None, overwrite=True):
        """
        Creates a blank clip entry in database.
        :param clip_id: id of the clip
        :param tracker: if provided stats from tracker are used for the clip stats
        :param overwrite: Overwrites existing clip (if it exists).
        """
        with hdf5_lock:
            f = h5py.File(self.database, 'a')
            clips = f['clips']
            if overwrite and clip_id in clips:
                del clips[clip_id]
            clip = clips.create_group(clip_id)

            if tracker is not None:
                stats = clip.attrs
                stats['filename'] = tracker.source_file
                stats['threshold'] = tracker.threshold
                stats['confidence'] = tracker.stats.get('confidence', 0)
                stats['trap'] = tracker.stats.get('trap', '') or ''
                stats['event'] = tracker.stats.get('event', '') or ''
                stats['average_background_delta'] = tracker.stats.get('average_background_delta',0)
                stats['mean_temp'] = tracker.stats.get('mean_temp', 0)
                stats['max_temp'] = tracker.stats.get('max_temp', 0)
                stats['min_temp'] = tracker.stats.get('min_temp', 0)

            f.flush()

            clip.attrs['finished'] = True
            f.close()

    def add_track(self, clip_id, track_id, track_data, track=None, opts=None):
        """
        Adds track to database.
        :param clip_id: id of the clip to add track to
        :param track_id: the tracks id
        :param track_data: data for track, numpy of shape [frames, channels, height, width, channels]
        :param track: the original track record, used to get stats for track
        :param opts: additional parameters used when creating dataset, if not provided defaults to lzf compression.
        """

        track_id = str(track_id)

        frames, channels, height, width = track_data.shape

        with hdf5_lock:
            f = h5py.File(self.database, 'a')
            clips = f['clips']
            track_entry = clips[clip_id]

            # using 9 frames (1 second) and seperating out the channels seems to work best.
            # Using a chunk size of 1 for channels has the advantage that we can quickly load just one channel
            chunks = (9, 1, height, width)

            dims = (frames, channels, height, width)

            # chunk the frames by channel
            if opts:
                dset = track_entry.create_dataset(track_id, dims, chunks=chunks,**opts, dtype=np.int16)
            else:
                dset = track_entry.create_dataset(track_id, dims, chunks=chunks, compression='lzf', shuffle=False, dtype=np.int16)

            dset[:,:,:,:] = track_data

            # write out attributes
            if track:
                track_stats = track.get_stats()

                stats = dset.attrs
                stats['id'] = track.id
                stats['tag'] = track.tag

                for name, value in track_stats._asdict().items():
                    stats[name] = value

                # frame history
                stats['mass_history'] = np.int32([bounds.mass for bounds in track.bounds_history])
                stats['bounds_history'] = np.int16([[bounds.left, bounds.top, bounds.right, bounds.bottom] for bounds in track.bounds_history])

            f.flush()

            # mark the record as have been writen to.
            # this means if we are interupted part way through the track will be overwritten
            track_entry.attrs['finished'] = True

            f.close()

# default lock for safe database writes.
# note for multiprocessing this will need to be overwritten with a shared lock for each process.
hdf5_lock = Lock()
