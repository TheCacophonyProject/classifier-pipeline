"""

Author Matthew Aitchison

Date December 2017

Handles reading and writing tracks (or segments) to a large database.  Uses HDF5 as a backing store.

"""

import os
from multiprocessing import Lock
import h5py
import numpy as np

# global lock to make sure two processes don't write to the file store at the same time.
hdf5_lock = Lock()

class TrackDatabase:

    def __init__(self, database_filename):
        """ Initialises given database.  If database does not exist an empty one is created. """

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
            has_record = clip_id in clips
            f.close()

        return has_record

    def create_clip(self, clip_id, overwrite=True):
        """
        Creates a blank clip entry in database.
        :param clip_id: id of the clip
        :param overwrite: Overwrites existing clip (if it exists).
        """
        with hdf5_lock:
            f = h5py.File(self.database, 'a')
            clips = f['clips']
            if overwrite and clip_id in clips:
                del clips[clip_id]
            clips.create_group(clip_id)
            f.close()

    def add_track(self, clip_id, track_id, track_data):
        """
        Adds track to database.
        :param clip_id: id of the clip to add track to
        :param track_id: the tracks id
        :param track_data: data for track, numpy of shape [frames, height, width, channels]
        """

        track_id = str(track_id)

        frames, height, width, channels = track_data.shape

        with hdf5_lock:
            f = h5py.File(self.database, 'a')
            clips = f['clips']
            track = clips[clip_id]

            # chunk the frames by channel
            dset = track.create_dataset(
                track_id,
                (frames, height, width, channels),
                chunks=(9, height, width, 1),
                compression='lzf', shuffle=True, dtype=np.int16
            )
            dset[:,:,:,:] = track_data
            f.close()


