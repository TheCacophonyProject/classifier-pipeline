"""

Author Matthew Aitchison

Date December 2017

Handles reading and writing tracks (or segments) to a large database.  Uses HDF5 as a backing store.

"""

import os
from multiprocessing import Lock
import h5py
import tables           # required for blosc compression to work
import numpy as np

class HDF5Manager:
    """ Class to handle locking of HDF5 files. """
    def __init__(self, db, mode = 'r'):
        self.mode = mode
        self.f = None
        self.db = db

    def __enter__(self):
        # note: we might not have to lock when in read only mode?
        # this could improve performance
        hdf5_lock.acquire()
        self.f = h5py.File(self.db, self.mode)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()
        hdf5_lock.release()


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
        with HDF5Manager(self.database) as f:
            clips = f['clips']
            has_record = clip_id in clips and 'finished' in clips[clip_id].attrs

        return has_record

    def create_clip(self, clip_id, tracker=None, overwrite=True):
        """
        Creates a clip entry in database.
        :param clip_id: id of the clip
        :param tracker: if provided stats from tracker are used for the clip stats
        :param overwrite: Overwrites existing clip (if it exists).
        """
        with HDF5Manager(self.database, 'a') as f:
            clips = f['clips']
            if overwrite and clip_id in clips:
                del clips[clip_id]
            clip = clips.create_group(clip_id)

            if tracker is not None:
                stats = clip.attrs
                stats['filename'] = tracker.source_file
                stats['start_time'] = tracker.video_start_time.isoformat()
                stats['threshold'] = tracker.threshold
                stats['confidence'] = tracker.stats.get('confidence', 0.0) or 0.0
                stats['trap'] = tracker.stats.get('trap', '') or ''
                stats['event'] = tracker.stats.get('event', '') or ''
                stats['average_background_delta'] = tracker.stats.get('average_background_delta',0)
                stats['mean_temp'] = tracker.stats.get('mean_temp', 0)
                stats['max_temp'] = tracker.stats.get('max_temp', 0)
                stats['min_temp'] = tracker.stats.get('min_temp', 0)
                stats['frame_temp_min'] = tracker.frame_stats_min
                stats['frame_temp_max'] = tracker.frame_stats_max
                stats['frame_temp_median'] = tracker.frame_stats_median
                stats['frame_temp_mean'] = tracker.frame_stats_mean

            f.flush()
            clip.attrs['finished'] = True

    def get_all_track_ids(self):
        """
        Returns a list of clip_id, track_number pairs.
        """
        with HDF5Manager(self.database) as f:
            clips = f['clips']
            result = []
            for clip in clips:
                for track in clips[clip]:
                    result.append((clip, track))
        return result

    def get_track_meta(self, clip_id, track_number):
        """
        Gets metadata for given track
        :param clip_id:
        :param track_number:
        :return:
        """
        with HDF5Manager(self.database) as f:
            result = {}
            for key, value in f['clips'][str(clip_id)][str(track_number)].attrs.items():
                result[key] = value
            # track number (id) is broken in the file (always 1) so we just read it off the node number
            result['id'] = track_number
        return result

    def get_clip_meta(self, clip_id):
        """
        Gets metadata for given clip
        :param clip_id:
        :return:
        """
        with HDF5Manager(self.database) as f:
            result = {}
            for key, value in f['clips'][clip_id].attrs.items():
                result[key] = value
            result['tracks'] = len(f['clips'][clip_id])
        return result

    def get_track(self, clip_id, track_number, start_frame=None, end_frame=None):
        """
        Fetches a track data from database with optional slicing.
        :param clip_id: id of the clip
        :param track_number: id of the track
        :param start_frame: first frame of slice to return (inclusive).
        :param end_frame: last frame of slice to return (exclusive).
        :return: a numpy array of shape [frames, channels, height, width] and of type np.int16
        """
        with HDF5Manager(self.database) as f:
            clips = f['clips']
            dset = clips[clip_id][str(track_number)]
            return dset[start_frame:end_frame]

    def remove_clip(self, clip_id):
        """
        Deletes clip from database.
        Note, as per hdf5 the space will not be recovered.  If many files are deleted repacking the dataset might
        be a good idea.
        :param clip_id: id of clip to remove
        :returns: true if clip was deleted, false if it could not be found.
        """
        with HDF5Manager(self.database, 'a') as f:
            clips = f['clips']
            if clip_id in clips:
                del clips[clip_id]
                return True
            else:
                return False


    def add_track(self, clip_id, track_number, track_data, track=None, opts=None):
        """
        Adds track to database.
        :param clip_id: id of the clip to add track to write
        :param track_number: the tracks id
        :param track_data: data for track, numpy of shape [frames, channels, height, width, channels]
        :param track: the original track record, used to get stats for track
        :param opts: additional parameters used when creating dataset, if not provided defaults to lzf compression.
        """

        track_number = str(track_number)

        frames, channels, height, width = track_data.shape

        with HDF5Manager(self.database, 'a') as f:
            clips = f['clips']
            clip_node = clips[clip_id]

            # using 9 frames (1 second) and seperating out the channels seems to work best.
            # Using a chunk size of 1 for channels has the advantage that we can quickly load just one channel
            chunks = (9, 1, height, width)

            dims = (frames, channels, height, width)

            # chunk the frames by channel
            if opts:
                dset = clip_node.create_dataset(track_number, dims, chunks=chunks, **opts, dtype=np.int16)
            else:
                dset = clip_node.create_dataset(track_number, dims, chunks=chunks, compression='lzf', shuffle=False, dtype=np.int16)

            dset[:,:,:,:] = track_data

            # write out attributes
            if track:
                track_stats = track.get_stats()

                stats = dset.attrs
                stats['id'] = track.id
                stats['tag'] = track.tag
                stats['start_frame'] = track.start_frame
                stats['start_time'] = track.start_time.isoformat()
                stats['end_time'] = track.end_time.isoformat()

                for name, value in track_stats._asdict().items():
                    stats[name] = value

                # frame history
                stats['mass_history'] = np.int32([bounds.mass for bounds in track.bounds_history])
                stats['bounds_history'] = np.int16([[bounds.left, bounds.top, bounds.right, bounds.bottom] for bounds in track.bounds_history])

            f.flush()

            # mark the record as have been writen to.
            # this means if we are interupted part way through the track will be overwritten
            clip_node.attrs['finished'] = True


# default lock for safe database writes.
#
# note for multiprocessing this will need to be overwritten with a shared lock for each process.
# which can be done via
#
# def init_workers(lock):
#    trackdatabase.hdf5_lock = lock
#
# pool = multiprocessing.Pool(self.workers_threads, initializer=init_workers, initargs=(shared_lock,))

hdf5_lock = Lock()
