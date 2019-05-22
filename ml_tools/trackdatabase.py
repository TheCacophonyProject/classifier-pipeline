"""

Author Matthew Aitchison

Date December 2017

Handles reading and writing tracks (or segments) to a large database.  Uses HDF5 as a backing store.

"""

import os
import logging
from multiprocessing import Lock

import h5py
import tables  # required for blosc compression to work
import numpy as np

# default lock for safe database writes.
#
# note for multiprocessing this will need to be overwritten with a shared lock for each process.
# which can be done via
#
# def init_workers(lock):
#    trackdatabase.HDF5_LOCK = lock
#
# pool = multiprocessing.Pool(self.workers_threads, initializer=init_workers, initargs=(shared_lock,))

HDF5_LOCK = Lock()


class HDF5Manager:
    """ Class to handle locking of HDF5 files. """

    def __init__(self, db, mode="r"):
        self.mode = mode
        self.f = None
        self.db = db

    def __enter__(self):
        # note: we might not have to lock when in read only mode?
        # this could improve performance
        HDF5_LOCK.acquire()
        self.f = h5py.File(self.db, self.mode)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()
        HDF5_LOCK.release()


class TrackDatabase:
    def __init__(self, database_filename):
        """
        Initialises given database.  If database does not exist an empty one is created.
        :param database_filename: filename of database
        """

        self.database = database_filename

        if not os.path.exists(database_filename):
            logging.info("Creating new database %s", database_filename)
            f = h5py.File(database_filename, "w")
            f.create_group("clips")
            f.close()

    def has_clip(self, clip_id):
        """
        Returns if database contains track information for given clip
        :param clip_id: name of clip
        :return: If the database contains given clip
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            has_record = clip_id in clips and "finished" in clips[clip_id].attrs

        return has_record

    def create_clip(self, clip, overwrite=True):
        """
        Creates a clip entry in database.
        :param clip_id: id of the clip
        :param tracker: if provided stats from tracker are used for the clip stats
        :param overwrite: Overwrites existing clip (if it exists).
        """
        print("creating clip {}".format(clip.get_id()))
        clip_id = str(clip.get_id())
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            if overwrite and clip_id in clips:
                del clips[clip_id]
            group = clips.create_group(clip_id)

            if clip is not None:
                stats = group.attrs

                stats.update(clip.background_stats)
                stats["filename"] = clip.source_file
                stats["start_time"] = clip.video_start_time.isoformat()
                stats["threshold"] = clip.threshold
                stats["frame_temp_min"] = clip.frame_stats_min
                stats["frame_temp_max"] = clip.frame_stats_max
                stats["frame_temp_median"] = clip.frame_stats_median
                stats["frame_temp_mean"] = clip.frame_stats_mean
                stats["device"] = clip.device
                stats["frames_per_second"] = clip.frames_per_second
                if clip.location:
                    stats["location"] = clip.location
            f.flush()
            group.attrs["finished"] = True

    def get_all_track_ids(self):
        """
        Returns a list of clip_id, track_number pairs.
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
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
            dataset = f["clips"][clip_id][str(track_number)]
            result = hdf5_attributes_dictionary(dataset)
            result["id"] = track_number
        return result

    def get_clip_meta(self, clip_id):
        """
        Gets metadata for given clip
        :param clip_id:
        :return:
        """

        with HDF5Manager(self.database) as f:
            dataset = f["clips"][clip_id]
            result = hdf5_attributes_dictionary(dataset)
            result["tracks"] = len(dataset)
        return result

    def get_track(self, clip_id, track_number, start_frame=None, end_frame=None):
        """
        Fetches a track data from database with optional slicing.
        :param clip_id: id of the clip
        :param track_number: id of the track
        :param start_frame: first frame of slice to return (inclusive).
        :param end_frame: last frame of slice to return (exclusive).
        :return: a list of numpy arrays of shape [channels, height, width] and of type np.int16
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            track_node = clips[clip_id][str(track_number)]

            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = track_node["frames"]

            result = []

            for frame_number in range(start_frame, end_frame):
                # we use [:,:,:] to force loading of all data.
                result.append(track_node[str(frame_number)][:, :, :])

            return result

    def remove_clip(self, clip_id):
        """
        Deletes clip from database.
        Note, as per hdf5 the space will not be recovered.  If many files are deleted repacking the dataset might
        be a good idea.
        :param clip_id: id of clip to remove
        :returns: true if clip was deleted, false if it could not be found.
        """
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            if clip_id in clips:
                del clips[clip_id]
                return True
            else:
                return False

    def add_track(self, clip_id, track=None, opts=None, start_time=None, end_time=None):
        """
        Adds track to database.
        :param clip_id: id of the clip to add track to write
        :param track_data: data for track, list of numpy arrays of shape [channels, height, width]
        :param track: the original track record, used to get stats for track
        :param opts: additional parameters used when creating dataset, if not provided defaults to no compression.
        """

        track_id = str(track.get_id())

        track_data = track.track_data
        frames = len(track.track_data)
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            clip_node = clips[clip_id]

            track_node = clip_node.create_group(track_id)

            # write each frame out individually, as they will probably be different sizes.

            for frame_number in range(frames):

                channels, height, width = track_data[frame_number].shape

                # using a chunk size of 1 for channels has the advantage that we can quickly load just one channel
                chunks = (1, height, width)

                dims = (channels, height, width)

                if opts is not None:
                    frame_node = track_node.create_dataset(
                        str(frame_number), dims, chunks=chunks, **opts, dtype=np.int16
                    )
                else:
                    frame_node = track_node.create_dataset(
                        str(frame_number), dims, chunks=chunks, dtype=np.int16
                    )

                frame_node[:, :, :] = track_data[frame_number]

            # write out attributes
            if track:
                track_stats = track.get_stats()
                stats = track_node.attrs
                stats["id"] = track_id
                stats["tag"] = track.tag
                stats["frames"] = frames
                stats["start_frame"] = track.start_frame
                stats["end_frame"] = track.end_frame
                stats["confidence"] = track.confidence
                if start_time:
                    stats["start_time"] = start_time.isoformat()
                if end_time:
                    stats["end_time"] = end_time.isoformat()

                for name, value in track_stats._asdict().items():
                    stats[name] = value

                # frame history
                stats["mass_history"] = np.int32(
                    [bounds.mass for bounds in track.bounds_history]
                )
                stats["bounds_history"] = np.int16(
                    [
                        [bounds.left, bounds.top, bounds.right, bounds.bottom]
                        for bounds in track.bounds_history
                    ]
                )

            f.flush()

            # mark the record as have been writen to.
            # this means if we are interupted part way through the track will be overwritten
            clip_node.attrs["finished"] = True

def hdf5_attributes_dictionary(dataset):
    result = {}
    for key, value in dataset.attrs.items():
        result[key] = value
    return result