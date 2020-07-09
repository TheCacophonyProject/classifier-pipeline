"""

Author Matthew Aitchison

Date December 2017

Handles reading and writing tracks (or segments) to a large database.  Uses HDF5 as a backing store.

"""
import h5py
import os
import logging
import filelock
import datetime
from multiprocessing import Lock
import numpy as np
from classify.trackprediction import TrackPrediction

special_datasets = ["background_frame", "predictions"]


class HDF5Manager:
    """ Class to handle locking of HDF5 files. """

    LOCK_FILE = "/var/lock/classifier-hdf5.lock"

    def __init__(self, db, mode="r"):
        self.mode = mode
        self.f = None
        self.db = db
        self.lock = filelock.FileLock(HDF5Manager.LOCK_FILE, timeout=60 * 3)
        filelock.logger().setLevel(logging.ERROR)

    def __enter__(self):
        # note: we might not have to lock when in read only mode?
        # this could improve performance
        self.lock.acquire()
        self.f = h5py.File(self.db, self.mode)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.f.close()
        finally:
            self.lock.release()


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

    def get_labels(self):
        with HDF5Manager(self.database) as f:
            return f.attrs.get("labels", None)

    def set_labels(self, labels):
        with HDF5Manager(self.database, "a") as f:
            f.attrs["labels"] = labels

    def has_clip(self, clip_id):
        """
        Returns if database contains track information for given clip
        :param clip_id: name of clip
        :return: If the database contains given clip
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            has_record = clip_id in clips and "finished" in clips[clip_id].attrs
            if has_record:
                return True
        return False

    def has_prediction(self, clip_id):
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            has_record = clip_id in clips and "finished" in clips[clip_id].attrs
            clip = clips[clip_id]
            if has_record:
                return clip.attrs.get("has_prediction", False)
        return False

    def add_predictions(self, clip_id, model):
        with HDF5Manager(self.database, "a") as f:
            clip = f["clips"][str(clip_id)]
            for track_id in clip:
                if track_id in special_datasets:
                    continue

                track_node = clip[track_id]
                track_tag = track_node.attrs.get("tag", "")
                if track_tag not in model.labels:
                    if track_tag != "":
                        logging.info(
                            "Tag not in model labels %s", track_node.get("tag")
                        )

                result = []
                frames = list(track_node.keys())
                frames.sort(key=int)
                if len(frames) == 0:
                    return
                track_prediction = TrackPrediction(track_id, int(frames[0]), True)
                for frame_number in frames:
                    frame = track_node[frame_number][:, :, :]
                    frame = model.preprocess(frame)
                    prediction = model.classify_frame(frame)
                    track_prediction.classified_frame(frame_number, prediction)

                self._add_prediction_data(track_node, track_prediction, model)

            clip.attrs["has_prediction"] = True

    def _add_prediction_data(self, track, track_prediction, model):
        track_attrs = track.attrs
        track_tag = track_attrs.get("tag", "")
        if track_tag not in model.labels:
            if track_tag != "":
                logging.info("Tag not in model labels %s", track_tag)
            return
        label_index = model.labels.index(track_tag)
        track_attrs["correct_prediction"] = (
            track_attrs["tag"] == model.labels[track_prediction.best_label_index]
        )
        value, best = track_prediction.best_frame(label=label_index)
        track_attrs["best_frame"] = [round(100 * value, 0), best[0][0]]
        best = np.array(track_prediction.best_gap())
        best[1] = round(100 * best[1])
        track_attrs["best_gap"] = np.int16(best)
        preds = np.int16(np.around(100 * np.array(track_prediction.predictions)))
        # track_attrs["predictions"]=preds
        height, width = preds.shape
        pred_data = track.create_dataset(
            "predictions", (height, width), chunks=(height, width), dtype=preds.dtype,
        )
        pred_data[:, :] = preds

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
                if clip.background is not None:
                    height, width = clip.background.shape
                    background_frame = group.create_dataset(
                        "background_frame",
                        (height, width),
                        chunks=(height, width),
                        dtype=clip.background.dtype,
                    )
                    background_frame[:, :] = clip.background
                group_attrs = group.attrs

                # group_attrs.update(clip.stats)
                group_attrs["filename"] = clip.source_file
                group_attrs["start_time"] = clip.video_start_time.isoformat()
                group_attrs["threshold"] = clip.threshold

                if clip.res_x and clip.res_y:
                    group_attrs["res_x"] = clip.res_x
                    group_attrs["res_y"] = clip.res_y
                if clip.crop_rectangle:
                    group_attrs["edge_pixels"] = clip.crop_rectangle.left

                group_attrs["mean_background_value"] = clip.stats.mean_background_value
                group_attrs["threshold"] = clip.stats.threshold
                group_attrs["max_temp"] = clip.stats.max_temp
                group_attrs["min_temp"] = clip.stats.min_temp
                group_attrs["mean_temp"] = clip.stats.mean_temp
                group_attrs["filtered_deviation"] = clip.stats.filtered_deviation
                group_attrs["filtered_sum"] = clip.stats.filtered_sum
                group_attrs["temp_thresh"] = clip.stats.temp_thresh
                group_attrs["threshold"] = clip.stats.threshold

                if not clip.background_is_preview:
                    group_attrs["average_delta"] = clip.stats.average_delta
                    group_attrs["is_static"] = clip.stats.is_static_background
                group_attrs["frame_temp_min"] = clip.stats.frame_stats_min
                group_attrs["frame_temp_max"] = clip.stats.frame_stats_max
                group_attrs["frame_temp_median"] = clip.stats.frame_stats_median
                group_attrs["frame_temp_mean"] = clip.stats.frame_stats_mean

                if clip.device:
                    group_attrs["device"] = clip.device
                group_attrs["frames_per_second"] = clip.frames_per_second
                if clip.location and clip.location.get("coordinates") is not None:
                    group_attrs["location"] = clip.location["coordinates"]
            f.flush()
            group.attrs["finished"] = True

    def get_all_clip_ids(self):
        """
        Returns a list of clip_id, track_number pairs.
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            results = {}
            for clip_id in clips:
                results[clip_id] = [track_id for track_id in clips[clip_id]]
        return results

    def get_all_track_ids(self):
        """
        Returns a list of clip_id, track_number pairs.
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            result = []
            for clip in clips:
                for track in clips[clip]:
                    if track not in special_datasets:
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
            dataset = f["clips"][str(clip_id)]
            result = hdf5_attributes_dictionary(dataset)
            result["tracks"] = len(dataset)
        return result

    def get_tag(self, clip_id, track_number):
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            track_node = clips[str(clip_id)][str(track_number)]
            return track_node.attrs["tag"]

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
            track_node = clips[str(clip_id)][str(track_number)]
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

    def add_track(
        self,
        clip_id,
        track,
        track_data,
        opts=None,
        start_time=None,
        end_time=None,
        prediction=None,
        model=None,
    ):
        """
        Adds track to database.
        :param clip_id: id of the clip to add track to write
        :param track_data: data for track, list of numpy arrays of shape [channels, height, width]
        :param track: the original track record, used to get stats for track
        :param opts: additional parameters used when creating dataset, if not provided defaults to no compression.
        """

        track_id = str(track.get_id())

        frames = len(track_data)
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            clip_node = clips[clip_id]
            has_prediction = False
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
                node_attrs = track_node.attrs
                node_attrs["id"] = track_id
                if track.track_tags:
                    node_attrs["track_tags"] = [
                        track["what"] for track in track.track_tags
                    ]
                node_attrs["tag"] = track.tag
                node_attrs["frames"] = frames
                node_attrs["start_frame"] = track.start_frame
                node_attrs["end_frame"] = track.end_frame
                if prediction:
                    self._add_prediction_data(track_node, prediction, model)
                    has_prediction = True
                elif track.confidence:
                    node_attrs["confidence"] = track.confidence
                if start_time:
                    node_attrs["start_time"] = start_time.isoformat()
                if end_time:
                    node_attrs["end_time"] = end_time.isoformat()

                for name, value in track_stats._asdict().items():
                    node_attrs[name] = value
                # frame history
                node_attrs["mass_history"] = np.int32(
                    [bounds.mass for bounds in track.bounds_history]
                )
                node_attrs["bounds_history"] = np.int16(
                    [
                        [bounds.left, bounds.top, bounds.right, bounds.bottom]
                        for bounds in track.bounds_history
                    ]
                )

            f.flush()

            # mark the record as have been writen to.
            # this means if we are interupted part way through the track will be overwritten
            clip_node.attrs["finished"] = True
            clip_node.attrs["has_prediction"] = has_prediction


def hdf5_attributes_dictionary(dataset):
    result = {}
    for key, value in dataset.attrs.items():
        result[key] = value
    return result
