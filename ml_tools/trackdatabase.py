"""

Author Matthew Aitchison

Date December 2017

Handles reading and writing tracks (or segments) to a large database.  Uses HDF5 as a backing store.

"""
import h5py
import os
import logging
import filelock
import numpy as np
from dateutil.parser import parse as parse_date
from .frame import Frame, TrackChannels
import json
from dateutil.parser import parse as parse_date

import numpy as np
from track.region import Region

special_datasets = [
    "tag_frames",
    "frames",
    "background_frame",
    "predictions",
    "overlay",
]


class HDF5Manager:
    """Class to handle locking of HDF5 files."""

    LOCK_FILE = "/var/lock/classifier-hdf5.lock"
    READ_ONLY = False

    def __init__(self, db, mode="r"):
        self.mode = mode
        self.f = None
        self.db = db
        self.lock = filelock.FileLock(HDF5Manager.LOCK_FILE, timeout=60 * 3)
        filelock.logger().setLevel(logging.ERROR)

    def __enter__(self):
        # note: we might not have to lock when in read only mode?
        # this could improve performance
        if HDF5Manager.READ_ONLY and self.mode != "r":
            raise ValueError("Only read can be done in readonly mode")
        if not HDF5Manager.READ_ONLY:
            self.lock.acquire()
        self.f = h5py.File(self.db, self.mode)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.f.close()
        finally:
            if not HDF5Manager.READ_ONLY:
                self.lock.release()


class TrackDatabase:
    def __init__(self, database_filename, read_only=False):
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
        HDF5Manager.READ_ONLY = read_only

    def set_read_only(self, read_only):
        HDF5Manager.READ_ONLY = read_only

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
            # has_record = clip_id in clips and "finished" in clips[clip_id].attrs
            clip = clips[clip_id]
            # if has_record:
            return clip.attrs.get("has_prediction", False)
        return False

    #
    # def add_prediction_data(self, clip_id, track, predictions, labels=None):
    #     """
    #     Add prediction data as a dataset to the track
    #     data should be  an array of int16 array
    #     """
    #     logging.info("Add pre data %s", predictions)
    #     track_attrs = track.attrs
    #     prediction = predictions[0]
    #     predicted_tag = prediction.get("label")
    #     score = prediction.get("confidence")
    #     print("tag is", predicted_tag), prediction
    #     if predicted_tag is not None:
    #         track_attrs["correct_prediction"] = track_attrs.get("tag") == predicted_tag
    #         track_attrs["predicted"] = predicted_tag
    #         print("predicted", predicted_tag)
    #     track_attrs["predicted_confidence"] = int(round(100 * score))
    #
    #     pred_data = track.create_dataset(
    #         "predictions",
    #         predictions.shape,
    #         chunks=predictions.shape,
    #         dtype=predictions.dtype,
    #     )
    #     pred_data[:, :] = predictions
    #     if labels is not None:
    #         track_attrs["prediction_classes"] = labels

    def finished_processing(self, clip_id):
        with HDF5Manager(self.database, "a") as f:
            clip_node = f["clips"][clip_id]
            clip_node.attrs["finished"] = True

    def get_labels(self):
        with HDF5Manager(self.database) as f:
            return f.attrs.get("labels", None)

    def create_clip(self, clip, overwrite=True):
        """
        Creates a clip entry in database.
        :param clip_id: id of the clip
        :param tracker: if provided stats from tracker are used for the clip stats
        :param overwrite: Overwrites existing clip (if it exists).
        """
        logging.info("creating clip {}".format(clip.get_id()))
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
                group_attrs["background_thresh"] = clip.background_thresh

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

                group_attrs["frame_temp_min"] = clip.stats.frame_stats_min
                group_attrs["frame_temp_max"] = clip.stats.frame_stats_max
                group_attrs["frame_temp_median"] = clip.stats.frame_stats_median
                group_attrs["frame_temp_mean"] = clip.stats.frame_stats_mean

                if clip.device:
                    group_attrs["device"] = clip.device
                group_attrs["frames_per_second"] = clip.frames_per_second
                if clip.location and clip.location.get("coordinates") is not None:
                    group_attrs["location"] = clip.location["coordinates"]
                if clip.tags:
                    clip_tags = []
                    for track in clip.tags:
                        if track["what"]:
                            clip_tags.append(track["what"])
                        elif track["detail"]:
                            clip_tags.append(track["detail"])
                    group_attrs["tags"] = clip_tags
                group_attrs["ffc_frames"] = clip.ffc_frames

            f.flush()

    def latest_date(self):
        start_time = None

        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            results = {}
            for clip_id in clips:
                clip_start = clips[clip_id].attrs["start_time"]
                if clip_start:
                    if start_time is None or clip_start > start_time:
                        start_time = clip_start

        return start_time

    def get_all_clip_ids(self, before_date=None, after_date=None, label=None):
        """
        Returns a list of clip_id, track_number pairs.
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            results = {}
            for clip_id in clips:
                clip = clips[clip_id]
                date = parse_date(clip.attrs["start_time"])
                if before_date and date >= before_date:
                    continue
                    if after_date and date < after_date:
                        continue
                if label is not None:
                    if clip.attrs.get("tag") != label:
                        continue
                results[clip_id] = [track_id for track_id in clip]
        return results

    def get_clip_tracks_ids(self, clip_id):
        """
        Returns a list of clip_id, track_number pairs.
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            tracks = []
            clip = clips[clip_id]
            if not clip.attrs.get("finished"):
                return tracks
            for track in clip:
                if track not in special_datasets:
                    tracks.append(track)
        return tracks

    def get_all_track_ids(self, before_date=None, after_date=None, label=None):
        """
        Returns a list of clip_id, track_number pairs.
        """
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            result = []
            for clip_id in clips:
                clip = clips[clip_id]
                if not clip.attrs.get("finished"):
                    continue
                date = parse_date(clip.attrs["start_time"])
                if before_date and date >= before_date:
                    continue
                if after_date and date < after_date:
                    continue
                for track in clip:
                    if label is not None:
                        if label != clip[track].attrs.get("tag"):
                            continue
                    if track not in special_datasets:
                        result.append((clip_id, track))
        return result

    def remove_tag_info(self, clip_id, track_id):
        with HDF5Manager(self.database, "a") as f:
            clip = f["clips"][clip_id]
            try:
                del clip["tag_frames"]
            except:
                pass

            if "tag" in clip.attrs:
                del clip.attrs["tag"]

            track = clip[track_id]
            track.attrs["tag_confirmed"] = False

    def update_tag(self, clip_id, track_id, frames, clip_tag, track_tag, regions=None):
        logging.info(
            "Tagging clip %s as %s and track %s as %s",
            clip_id,
            clip_tag,
            track_id,
            track_tag,
        )
        with HDF5Manager(self.database, "a") as f:
            clip = f["clips"][clip_id]
            clip_tags = clip.get("tag_frames")
            if clip_tags is None:
                clip_tags = clip.create_group("tag_frames")
                print("creating clip tags")
            tag_regions = clip_tags.get("tag_regions")
            if tag_regions is None:
                tag_regions = clip_tags.create_group("tag_regions")
            print("clip tag", clip_tag)
            if clip_tag in clip_tags.attrs:
                clip_frames = list(clip_tags.attrs[clip_tag])
            else:
                clip_frames = set()

            clip_frames = set(clip_frames)
            add_regions = clip_tag == track_tag
            label_regions = []
            if add_regions:
                if clip_tag in tag_regions.attrs:
                    label_regions = list(tag_regions.attrs[clip_tag])
                    for i in range(len(label_regions)):
                        r = label_regions[i]
                        r[6] = 1 if r[6] else 0
                        r = np.uint16(r)
                        label_regions[i] = r
            for f in frames:
                if f.frame_number not in clip_frames:
                    clip_frames.add(f.frame_number)
                if add_regions:
                    if f.region.blank:
                        continue
                    r_array = f.region.to_array()

                    unique_region = True
                    for other in label_regions:
                        diff = False
                        for i in range(len(other)):
                            if other[i] != r_array[i]:
                                diff = True
                                break
                        if not diff:
                            unique_region = False
                            break
                    if unique_region:
                        label_regions.append(r_array)

            if add_regions:
                tag_regions.attrs[clip_tag] = label_regions
                # for k, v in tag_regions.attrs.items():
                #     print("K", k, v)

            clip_frames = list(clip_frames)
            clip_frames.sort()
            clip_tags.attrs[clip_tag] = np.array(clip_frames)

            # clip.attrs["tag_frames"] = clip_tags
            clip.attrs["tag"] = clip_tag
            track = clip[track_id]
            if track_tag is not None:
                track.attrs["tag"] = track_tag
                track.attrs["tag_confirmed"] = True

    def get_track_meta(self, clip_id, track_id):
        """
        Gets metadata for given track
        :param clip_id:
        :param track_number:
        :return:
        """
        with HDF5Manager(self.database) as f:
            dataset = f["clips"][str(clip_id)][str(track_id)]
            result = self.dataset_track(dataset, track_id)

        return result

    def dataset_track(self, dataset, track_id):
        result = hdf5_attributes_dictionary(dataset)
        preds = dataset.get("model_predictions")
        if preds is not None:
            result["model_predictions"] = {}
            for model in preds:
                model_preds = hdf5_attributes_dictionary(preds[model])
                result["model_predictions"][model] = model_preds
        result["id"] = track_id

        return result

    def get_track_predictions(self, clip_id, track_number):
        """
        Gets metadata for given track
        :param clip_id:
        :param track_number:
        :return:
        """
        with HDF5Manager(self.database) as f:
            track = f["clips"][clip_id][str(track_number)]
            if "predictions" in track:
                return track["predictions"][:]
        return None

    def get_clip_background(self, clip_id):
        with HDF5Manager(self.database) as f:
            clip = f["clips"][str(clip_id)]
            if "background_frame" in clip:
                return clip["background_frame"][:]
        return None

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
            tag_frames = dataset.get("tag_frames")
            if tag_frames:
                result["tag_frames"] = {}
                for key, value in tag_frames.attrs.items():
                    result["tag_frames"][key] = value

                tag_regions = tag_frames.get("tag_regions")
                if tag_regions is not None:
                    result["tag_frames"]["tag_regions"] = {}
                    for key, value in tag_regions.attrs.items():
                        result["tag_frames"]["tag_regions"][key] = value
        return result

    def get_clip_tracks(self, clip_id):
        """
        Gets metadata for given clip
        :param clip_id:
        :return:
        """
        tracks = []
        with HDF5Manager(self.database) as f:
            clip = f["clips"][str(clip_id)]
            for track_id in clip:
                if track_id in special_datasets:
                    continue
                track = self.dataset_track(clip[track_id], track_id)
                tracks.append(track)
        return tracks

    def get_tag(self, clip_id, track_number):
        with HDF5Manager(self.database) as f:
            clips = f["clips"]
            track_node = clips[str(clip_id)][str(track_number)]
            return track_node.attrs["tag"]

    def get_frame(self, clip_id, track_id, frame, original=False):
        frames = self.get_track(
            clip_id, track_id, frame_numbers=[frame], original=original
        )
        if len(frames) == 1:
            return frames[0]
        return None

    def get_clip(
        self,
        clip_id,
        frame_numbers=None,
        channels=None,
    ):
        frames = []
        with HDF5Manager(self.database) as f:
            clip = f["clips"][str(clip_id)]
            if "frames" not in clip:
                return None
            frames_node = clip["frames"]
            if frame_numbers is None:
                frame_numbers = []
                for f_i in frames_node:
                    frame_numbers.append(int(f_i))
                frame_numbers.sort()
                print("using frames", frame_numbers)
            frame_iter = iter(frame_numbers)

            for frame_number in frame_iter:

                frame = frames_node[str(frame_number)][:, :]
                frames.append(
                    Frame.from_channels([frame], [TrackChannels.thermal], frame_number)
                )
        return frames

    def get_track(
        self,
        clip_id,
        track_number,
        start_frame=None,
        end_frame=None,
        original=False,
        frame_numbers=None,
        channels=None,
    ):
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
            clip_node = clips[str(clip_id)]
            track_node = clip_node[str(track_number)]

            bounds = track_node.attrs["bounds_history"]
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = track_node.attrs["frames"]
            track_start = track_node.attrs.get("start_frame")
            bad_frames = track_node.attrs.get("skipepd_frames", [])
            result = []
            if original:
                track_node = clip_node["frames"]
            else:
                if "cropped" in track_node:
                    track_node = track_node["cropped"]

            if frame_numbers is None:
                frame_iter = range(start_frame, end_frame)
            else:
                frame_iter = iter(frame_numbers)

            for frame_number in frame_iter:

                if original:

                    region = Region.region_from_array(bounds[frame_number])
                    region.frame_number = frame_number + track_start
                    frame = track_node[str(frame_number + track_start)][:, :]
                    result.append(
                        Frame.from_channels(
                            [frame],
                            [TrackChannels.thermal],
                            frame_number + track_start,
                            region=region,
                        )
                    )
                else:
                    if frame_number in bad_frames:
                        continue
                    region = Region.region_from_array(bounds[frame_number])
                    if channels is None:
                        try:
                            frame = track_node[str(frame_number)][:, :, :]
                            if frame.shape[0] == 3:
                                f = Frame.from_channels(
                                    frame,
                                    [
                                        TrackChannels.thermal,
                                        TrackChannels.filtered,
                                        TrackChannels.mask,
                                    ],
                                    frame_number + track_start,
                                    region=region,
                                )
                            else:
                                f = Frame.from_array(
                                    frame,
                                    frame_number + track_start,
                                    flow_clipped=True,
                                    region=region,
                                )
                            result.append(f)
                        except:
                            logging.debug(
                                "trying to get clip %s track %s frame %s",
                                clip_id,
                                track_number,
                                frame_number + track_start,
                                exc_info=True,
                            )
                    else:
                        try:
                            frame = track_node[str(frame_number)][channels, :, :]
                            result.append(
                                Frame.from_channels(
                                    frame,
                                    channels,
                                    frame_number + track_start,
                                    region=region,
                                )
                            )
                        except:
                            logging.debug(
                                "trying to get clip %s track %s frame %s",
                                clip_id,
                                track_number,
                                frame_number + track_start,
                                exc_info=True,
                            )

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

    def set_sample_frames(self, clip_id, track_id, sample_frames):
        sample_frames.sort()
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            clip_node = clips[str(clip_id)]
            track_node = clip_node[str(track_id)]
            track_node.attrs["sample_frames"] = np.uint16(sample_frames)

    def add_prediction(self, clip_id, track_id, track_prediction):
        logging.warn("Not adding prediction data as code needs to be written")

    # TODO IF NEEDED
    #     with HDF5Manager(self.database, "a") as f:
    #         clip = f["clips"][(str(clip_id))]
    #         track_node = clip[str(track_id)]
    #         predicted_tag = track_prediction.predicted_tag()
    #         if track_prediction.num_frames_classified > 0:
    #             self.all_class_confidences = track_prediction.class_confidences()
    #             predictions = np.int16(
    #                 np.around(100 * np.array(track_prediction.predictions))
    #             )
    #             predicted_confidence = int(round(100 * track_prediction.max_score))
    #
    #             self.add_prediction_data(
    #                 track_node,
    #                 predictions,
    #             )
    #         clip.attrs["has_prediction"] = True

    def add_prediction_data(self, track, model_predictions):
        """
        Add prediction data as a dataset to the track
        data should be  an array of int16 array
        """
        track_attrs = track.attrs

        model_group = track.create_group("model_predictions")
        for prediction in model_predictions:
            key = f'{prediction.get("model_id", 0)}'
            if key in model_group:
                continue
            pred_g = model_group.create_group(key)
            predicted_tag = prediction.get("label")
            if predicted_tag is not None:
                pred_g.attrs["correct_prediction"] = track_attrs["tag"] == predicted_tag
                pred_g.attrs["predicted"] = predicted_tag
                pred_g.attrs["confidence"] = prediction.get("confidence", 0)
            # track_attrs["predicted_confidence"] = prediction.get("confidence", 0)

            prediction_data = np.int16(prediction.get("predictions"))
            raw_predictions = pred_g.create_dataset(
                "predictions",
                prediction_data.shape,
                chunks=prediction_data.shape,
                dtype=prediction_data.dtype,
            )
            raw_predictions[:, :] = prediction_data
            labels = prediction.get("labels")
            if labels is not None:
                pred_g.attrs["prediction_classes"] = labels
            track_attrs["has_prediction"] = True

    def get_unique_clip_id(self):
        clip_ids = list(self.get_all_clip_ids().keys())
        clip_ids = np.array(clip_ids).astype(np.int)
        if len(clip_ids) == 0:
            return 1
        max_clip = np.amax(clip_ids)
        return max_clip + 1

    def add_track(
        self,
        clip_id,
        track,
        cropped_data,
        sample_frames=None,
        opts=None,
        start_time=None,
        end_time=None,
        prediction=None,
        prediction_classes=None,
        original_thermal=None,
    ):
        """
        Adds track to database.
        :param clip_id: id of the clip to add track to write
        :param cropped_data: data for track, list of numpy arrays of shape [channels, height, width]
        :param track: the original track record, used to get stats for track
        :param opts: additional parameters used when creating dataset, if not provided defaults to no compression.
        """

        track_id = str(track.get_id())
        logging.info("Adding track %s", track_id)
        if opts is None:
            opts = {}
        with HDF5Manager(self.database, "a") as f:
            clips = f["clips"]
            clip_node = clips[clip_id]
            has_prediction = False
            track_node = clip_node.create_group(track_id)
            cropped_frame = track_node.create_group("cropped")
            if "frames" in clip_node:
                original_group = clip_node["frames"]
            else:
                original_group = clip_node.create_group("frames")
            skipped_frames = []
            # write each frame out individually, as they will probably be different sizes.
            original = None
            for frame_i, cropped in enumerate(cropped_data):
                if original_thermal is not None:
                    original = original_thermal[frame_i]
                # using a chunk size of 1 for channels has the advantage that we can quickly load just one channel

                if (
                    original is not None
                    and str(frame_i + track.start_frame) not in original_group
                ):
                    thermal_node = original_group.create_dataset(
                        str(frame_i + track.start_frame),
                        original.shape,
                        chunks=original.shape,
                        **opts,
                        dtype=np.int16,
                    )
                if original is not None:
                    thermal_node[:, :] = original
                if cropped.thermal.size > 0:
                    height, width = cropped.shape
                    chunks = (1, height, width)
                    cropped_array = cropped.as_array()
                    dims = cropped_array.shape
                    frame_node = cropped_frame.create_dataset(
                        str(frame_i), dims, chunks=chunks, **opts, dtype=np.int16
                    )
                    frame_node[:, :, :] = cropped_array
                else:
                    skipped_frames.append(frame_i)

            # write out attributes
            track_stats = track.get_stats()
            node_attrs = track_node.attrs
            node_attrs["id"] = track_id
            if track.tags:
                node_attrs["track_tags"] = json.dumps(track.tags)
            if sample_frames is not None:
                node_attrs["sample_frames"] = np.uint16(sample_frames)
            node_attrs["tag"] = track.tag
            node_attrs["frames"] = len(cropped_data)
            node_attrs["skipped_frames"] = np.uint16(skipped_frames)
            node_attrs["start_frame"] = track.start_frame
            node_attrs["end_frame"] = track.end_frame
            if track.predictions is not None:
                self.add_prediction_data(
                    track_node,
                    track.predictions,
                )
                has_prediction = True
            if track.confidence:
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
            clip_node.attrs["has_prediction"] = has_prediction


def hdf5_attributes_dictionary(dataset):
    result = {}
    for key, value in dataset.attrs.items():
        result[key] = value
    return result
