"""
Author: Matthew Aitchison
Date: December 2017

Dataset used for training a tensorflow model from track data.

Tracks are broken into segments.  Filtered, and then passed to the trainer using a weighted random sample.

"""
import logging
import math
import multiprocessing
import os
import queue
import random
import threading
import time
import dateutil
import numpy as np
import gc
from ml_tools.datasetstructures import TrackHeader, SegmentHeader
from ml_tools.trackdatabase import TrackDatabase
from ml_tools import tools
from ml_tools import imageprocessing

# from ml_tools.kerasmodel import KerasModel


class TrackChannels:
    """Indexes to channels in track."""

    thermal = 0
    filtered = 1
    flow_h = 2
    flow_v = 3
    mask = 4
    flow = 5


def dataset_db_path(config):
    return os.path.join(config.tracks_folder, "datasets.dat")


class Dataset:
    """
    Stores visit, clip, track, and segment information headers in memory, and allows track / segment streaming from
    disk.
    """

    # Number of threads to use for async loading
    WORKER_THREADS = 2

    # If true uses processes instead of threads.  Threads do not scale as well due to the GIL, however there is no
    # transfer time required per segment.  Processes scale much better but require ~1ms to pickling the segments
    # across processes.
    # In general if worker threads is one set this to False, if it is two or more set it to True.
    PROCESS_BASED = True

    # number of pixels to inset from frame edges by default
    DEFAULT_INSET = 2

    def __init__(
        self,
        db_file,
        name="Dataset",
        config=None,
        use_segments=True,
        use_predictions=False,
        consecutive_segments=False,
        labels=[],
    ):
        self.consecutive_segments = consecutive_segments
        # self.camera_bins = {}
        self.use_segments = use_segments
        # database holding track data
        self.db_file = db_file
        self.db = None
        self.load_db()
        self.label_mapping = None
        # name of this dataset
        self.name = name
        # list of our tracks
        self.tracks = []
        self.tracks_by_label = {}
        self.tracks_by_bin = {}
        self.tracks_by_id = {}
        self.camera_names = set()
        # self.cameras_by_id = {}

        # cumulative distribution function for segments.  Allows for super fast weighted random sampling.
        self.segment_cdf = []
        self.segment_label_cdf = {}
        # segments list
        self.segments = []
        self.segments_by_label = {}
        self.segments_by_id = {}

        self.frame_cdf = []
        self.frame_label_cdf = {}

        self.frame_samples = []
        self.frames_by_label = {}
        self.frames_by_id = {}

        # list of label names
        self.labels = labels
        self.label_mapping = None

        self.enable_augmentation = False
        self.label_caps = {}

        if config:
            self.min_frame_mass = config.build.train_min_mass
            self.segment_length = config.build.segment_length
            # number of seconds segments are spaced apart
            self.segment_spacing = config.build.segment_spacing
            self.banned_clips = config.build.banned_clips
            self.included_labels = config.labels
            self.clip_before_date = config.build.clip_end_date
            self.segment_min_mass = config.build.train_min_mass
        else:
            # number of seconds each segment should be
            self.segment_length = 3
            # number of seconds segments are spaced apart
            self.segment_spacing = 1
        self.filtered_stats = {
            "confidence": 0,
            "trap": 0,
            "banned": 0,
            "date": 0,
            "tags": 0,
            "segment_mass": 0,
            "no_data": 0,
        }
        self.lbl_p = None
        self.numpy_file = None

    def clear_tracks(self):
        del self.tracks
        del self.tracks_by_label
        del self.tracks_by_bin
        del self.tracks_by_id
        # self.tracks = []
        # self.tracks_by_label = {}
        # self.tracks_by_bin = {}
        # self.tracks_by_id = {}

    def load_db(self):
        self.db = TrackDatabase(self.db_file)

    def clear_unused(self):
        if self.use_segments:

            del self.frame_cdf
            del self.frame_label_cdf

            del self.frame_samples
            del self.frames_by_label
            del self.frames_by_id
            # self.frame_cdf = []
            # self.frame_label_cdf = {}
            #
            # self.frame_samples = []
            # self.frames_by_label = {}
            # self.frames_by_id = {}
        else:
            self.segment_cdf = []
            self.segment_label_cdf = {}
            # segments list
            self.segments = []
            self.segments_by_label = {}
            self.segments_by_id = {}
        gc.collect()

    def set_read_only(self, read_only):
        if self.db is not None:
            self.db.set_read_only(read_only)

    def random_segments_only(self):
        remove = [segment for segment in self.segments if segment.top_mass]
        print("removing the tops", len(remove))
        for segment in remove:
            segment.track.segments.remove(segment)
            self.segments_by_label[segment.label].remove(segment)
            del self.segments_by_id[segment.id]

        self.segments = [segment for segment in self.segments if not segment.top_mass]

        self.rebuild_cdf()

    def highest_mass_only(self, best_only=False):
        if best_only:
            remove = [segment for segment in self.segments if not segment.best_mass]
        else:
            remove = [segment for segment in self.segments if not segment.top_mass]

        for segment in remove:
            segment.track.segments.remove(segment)
            self.segments_by_label[segment.label].remove(segment)
            del self.segments_by_id[segment.id]
        if best_only:
            self.segments = [segment for segment in self.segments if segment.best_mass]
        else:
            self.segments = [segment for segment in self.segments if segment.top_mass]

        self.rebuild_cdf()

    @property
    def sample_count(self):
        return len(self.samples())

    def samples(self):
        if self.use_segments:
            return self.segments
        return self.frame_samples

    def set_samples(self, samples):
        if self.use_segments:
            self.segments = samples
        else:
            self.frame_samples = samples

    def set_samples_for(self, label, samples):
        if self.use_segments:
            self.segments_by_label[label] = samples
        else:
            self.frames_by_label[label] = samples

    def get_label_caps(self, labels, remapped=False):
        counts = []
        for label in labels:
            counts.append(len(self.samples_for(label, remapped=remapped)))
        index = math.floor(len(counts) * 0.40)
        counts.sort()
        birds = self.samples_for("bird", remapped=remapped)
        if len(birds) > 0:
            return len(birds)
        # return 4096

        return counts[index]
        # return int(np.percentile(counts, 25))

    def samples_for(self, label, remapped=False):
        labels = []
        if remapped and self.label_mapping:
            labels = [
                key
                for key, mapped in self.label_mapping.items()
                if mapped.lower() == label.lower()
            ]
            labels.sort()
        else:
            labels.append(label)
        samples = []
        for l in labels:
            if self.use_segments:
                samples.extend(self.segments_by_label.get(l, []))
            else:
                samples.extend(self.frames_by_label.get(l, []))
        return samples

    def get_counts(self, label):
        """
        Gets number of examples for given label
        :label: label to check
        :return: (segments, tracks, bins, weight)
        """
        segments = 0
        tracks = 0
        bins = 0
        weight = 0
        frames = 0
        if self.label_mapping:
            for key, value in self.label_mapping.items():
                if key == label or value == label:
                    label_tracks = self.tracks_by_label.get(key, [])
                    tracks += len(label_tracks)
                    segments += sum(len(track.segments) for track in label_tracks)
                    frames += sum(
                        len(track.get_sample_frames()) for track in label_tracks
                    )

        else:
            label_tracks = self.tracks_by_label.get(label, [])
            segments = sum(len(track.segments) for track in label_tracks)
            weight = self.get_label_weight(label)
            tracks = len(label_tracks)
            frames = sum(len(track.get_sample_frames()) for track in label_tracks)
            bins = len(
                [
                    tracks
                    for bin_name, tracks in self.tracks_by_bin.items()
                    if len(tracks) > 0 and tracks[0].label == label
                ]
            )
        return segments, frames, tracks, bins, weight

    def load_tracks(self, shuffle=False, before_date=None, after_date=None):
        """
        Loads track headers from track database with optional filter
        :return: [number of tracks added, total tracks].
        """
        labels = self.db.get_labels()
        counter = 0
        track_ids = self.db.get_all_track_ids(
            before_date=before_date, after_date=after_date
        )
        if shuffle:
            np.random.shuffle(track_ids)
        for clip_id, track_id in track_ids:
            if self.load_track(clip_id, track_id, labels):
                counter += 1
        return [counter, len(track_ids)]

    def add_tracks(self, tracks, max_segments_per_track=None):
        """
        Adds list of tracks to dataset
        :param tracks: list of TrackHeader
        :param track_filter: optional filter
        """
        result = 0
        for track in tracks:
            if self.add_track_header(track, max_segments_per_track):
                result += 1
        return result

    def add_track_header(self, track_header, max_segments_per_track=None):
        if track_header.unique_id in self.tracks_by_id:
            return False

        # gp test less segments more tracks
        if max_segments_per_track is not None:
            if len(track_header.segments) > max_segments_per_track:
                segments = random.sample(track_header.segments, max_segments_per_track)
                track_header.segments = segments

        self.tracks.append(track_header)
        self.add_track_to_mappings(track_header)
        self.segments.extend(track_header.segments)
        return True

    def load_track(self, clip_id, track_id, labels):
        """
        Creates segments for track and adds them to the dataset
        :param clip_id: id of tracks clip
        :param track_id: track number
        :param track_filter: if provided a function filter(clip_meta, track_meta) that returns true when a track should
                be ignored)
        :return: True if track was added, false if it was filtered out.
        :return:
        """

        # make sure we don't already have this track
        if "{}-{}".format(clip_id, track_id) in self.tracks_by_bin:
            return False

        clip_meta = self.db.get_clip_meta(clip_id)
        track_meta = self.db.get_track_meta(clip_id, track_id)
        # predictions = self.db.get_track_predictions(clip_id, track_id)
        if self.filter_track(clip_meta, track_meta):
            return False
        track_header = TrackHeader.from_meta(clip_id, clip_meta, track_meta)
        self.tracks.append(track_header)

        segment_frame_spacing = int(
            round(self.segment_spacing * track_header.frames_per_second)
        )
        segment_width = int(round(self.segment_length * track_header.frames_per_second))
        track_header.calculate_segments(
            track_meta["mass_history"],
            segment_frame_spacing,
            segment_width,
            self.segment_min_mass,
        )

        self.filtered_stats["segment_mass"] += track_header.filtered_stats[
            "segment_mass"
        ]
        self.segments.extend(track_header.segments)
        self.add_track_to_mappings(track_header)

        return True

    def filter_track(self, clip_meta, track_meta):
        # some clips are banned for various reasons
        source = os.path.basename(clip_meta["filename"])
        if self.banned_clips and source in self.banned_clips:
            self.filtered_stats["banned"] += 1
            return True
        if "tag" not in track_meta:
            self.filtered_stats["tags"] += 1
            return True
        if track_meta["tag"] not in self.included_labels:
            self.filtered_stats["tags"] += 1
            return True

        # filter by date
        if (
            self.clip_before_date
            and dateutil.parser.parse(clip_meta["start_time"]).date()
            > self.clip_before_date.date()
        ):
            self.filtered_stats["date"] += 1
            return True

        # always let the false-positives through as we need them even though they would normally
        # be filtered out.
        if "bounds_history" not in track_meta or len(track_meta["bounds_history"]) == 0:
            self.filtered_stats["no_data"] += 1
            return True

        if track_meta["tag"] == "false-positive":
            return False

        # for some reason we get some records with a None confidence?
        if track_meta.get("confidence", 0.0) <= 0.6:
            self.filtered_stats["confidence"] += 1
            return True

        # remove tracks of trapped animals
        if (
            "trap" in clip_meta.get("event", "").lower()
            or "trap" in clip_meta.get("trap", "").lower()
        ):
            self.filtered_stats["trap"] += 1
            return True

        return False

    def add_track_to_mappings(self, track_header):
        if self.label_mapping and track_header.label in self.label_mapping:
            track_header.label = self.mapped_label(track_header.label)

        self.tracks_by_id[track_header.unique_id] = track_header
        bins = self.tracks_by_bin.setdefault(track_header.bin_id, [])
        bins.append(track_header)

        if track_header.label not in self.labels:
            self.labels.append(track_header.label)
        if track_header.label not in self.tracks_by_label:
            self.tracks_by_label[track_header.label] = []
        self.tracks_by_label[track_header.label].append(track_header)
        segs = self.segments_by_label.setdefault(track_header.label, [])
        segs.extend(track_header.segments)
        for seg in segs:
            self.segments_by_id[seg.id] = seg
        frames = self.frames_by_label.setdefault(track_header.label, [])
        samples = track_header.get_sample_frames()
        for sample in samples:
            self.frames_by_id[sample.id] = sample
        self.frame_samples.extend(samples)
        frames.extend(samples)
        # camera = self.cameras_by_id.setdefault(
        #     track_header.camera_id, Camera(track_header.camera_id)
        # )
        self.camera_names.add(track_header.camera_id)
        # camera.add_track(track_header)

    def filter_segments(self, avg_mass, ignore_labels=None):
        """
        Removes any segments with an average mass less than the given avg_mass
        :param avg_mass: segments with less avarage mass per frame than this will be removed from the dataset.
        :param ignore_labels: these labels will not be filtered
        :return: number of segments removed
        """

        num_filtered = 0
        new_segments = []

        for segment in self.segments:

            pass_mass = segment.avg_mass >= avg_mass
            if (not ignore_labels and segment.label in ignore_labels) or (pass_mass):
                new_segments.append(segment)
            else:
                num_filtered += 1

        self.segments = new_segments

        self._purge_track_segments()

        self.rebuild_cdf()

        return num_filtered

    def fetch_track(
        self,
        track: TrackHeader,
        original=False,
        important_frames=False,
    ):
        """
        Fetches data for an entire track
        :param track: the track to fetch
        :return: segment data of shape [frames, channels, height, width]
        """
        frame_numbers = None
        if important_frames:
            frame_numbers = [frame.frame_num for frame in track.important_frames]
            frame_numbers.sort()
        frames = self.db.get_track(
            track.clip_id,
            track.track_id,
            original=original,
            frame_numbers=frame_numbers,
        )
        return frames

    def fetch_random_sample(self, sample, channel=None):
        important_frames = sample.track.important_frames
        np.random.shuffle(important_frames)
        important_frames = important_frames[: sample.frames]
        important_frames = [frame.frame_num for frame in important_frames]
        important_frames.sort()
        frames = self.db.get_track(
            sample.track.clip_id,
            sample.track.track_id,
            frame_numbers=important_frames,
            channels=channel,
        )
        return frames

    def fetch_segment_data(self, sample, channel=None):

        frames = self.db.get_track(
            sample.track.clip_id,
            sample.track.track_id,
            frame_numbers=sample.frame_indices,
            channels=TrackChannels.thermal,
        )
        background = self.db.get_clip_background(sample.track.clip_id)
        for frame in frames:
            region = sample.track.track_bounds[frame.frame_number]
            region = tools.Rectangle.from_ltrb(*region)
            cropped = region.subimage(background)
            frame.filtered = frame.thermal - cropped
        return frames

    def fetch_frame(self, frame_sample, channels=None):
        frame = self.db.get_frame(
            frame_sample.clip_id,
            frame_sample.track_id,
            frame_sample.frame_num,
        )

        return data

    def epoch_samples(
        self, cap_samples=None, replace=True, random=True, cap_at=None, label_cap=None
    ):
        if len(self.labels) == 0:
            return []
        labels = self.labels.copy()
        samples = []

        if (cap_at or cap_samples) and label_cap is None:
            if cap_at:
                label_cap = len(self.samples_for(cap_at, remapped=True))
            else:
                label_cap = self.get_label_caps(labels, remapped=True)

        cap = None
        for label in labels:
            if label_cap:
                cap = min(label_cap, len(self.samples_for(label, remapped=True)))
            if label == "false-positive":
                if cap is None:
                    cap = int(label_cap * 0.5)
                else:
                    cap = min(cap, int(label_cap * 0.5))
            new = self.get_sample(cap=cap, replace=replace, label=label, random=random)
            if new is not None and len(new) > 0:
                samples.extend(new)
        labels = [sample.label for sample in samples]
        return samples

    def cdf(self):
        if self.use_segments:
            return self.segment_cdf
        return self.frame_cdf

    def label_cdf(self, label):
        if self.use_segments:
            return self.segment_label_cdf.get(label, [])
        return self.frame_label_cdf.get(label, [])

    def get_sample(self, cap=None, replace=True, label=None, random=True):
        """Returns a random frames from weighted list."""
        if label:
            samples = self.samples_for(label, remapped=True)
            cdf = self.label_cdf(label)
        else:
            samples = self.samples()
            cdf = self.cdf()
        if not samples:
            return None
        if cap is None:
            return samples
        if random:
            return np.random.choice(samples, cap, replace=replace, p=cdf)
        else:
            cap = min(cap, len(samples))
            return samples[:cap]

    def balance_weights(self, weight_modifiers=None):
        """
        Adjusts weights so that every class is evenly represented.
        :param weight_modifiers: if specified is a dictionary mapping from label to weight modifier,
            where < 1 sampled less frequently, and > 1 is sampled more frequently.
        :return:
        """

        label_weight = {}
        mean_label_weight = 0

        for label in self.labels:
            label_weight[label] = self.get_label_weight(label)
            mean_label_weight += label_weight[label] / len(self.labels)

        scale_factor = {}
        for label in self.labels:
            modifier = (
                1.0 if weight_modifiers is None else weight_modifiers.get(label, 1.0)
            )
            if label_weight[label] == 0:
                scale_factor[label] = 1.0
            else:
                scale_factor[label] = mean_label_weight / label_weight[label] * modifier

        for segment in self.segments:
            segment.weight *= scale_factor.get(segment.label, 1.0)
        self.rebuild_cdf()

    def balance_bins(self, max_bin_weight=None):
        """
        Adjusts weights so that bins with a number number of segments aren't sampled so frequently.
        :param max_bin_weight: bins with more weight than this number will be scaled back to this weight.
        """

        for bin_name, tracks in self.tracks_by_bin.items():
            bin_weight = sum(track.weight for track in tracks)
            if bin_weight == 0:
                continue
            if max_bin_weight is None:
                scale_factor = 1 / bin_weight
                # means each bin has equal possiblity
            elif bin_weight > max_bin_weight:
                scale_factor = max_bin_weight / bin_weight
            else:
                scale_factor = 1
            for track in tracks:
                for segment in track.segments:
                    segment.weight *= scale_factor
        self.rebuild_cdf()

    def get_bin_segments_count(self, bin_id):
        return sum(len(track.segments) for track in self.tracks_by_bin[bin_id])

    def get_bin_max_track_duration(self, bin_id):
        return max(track.duration for track in self.tracks_by_bin[bin_id])

    def remove_label(self, label_to_remove):
        """
        Removes all segments of given label from dataset. Label remains in dataset.labels however, so as to not
        change the ordinal value of the labels.
        """
        if label_to_remove not in self.labels:
            return
        self.segments = [
            segment for segment in self.segments if segment.label != label_to_remove
        ]
        self._purge_track_segments()
        self.rebuild_cdf()

    def _purge_track_segments(self):
        """Removes any segments from track_headers where the segment has been deleted"""
        # remove segments from tracks
        for track in self.tracks:
            track.segments = []

    def mapped_label(self, label):
        if self.label_mapping:
            return self.label_mapping.get(label, label)
        return label

    def rebuild_cdf(self, lbl_p=None):
        """Calculates the CDF used for fast random sampling for frames and
        segments, if balance labels is set each label has an equal chance of
        being chosen
        """
        if lbl_p is None:
            lbl_p = self.lbl_p
        self.rebuild_segment_cdf(lbl_p=lbl_p)
        self.rebuild_frame_cdf(lbl_p=lbl_p)

    def rebuild_frame_cdf(self, lbl_p=None):
        self.frame_cdf = []
        total = 0
        self.frame_label_cdf = {}

        for track in self.tracks:
            for frame in track.important_frames:
                frame_weight = track.frame_weight
                if lbl_p and track.label in lbl_p:
                    frame_weight *= lbl_p[track.label]
                total += frame_weight

                self.frame_cdf.append(frame_weight)

                cdf = self.frame_label_cdf.setdefault(track.label, [])
                cdf.append(track.frame_weight)

        if len(self.frame_cdf) > 0:
            self.frame_cdf = [x / total for x in self.frame_cdf]

        for key, cdf in self.frame_label_cdf.items():
            total = sum(cdf)
            self.frame_label_cdf[key] = [x / total for x in cdf]

        if self.label_mapping:
            mapped_cdf = {}
            labels = list(self.label_mapping.keys())
            labels.sort()
            for label in labels:
                if label not in self.frame_label_cdf:
                    continue
                label_cdf = self.frame_label_cdf[label]
                new_label = self.label_mapping[label]
                if lbl_p and label in lbl_p:
                    label_cdf = np.float64(label_cdf)
                    label_cdf *= lbl_p[label]
                cdf = mapped_cdf.setdefault(new_label, [])
                cdf.extend(label_cdf)

            for key, cdf in mapped_cdf.items():
                total = sum(cdf)
                mapped_cdf[key] = [x / total for x in cdf]
            self.frame_label_cdf = mapped_cdf

    def rebuild_segment_cdf(self, lbl_p=None):
        """Calculates the CDF used for fast random sampling"""
        self.segment_cdf = []
        total = 0
        self.segment_label_cdf = {}
        for segment in self.segments:
            seg_weight = segment.weight
            if lbl_p and segment.label in lbl_p:
                seg_weight *= lbl_p[segment.label]
            total += seg_weight
            self.segment_cdf.append(seg_weight)

        # guarantee it's in the order we will sample by
        for label, segments in self.segments_by_label.items():
            cdf = self.segment_label_cdf.setdefault(label, [])
            for segment in segments:
                cdf.append(segment.weight)

        if len(self.segment_cdf) > 0:
            self.segment_cdf = [x / total for x in self.segment_cdf]
        for key, cdf in self.segment_label_cdf.items():
            total = sum(cdf)
            if total > 0:
                self.segment_label_cdf[key] = [x / total for x in cdf]
            else:
                self.segment_label_cdf[key] = []
        # do this after so labels are balanced
        if self.label_mapping:
            mapped_cdf = {}
            labels = list(self.label_mapping.keys())
            labels.sort()
            for label in labels:
                if label not in self.segment_label_cdf:
                    continue
                label_cdf = self.segment_label_cdf[label]
                new_label = self.label_mapping[label]

                if lbl_p and label in lbl_p:
                    label_cdf = np.float64(label_cdf)
                    label_cdf *= lbl_p[label]
                cdf = mapped_cdf.setdefault(new_label, [])
                cdf.extend(label_cdf)

            for key, cdf in mapped_cdf.items():
                total = sum(cdf)

                mapped_cdf[key] = [x / total for x in cdf]
            self.segment_label_cdf = mapped_cdf

    def get_label_weight(self, label):
        """Returns the total weight for all segments of given label."""
        tracks = self.tracks_by_label.get(label)
        return sum(track.weight for track in tracks) if tracks else 0

    def get_label_segments(self, label):
        """Returns all segments of given class."""
        return self.segments_by_label.get(label, [])

    def regroup(
        self,
        groups,
        shuffle=True,
    ):
        """
        regroups the dataset so multiple animals can be under a single label
        """
        self.label_mapping = {}
        counts = []
        tracks_by_bin = {}
        samples = []
        for mapped_label, labels in groups.items():
            count = 0
            for label in labels:
                lbl_samples = self.samples_for(label)
                count += len(lbl_samples)
                samples.extend(lbl_samples)
                self.label_mapping[label] = mapped_label
                for sample in lbl_samples:
                    track = self.tracks_by_id[sample.unique_track_id]
                    tracks_by_bin[track.bin_id] = track
            counts.append(count)

        self.labels = list(groups.keys())
        self.tracks_by_bin = tracks_by_bin
        self.set_samples(samples)
        if self.use_segments:
            self.segments_by_id == {}
            for seg in samples:
                self.segments_by_id[seg.id] = seg

            if shuffle:
                np.random.shuffle(self.segments)
        elif shuffle:
            self.frames_by_id == {}
            for sample in samples:
                self.frames_by_id[sample.id] = sample
            np.random.shuffle(self.frame_samples)
        self.rebuild_cdf()

    def has_data(self):
        if self.use_segments:
            return len(self.segments) > 0
        else:
            return len(self.frame_samples) > 0

    def recalculate_segments(self, scale=1.0, segment_type=1):
        self.segments_by_id.clear()
        self.segments_by_label.clear()
        del self.segments[:]
        del self.segments
        self.segments = []
        self.segments_by_label = {}
        self.segments_by_id = {}
        logging.info(
            "%s generating segments scale %s type %s", self.name, scale, segment_type
        )
        start = time.time()
        empty_tracks = []
        filtered_stats = 0

        for track in self.tracks:
            segment_frame_spacing = int(
                round(self.segment_spacing * track.frames_per_second)
            )
            segment_width = int(round(self.segment_length * track.frames_per_second))
            use_important = True
            random_frames = True
            top_frames = False
            random_sections = False
            segment_min_mass = self.segment_min_mass
            if segment_type == 0:
                use_important = True
                random_frames = True
                segment_min_mass = self.segment_min_mass
            elif segment_type == 1:
                use_important = False
                random_frames = True
                segment_min_mass = self.segment_min_mass
            elif segment_type == 2:
                use_important = True
                random_frames = False
            elif segment_type == 3:
                use_important = False
                random_frames = False
                segment_min_mass = self.segment_min_mass
            elif segment_type == 4:
                top_frames = True
            elif segment_type == 5:
                use_important = False
                random_frames = True
                segment_min_mass = self.segment_min_mass
                random_sections = True
            elif segment_type == 6:
                use_important = False
                random_frames = False
                segment_min_mass = None
            elif segment_type == 7:
                use_important = False
                random_frames = True
                top_frames = True
                segment_min_mass = None
            track.calculate_segments(
                track.frame_mass,
                segment_frame_spacing,
                segment_width,
                scale=scale,
                random_frames=random_frames,
                use_important=use_important,
                top_frames=top_frames,
                segment_min_mass=segment_min_mass,
                random_sections=random_sections,
            )
            filtered_stats = filtered_stats + track.filtered_stats["segment_mass"]
            if len(track.segments) == 0:
                empty_tracks.append(track)
                continue
            for seg in track.segments:
                self.segments_by_id[seg.id] = seg

            self.segments.extend(track.segments)
            segs = self.segments_by_label.setdefault(track.label, [])
            segs.extend(track.segments)
        # for track in empty_tracks:
        #     self.remove_track(track)

        self.rebuild_cdf()
        logging.info(
            "%s #segments %s filtered stats are %s took  %s",
            self.name,
            len(self.segments),
            filtered_stats,
            time.time() - start,
        )
        # print(self.name, "has", len(self.segments))
        # for segment in self.segments:
        #     print(
        #         f"{segment.track.label} - {segment.track.clip_id}-{segment.track.track_id} -{segment.track.start_frame} segment {segment.id} {segment.start_frame} frame_indices {segment.frame_indices} best? { segment.best_mass} top {segment.top_mass}"
        #     )

    def remove_track(self, track):
        self.tracks.remove(track)
        del self.tracks_by_id[track.unique_id]
        if track.bin_id in self.tracks_by_bin:
            del self.tracks_by_bin[track.bin_id]
        self.tracks_by_label[track.label].remove(track)

    def add_important(self):
        for track_header in self.tracks:
            frames = self.db.get_track(track_header.clip_id, track_header.track_id)
            track_header.set_important_frames(self.min_frame_mass, frame_data=frames)
            self.db.set_important_frames(
                track_header.clip_id,
                track_header.track_id,
                [sample.frame_num for sample in track_header.important_frames],
            )

    def add_overlay(self):
        for track in self.tracks:
            if self.db.has_overlay(track.clip_id, track.track_id):
                continue
            print("no overlay for ", track.clip_id, track.track_id)
            frames = self.db.get_track(track.clip_id, track.track_id)
            regions = []
            for region in track.track_bounds:
                regions.append(tools.Rectangle.from_ltrb(*region))

            _, overlay = imageprocessing.movement_images(
                frames,
                regions,
                dim=(120, 160),
                require_movement=True,
            )
            self.db.add_overlay(track.clip_id, track.track_id, overlay)
