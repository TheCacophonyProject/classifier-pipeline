"""
Author: Matthew Aitchison
Date: December 2017

Dataset used for training a tensorflow model from track data.

Tracks are broken into segments.  Filtered, and then passed to the trainer using a weighted random sample.

"""
import logging
import math
import os
import time
import numpy as np
import gc
from ml_tools.datasetstructures import NumpyMeta, TrackHeader, TrackingSample
from ml_tools.trackdatabase import TrackDatabase
from ml_tools import tools

# from ml_tools.kerasmodel import KerasModel
from enum import Enum


class SegmentType(Enum):
    IMPORTANT_RANDOM = 0
    ALL_RANDOM = 1
    IMPORTANT_SEQUENTIAL = 2
    ALL_SEQUENTIAL = 3
    TOP_SEQUENTIAL = 4
    ALL_RANDOM_SECTIONS = 5
    TOP_RANDOM = 6
    ALL_RANDOM_NOMIN = 6


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
        self.samples_by_label = {}
        self.samples_by_bin = {}
        self.samples = []
        self.sample_cdf = []
        self.tracks = []
        self.tracks_by_label = {}
        self.tracks_by_bin = {}
        self.tracks_by_id = {}
        self.camera_names = set()
        # self.cameras_by_id = {}

        # list of label names
        self.labels = labels
        self.label_mapping = None

        self.enable_augmentation = False
        self.label_caps = {}

        if config:
            self.segment_length = config.build.segment_length
            # number of seconds segments are spaced apart
            self.segment_spacing = config.build.segment_spacing
            self.banned_clips = config.build.banned_clips
            self.included_labels = config.labels
            self.segment_min_avg_mass = config.build.segment_min_avg_mass
        else:
            # number of seconds each segment should be
            self.segment_length = 25
            # number of seconds segments are spaced apart
            self.segment_spacing = 1
            self.segment_min_avg_mass = None
        self.filtered_stats = {
            "confidence": 0,
            "trap": 0,
            "banned": 0,
            "date": 0,
            "tags": 0,
            "segment_mass": 0,
            "no_data": 0,
            "not-confirmed": 0,
        }
        self.lbl_p = None
        self.numpy_data = None

    # is much faster to read from numpy array when trianing
    def saveto_numpy(self, path):
        file = os.path.join(path, self.name)
        self.numpy_data = NumpyMeta(f"{file}.npy")
        self.numpy_data.save_tracks(self.db, self.tracks)
        self.numpy_data.f = None

    def clear_tracks(self):
        del self.tracks
        del self.tracks_by_label
        del self.tracks_by_bin
        del self.tracks_by_id

    def load_db(self):
        self.db = TrackDatabase(self.db_file)

    def clear_samples(self):
        self.samples_by_label = {}
        self.samples_by_bin = {}
        self.samples = []
        self.sample_cdf = []
        gc.collect()

    def set_read_only(self, read_only):
        if self.db is not None:
            self.db.set_read_only(read_only)

    # def highest_mass_only(self):
    #     # top_frames for i3d generates all segments above a  min average mass
    #     # use this to take only the best
    #     remove = [segment for segment in self.segments if not segment.best_mass]
    #
    #     for segment in remove:
    #         segment.track.segments.remove(segment)
    #         self.segments_by_label[segment.label].remove(segment)
    #         del self.segments_by_id[segment.id]
    #     self.segments = [segment for segment in self.segments if segment.best_mass]
    #
    #     self.rebuild_cdf()

    @property
    def sample_count(self):
        return len(self.samples)

    #
    # def samples(self):
    #     if self.use_segments:
    #         return self.segments
    #     return self.frame_samples

    def set_samples(self, samples):
        self.samples = samples

    def set_samples_for(self, label, samples):
        self.samples_by_label[label] = samples

    def get_label_caps(self, labels, remapped=False):
        counts = []
        for label in labels:
            counts.append(len(self.samples_for(label, remapped=remapped)))
        index = math.floor(len(counts) * 0.40)
        counts.sort()
        birds = self.samples_for("bird", remapped=remapped)
        if len(birds) > 0:
            return len(birds)

        return counts[index]

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
            samples.extend(self.samples_by_label.get(l, []))
        return samples

    def get_counts(self, label):
        """
        Gets number of examples for given label
        :label: label to check
        :return: (segments, tracks, bins, weight)
        """
        tracks = 0
        bins = 0
        weight = 0
        samples = 0
        if self.label_mapping:
            for key, value in self.label_mapping.items():
                if key == label or value == label:
                    label_tracks = self.tracks_by_label.get(key, [])
                    tracks += len(label_tracks)
                    samples += len(self.samples_by_label.get(key, []))
                    # segments += sum(len(track.segments) for track in label_tracks)
                    # frames += sum(
                    #     len(track.get_sample_frames())
                    #     for track in label_tracks
                    #     if track.sample_frames is not None
                    # )

        else:
            samples = len(self.samples_by_label.get(label, []))
            label_tracks = self.tracks_by_label.get(label, [])
            weight = self.get_label_weight(label)
            tracks = len(label_tracks)
            bins = len(
                [
                    tracks
                    for bin_name, tracks in self.tracks_by_bin.items()
                    if len(tracks) > 0 and tracks[0].label == label
                ]
            )
        return samples, tracks, bins, weight

    def load_clips(self, shuffle=False, before_date=None, after_date=None, label=None):
        """
        Loads track headers from track database with optional filter
        :return: [number of tracks added, total tracks].
        """
        counter = 0
        clip_ids = self.db.get_all_clip_ids()
        if shuffle:
            np.random.shuffle(clip_ids)
        for clip_id in clip_ids:
            if self.load_clip(clip_id):
                counter += 1
            if counter % 50 == 0:
                logging.debug("Dataset loaded %s / %s", counter, len(clip_ids))

        return [counter, len(clip_ids)]

    def load_clip(self, clip_id):
        clip_meta = self.db.get_clip_meta(clip_id)
        if "tag" not in clip_meta:
            self.filtered_stats["not-confirmed"] += 1
            return False
        clip_id = int(clip_id)
        samples = {}

        # self.clip_samples[clip_id] = samples
        for label, frames in clip_meta.get("tag_frames", {}).items():
            if label == "tag_regions":
                continue
            for frame in frames:
                samples_key = f"{clip_id}-None-{frame}"
                if samples_key in samples:
                    existing_sample = samples[samples_key]
                    existing_sample.labels.append(label)
                else:
                    sample = TrackingSample(
                        clip_id,
                        None,
                        frame,
                        label,
                        clip_meta["frame_temp_median"][frame],
                        None,
                        clip_meta["start_time"],
                        clip_meta.get("device", "unknown"),
                        clip_meta.get("filename", "unknown"),
                    )
                    samples[samples_key] = sample
                    self.add_clip_sample_mappings(sample)

        return True

    def add_samples(self, samples):
        """
        Adds list of samples to dataset
        :param track_filter: optional filter
        """
        result = 0
        for sample in samples:
            if self.add_clip_sample_mappings(sample):
                result += 1
        return result

    def filter_sample(self, sample):
        if sample.label not in self.included_labels:
            self.filtered_stats["tags"] += 1
            return True

    def add_clip_sample_mappings(self, sample):
        if self.filter_sample(sample):
            return False
        self.samples.append(sample)

        if self.label_mapping and sample.label in self.label_mapping:
            sample.label = self.mapped_label(sample.label)

        if sample.label not in self.labels:
            self.labels.append(sample.label)

        if sample.label not in self.samples_by_label:
            self.samples_by_label[sample.label] = []
        self.samples_by_label[sample.label].append(sample)

        bins = self.samples_by_bin.setdefault(sample.bin_id, [])
        bins.append(sample)
        self.camera_names.add(sample.camera)
        return True

    def load_tracks(self, shuffle=False, before_date=None, after_date=None):
        """
        Loads track headers from track database with optional filter
        :return: [number of tracks added, total tracks].
        """
        counter = 0
        track_ids = self.db.get_all_track_ids(
            before_date=before_date, after_date=after_date
        )
        if shuffle:
            np.random.shuffle(track_ids)
        for clip_id, track_id in track_ids:
            if self.load_track(clip_id, track_id):
                counter += 1
            if counter % 50 == 0:
                logging.debug("Dataset loaded %s / %s", counter, len(track_ids))
        return [counter, len(track_ids)]

    def add_tracks(self, tracks):
        """
        Adds list of tracks to dataset
        :param tracks: list of TrackHeader
        :param track_filter: optional filter
        """
        result = 0
        for track in tracks:
            if self.add_track_header(track):
                result += 1
        return result

    def add_track_header(self, track_header):
        if track_header.unique_id in self.tracks_by_id:
            return False

        self.tracks.append(track_header)
        self.add_track_to_mappings(track_header)
        self.segments.extend(track_header.segments)
        return True

    def load_track(self, clip_id, track_id):
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
        if self.filter_track(clip_meta, track_meta):
            print("filtering track", track_meta["id"])
            return False
        track_header = TrackHeader.from_meta(clip_id, clip_meta, track_meta)
        self.tracks.append(track_header)
        if self.use_segments:
            segment_frame_spacing = int(
                round(self.segment_spacing * track_header.frames_per_second)
            )
            segment_width = self.segment_length

            track_header.calculate_segments(
                segment_frame_spacing,
                segment_width,
                self.segment_min_avg_mass,
            )
            self.filtered_stats["segment_mass"] += track_header.filtered_stats[
                "segment_mass"
            ]
            for segment in track_header.segments:
                self.add_clip_sample_mappings(segment)
        else:
            sample_frames = track_header.get_sample_frames()
            for sample in sample_frames:
                self.add_clip_sample_mappings(sample)
        return True

    def filter_track(self, clip_meta, track_meta):
        # some clips are banned for various reasons
        source = os.path.basename(clip_meta["filename"])
        if self.banned_clips and source in self.banned_clips:
            self.filtered_stats["banned"] += 1
            return True
        if "tag" not in track_meta:
            self.filtered_stats["notags"] += 1
            return True
        if track_meta["tag"] not in self.included_labels:
            self.filtered_stats["tags"] += 1
            return True

        # always let the false-positives through as we need them even though they would normally
        # be filtered out.
        if "bounds_history" not in track_meta or len(track_meta["bounds_history"]) == 0:
            self.filtered_stats["no_data"] += 1
            return True

        if track_meta["tag"] == "false-positive":
            return False

        # for some reason we get some records with a None confidence?
        if track_meta.get("confidence", 1.0) <= 0.6:
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
        return self.sample_cdf

    def label_cdf(self, label):
        return self.sample_label_cdf.get(label, [])

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

    def balance_bins(self, max_bin_weight=None):
        """
        Adjusts weights so that bins with a number number of segments aren't sampled so frequently.
        :param max_bin_weight: bins with more weight than this number will be scaled back to this weight.
        """

        for bin_name, samples in self.samples_by_bin.items():
            bin_weight = sum(sample.sample_weight for sample in samples)
            if bin_weight == 0:
                continue
            if max_bin_weight is None:
                scale_factor = 1 / bin_weight
                # means each bin has equal possiblity
            elif bin_weight > max_bin_weight:
                scale_factor = max_bin_weight / bin_weight
            else:
                scale_factor = 1
            for sample in samples:
                sample.weight = np.float16(sample.sample_weight * scale_factor)
        self.rebuild_cdf()

    def remove_label(self, label_to_remove):
        """
        Removes all segments of given label from dataset. Label remains in dataset.labels however, so as to not
        change the ordinal value of the labels.
        """
        if label_to_remove not in self.labels:
            return
        tracks = self.tracks_by_label[label_to_remove]
        for track in tracks:
            self.remove_label()

        self.rebuild_cdf()

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

        self.sample_cdf = []
        total = 0
        self.sample_label_cdf = {}

        for sample in self.samples:
            sample_weight = sample.sample_weight
            if lbl_p and sample.label in lbl_p:
                sample_weight *= lbl_p[sample.label]
            total += sample_weight

            self.sample_cdf.append(sample_weight)

            cdf = self.sample_label_cdf.setdefault(sample.label, [])
            cdf.append(sample.sample_weight)

        if len(self.sample_cdf) > 0:
            self.sample_cdf = [x / total for x in self.sample_cdf]

        for key, cdf in self.sample_label_cdf.items():
            total = sum(cdf)
            self.sample_label_cdf[key] = [x / total for x in cdf]

        if self.label_mapping:
            mapped_cdf = {}
            labels = list(self.label_mapping.keys())
            labels.sort()
            for label in labels:
                if label not in self.sample_label_cdf:
                    continue
                label_cdf = self.sample_label_cdf[label]
                new_label = self.label_mapping[label]
                if lbl_p and label in lbl_p:
                    label_cdf = np.float64(label_cdf)
                    label_cdf *= lbl_p[label]
                cdf = mapped_cdf.setdefault(new_label, [])
                cdf.extend(label_cdf)

            for key, cdf in mapped_cdf.items():
                total = sum(cdf)
                mapped_cdf[key] = [x / total for x in cdf]
            self.sample_label_cdf = mapped_cdf

    def get_label_weight(self, label):
        """Returns the total weight for all segments of given label."""
        samples = self.samples_by_label.get(label)
        return sum(sample.weight for sample in samples) if samples else 0

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
        samples_by_bin = {}
        samples = []
        for mapped_label, labels in groups.items():
            count = 0
            for label in labels:
                lbl_samples = self.samples_for(label)
                count += len(lbl_samples)
                samples.extend(lbl_samples)
                self.label_mapping[label] = mapped_label
                for sample in lbl_samples:
                    samples_by_bin[sample.bin_id] = sample
            counts.append(count)

        self.labels = list(groups.keys())
        self.labels.sort()
        self.samples_by_bin = samples_by_bin
        self.set_samples(samples)
        # self.samples_by_id == {}

        if shuffle:
            np.random.shuffle(self.samples)
        self.rebuild_cdf()

    def has_data(self):
        return len(self.samples) > 0

    def recalculate_segments(self, segment_type=SegmentType.ALL_RANDOM):
        self.samples_by_bin.clear()
        self.samples_by_label.clear()
        del self.samples[:]
        del self.samples
        self.samples = []
        self.samples_by_label = {}
        self.samples_by_bin = {}
        logging.info("%s generating segments  type %s", self.name, segment_type)
        start = time.time()
        empty_tracks = []
        filtered_stats = 0

        for track in self.tracks:
            segment_frame_spacing = int(
                round(self.segment_spacing * track.frames_per_second)
            )
            segment_width = self.segment_length
            use_important = True
            random_frames = True
            top_frames = False
            random_sections = False
            segment_min_avg_mass = self.segment_min_avg_mass
            if segment_type == SegmentType.IMPORTANT_RANDOM:
                use_important = True
                random_frames = True
                segment_min_avg_mass = self.segment_min_avg_mass
            elif segment_type == SegmentType.ALL_RANDOM:
                use_important = False
                random_frames = True
                segment_min_avg_mass = self.segment_min_avg_mass
            elif segment_type == SegmentType.IMPORTANT_SEQUENTIAL:
                use_important = True
                random_frames = False
            elif segment_type == SegmentType.ALL_SEQUENTIAL:
                use_important = False
                random_frames = False
                segment_min_avg_mass = self.segment_min_avg_mass
            elif segment_type == SegmentType.TOP_SEQUENTIAL:
                random_frames = False
                top_frames = True
            elif segment_type == SegmentType.ALL_RANDOM_SECTIONS:
                use_important = False
                random_frames = True
                segment_min_avg_mass = self.segment_min_avg_mass
                random_sections = True
            elif segment_type == SegmentType.ALL_RANDOM_NOMIN:
                use_important = False
                random_frames = False
                segment_min_avg_mass = None
            elif segment_type == SegmentType.TOP_RANDOM:
                use_important = False
                random_frames = True
                top_frames = True
            track.calculate_segments(
                segment_frame_spacing,
                segment_width,
                random_frames=random_frames,
                use_important=use_important,
                top_frames=top_frames,
                segment_min_mass=segment_min_avg_mass,
                random_sections=random_sections,
            )
            filtered_stats = filtered_stats + track.filtered_stats["segment_mass"]
            if len(track.segments) == 0:
                empty_tracks.append(track)
                continue
            for sample in track.segments:
                self.add_clip_sample_mappings(sample)

        self.rebuild_cdf()
        logging.info(
            "%s #segments %s filtered stats are %s took  %s",
            self.name,
            len(self.samples),
            filtered_stats,
            time.time() - start,
        )

    def remove_sample(self, sample):
        self.samples.remove(sample)
        self.samples_by_label[sample.label].remove(sample)

    def fetch_track(
        self,
        track: TrackHeader,
        original=False,
        sample_frames=False,
    ):
        """
        Fetches data for an entire track
        :param track: the track to fetch
        :return: segment data of shape [frames, channels, height, width]
        """
        frame_numbers = None
        if sample_frames:
            frame_numbers = [frame.frame_num for frame in track.sample_frames]
            frame_numbers.sort()
        frames = self.db.get_track(
            track.clip_id,
            track.track_id,
            original=original,
            frame_numbers=frame_numbers,
        )
        return frames
