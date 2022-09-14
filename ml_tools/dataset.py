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
from ml_tools.datasetstructures import TrackHeader, TrackingSample, SegmentType
from ml_tools.trackdatabase import TrackDatabase
from ml_tools import tools
from track.region import Region
import json


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
        use_predictions=False,
        consecutive_segments=False,
        labels=[],
    ):
        self.consecutive_segments = consecutive_segments
        # self.camera_bins = {}
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
        self.use_segments = True
        if config:
            self.type = config.train.type
            if config.train.type == "IR":
                self.use_segments = False
            else:
                self.use_segments = config.train.hyper_params.get("use_segments", True)
            self.segment_length = config.build.segment_length
            # number of seconds segments are spaced apart
            self.segment_spacing = config.build.segment_spacing
            self.banned_clips = config.build.banned_clips
            self.included_labels = config.labels
            self.segment_min_avg_mass = config.build.segment_min_avg_mass
            self.excluded_tags = config.load.excluded_tags
            self.min_frame_mass = config.build.min_frame_mass
            self.filter_by_lq = config.build.filter_by_lq
            self.segment_type = SegmentType.ALL_RANDOM

        else:
            self.filter_by_lq = True
            # number of seconds each segment should be
            self.segment_length = 25
            # number of seconds segments are spaced apart
            self.segment_spacing = 1
            self.segment_min_avg_mass = None
            self.min_frame_mass = 16
            self.segment_type = SegmentType.ALL_RANDOM
        self.filtered_stats = {
            "confidence": 0,
            "trap": 0,
            "banned": 0,
            "date": 0,
            "tags": 0,
            "segment_mass": 0,
            "no_data": 0,
            "not-confirmed": 0,
            "tag_names": set(),
            "notags": 0,
            "bad_track_json": 0,
        }
        self.lbl_p = None
        self.numpy_data = None

    def load_db(self):
        self.db = TrackDatabase(self.db_file)

    def set_read_only(self, read_only):
        if self.db is not None:
            self.db.set_read_only(read_only)

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
        samples_count = 0
        if self.label_mapping:
            for key, value in self.label_mapping.items():
                if key == label or value == label:
                    label_tracks = self.tracks_by_label.get(key, [])
                    tracks += len(label_tracks)
                    samples_count += len(self.samples_by_label.get(key, []))
                    # segments += sum(len(track.segments) for track in label_tracks)
                    # frames += sum(
                    #     len(track.get_sample_frames())
                    #     for track in label_tracks
                    #     if track.sample_frames is not None
                    # )

                if key == label or value == label:
                    label_tracks = []
                    segments = self.segments_by_label.get(key, [])
                    tracks += len(set([segment.track_id for segment in segments]))
                    segments_count += len(segments)
                    frames += sum([segment.frames for segment in segments])
        else:
            samples = self.samples_by_label.get(label, [])
            tracks = len(set([sample.track_id for sample in samples]))
            weight = self.get_label_weight(label)
            bins = len(set([sample.bin_id for sample in samples]))
            samples_count = len(samples)
        return samples_count, tracks, bins, weight

    def load_clips(
        self,
        shuffle=False,
        before_date=None,
        after_date=None,
        label=None,
    ):
        """
        Loads track headers from track database with optional filter
        :return: [number of tracks added, total tracks].
        """
        counter = 0
        clip_ids = self.db.get_all_clip_ids(before_date, after_date, label)

        if shuffle:
            np.random.shuffle(clip_ids)
        for clip_id in clip_ids:
            self.load_clip(clip_id)
            counter += 1
            if counter % 50 == 0:
                logging.debug("Dataset loaded %s / %s", counter, len(clip_ids))
        return [counter, len(clip_ids)]

    def load_clip(self, clip_id):
        clip_meta = self.db.get_clip_meta(clip_id)
        tracks = self.db.get_clip_tracks(clip_id)
        filtered = 0
        for track_meta in tracks:
            if self.filter_track(clip_meta, track_meta):
                filtered += 1
                continue
            track_header = TrackHeader.from_meta(clip_id, clip_meta, track_meta)
            if self.use_segments:
                segment_frame_spacing = int(
                    round(self.segment_spacing * track_header.frames_per_second)
                )
                segment_width = self.segment_length
                track_header.calculate_segments(
                    segment_frame_spacing,
                    segment_width,
                    self.segment_type,
                    self.segment_min_avg_mass,
                )
                self.filtered_stats["segment_mass"] += track_header.filtered_stats[
                    "segment_mass"
                ]
                for segment in track_header.segments:
                    self.add_clip_sample_mappings(segment)
            else:
                sample_frames = track_header.get_sample_frames()
                skip_x = None
                if self.type == "IR":
                    skip_last = int(len(sample_frames) * 0.1)
                    sample_frames = sample_frames[:-skip_last]
                for sample in sample_frames:
                    if not self.filter_by_lq or (
                        sample.mass >= track_header.lower_mass
                        and sample.mass <= track_header.upper_mass
                    ):
                        self.add_clip_sample_mappings(sample)
        return filtered

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

        if self.min_frame_mass and sample.mass < self.min_frame_mass:
            return True
        return False

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

    def filter_track(self, clip_meta, track_meta):
        # some clips are banned for various reasons4
        source = os.path.basename(clip_meta["filename"])
        if self.banned_clips and source in self.banned_clips:
            self.filtered_stats["banned"] += 1
            return True
        if "tag" not in track_meta:
            self.filtered_stats["notags"] += 1
            return True
        if track_meta["tag"] not in self.included_labels:
            self.filtered_stats["tags"] += 1
            self.filtered_stats["tag_names"].add(track_meta["tag"])
            return True
        track_tags = track_meta.get("track_tags")
        if track_tags is not None:
            try:
                track_tags = json.loads(track_tags)
            except:
                logging.error(
                    "Error loading track tags json for %s clip %s track %s",
                    track_tags,
                    clip_meta.get("id"),
                    track_meta.get("id"),
                )
                self.filtered_stats["bad_track_json"] += 1

                return True
            excluded_tags = [
                tag["what"]
                for tag in track_tags
                if not tag.get("automatic", False)
                and tag.get("what") in self.excluded_tags
            ]
            if len(excluded_tags) > 0:
                self.filtered_stats["tag_names"] |= set(excluded_tags)

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
            track.calculate_segments(
                segment_frame_spacing,
                segment_width,
                segment_type,
                segment_min_mass=segment_min_avg_mass,
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
