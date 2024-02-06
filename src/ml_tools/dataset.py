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
from ml_tools.rawdb import RawDatabase
from ml_tools import tools
from track.region import Region
import json
from config.loadconfig import LoadConfig
from pathlib import Path


class Dataset:
    """
    Stores visit, clip, track, and segment information headers in memory, and allows track / segment streaming from
    disk.
    """

    def __init__(
        self,
        dataset_dir,
        name="Dataset",
        config=None,
        use_predictions=False,
        consecutive_segments=False,
        labels=[],
        label_mapping=None,
        raw=True,
        ext=".cptv",
    ):
        self.ext = ext
        self.raw = raw
        # if self.raw and self.ext == ".cptv":
        # raise Exception("Raw CPTV not implemented yet")
        self.dataset_dir = Path(dataset_dir)
        self.consecutive_segments = consecutive_segments
        self.label_mapping = label_mapping
        # name of this dataset
        self.name = name
        # list of our tracks
        self.samples_by_bin = {}
        # self.samples = []
        self.samples_by_id = {}
        self.clips = []

        # list of label names
        self.labels = labels

        self.enable_augmentation = False
        self.label_caps = {}
        self.use_segments = True
        if config:
            self.tag_precedence = config.load.tag_precedence
            self.type = config.train.type
            if config.train.type == "IR":
                self.use_segments = False
                self.segment_length = 1
            else:
                self.use_segments = config.train.hyper_params.get("use_segments", True)
                if self.use_segments:
                    self.segment_length = config.build.segment_length
                else:
                    self.segment_length = 1
            # number of seconds segments are spaced apart
            self.segment_spacing = config.build.segment_spacing
            self.banned_clips = config.build.banned_clips
            self.included_labels = config.labels
            self.segment_min_avg_mass = config.build.segment_min_avg_mass
            self.excluded_tags = config.load.excluded_tags
            self.min_frame_mass = config.build.min_frame_mass
            self.filter_by_lq = config.build.filter_by_lq
            self.segment_type = SegmentType.ALL_RANDOM
            self.max_segments = config.build.max_segments
        else:
            self.tag_precedence = LoadConfig.DEFAULT_GROUPS
            self.filter_by_lq = False
            # number of seconds each segment should be
            if self.use_segments:
                self.segment_length = 25
            else:
                self.segment_length = 1
            # number of seconds segments are spaced apart
            self.segment_spacing = 1
            self.segment_min_avg_mass = 10
            self.min_frame_mass = 16
            self.segment_type = SegmentType.ALL_RANDOM
        self.max_frame_mass = None
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

        self.skip_ffc = True

    @property
    def samples(self):
        return self.samples_by_id.values()

    @property
    def sample_count(self):
        return len(self.samples_by_id)

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
        dont_filter_segment=False,
    ):
        """
        Loads track headers from track database with optional filter
        :return: [number of tracks added, total tracks].
        """

        counter = 0
        logging.info("Loading clips")
        for db_clip in self.dataset_dir.glob(f"**/*{self.ext}"):
            tracks_added = self.load_clip(db_clip, dont_filter_segment)
            if tracks_added == 0:
                logging.info("No tracks added for %s", db_clip)
            counter += 1
            if counter % 50 == 0:
                logging.debug("Dataset loaded %s", counter)
        return [counter, counter]

    def load_clip(self, db_clip, dont_filter_segment=False):
        if self.raw:
            db = RawDatabase(db_clip)
        else:
            db = TrackDatabase(db_clip)
        try:
            clip_header = db.get_clip_tracks(self.tag_precedence)
        except:
            logging.error("Could not load %s", db_clip, exc_info=True)
            return 0
        if clip_header is None or filter_clip(clip_header):
            return 0
        filtered = 0
        added = 0
        clip_header.tracks = [
            track
            for track in clip_header.tracks
            if not filter_track(track, self.excluded_tags, self.filtered_stats)
        ]
        self.clips.append(clip_header)
        for track_header in clip_header.tracks:
            if self.label_mapping:
                track_header.remapped_label = self.label_mapping.get(
                    track_header.original_label, track_header.original_label
                )
            added += 1
            if self.use_segments:
                segment_frame_spacing = int(
                    round(self.segment_spacing * clip_header.frames_per_second)
                )
                segment_width = self.segment_length
                track_header.get_segments(
                    segment_width,
                    segment_frame_spacing,
                    self.segment_type,
                    self.segment_min_avg_mass,
                    max_segments=self.max_segments,
                    dont_filter=dont_filter_segment,
                    skip_ffc=self.skip_ffc,
                    ffc_frames=clip_header.ffc_frames,
                )
                self.filtered_stats["segment_mass"] += track_header.filtered_stats[
                    "segment_mass"
                ]

            else:
                skip_last = None
                if self.type == "IR":
                    # a lot of ir clips have bad tracking near end so just reduce track length
                    skip_last = 0.1
                track_header.calculate_sample_frames(
                    min_mass=(
                        self.min_frame_mass
                        if not self.filter_by_lq
                        else track_header.lower_mass
                    ),
                    max_mass=(
                        self.max_frame_mass
                        if not self.filter_by_lq
                        else track_header.upper_mass
                    ),
                    ffc_frames=clip_header.ffc_frames,
                    skip_last=skip_last,
                )
                if track_header.label not in self.labels:
                    self.labels.append(track_header.label)
            for sample in track_header.samples:
                self.add_clip_sample_mappings(sample)
        return added

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

    @property
    def samples_by_label(self):
        samples_by_label = {}
        for sample in self.samples:
            if sample.label not in samples_by_label:
                samples_by_label[sample.label] = []
            samples_by_label[sample.label].append(sample)
        return samples_by_label

    def add_clip_sample_mappings(self, sample):
        self.samples_by_id[sample.id] = sample
        # self.samples[sample.id] = sample

        if self.label_mapping:
            sample.remapped_label = self.label_mapping.get(
                sample.original_label, sample.original_label
            )

        if sample.label not in self.labels:
            self.labels.append(sample.label)
        bins = self.samples_by_bin.setdefault(sample.bin_id, {})
        bins[sample.id] = sample
        # (sample)
        return True

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

    def get_samples_by_source(self):
        samples_by_source = {}
        for s in self.samples_by_id.values():
            samples_by_source.setdefault(s.source_file, []).append(s)
        return samples_by_source

    #
    # def get_sample(self, cap=None, replace=True, label=None, random=True):
    #     """Returns a random frames from weighted list."""
    #     if label:
    #         samples = self.samples_for(label, remapped=True)
    #         cdf = self.label_cdf(label)
    #     else:
    #         samples = self.samples()
    #         cdf = self.cdf()
    #     if not samples:
    #         return None
    #     if cap is None:
    #         return samples
    #     if random:
    #         return np.random.choice(samples, cap, replace=replace, p=cdf)
    #     else:
    #         cap = min(cap, len(samples))
    #         return samples[:cap]
    #
    # def balance_bins(self, max_bin_weight=None):
    #     """
    #     Adjusts weights so that bins with a number number of segments aren't sampled so frequently.
    #     :param max_bin_weight: bins with more weight than this number will be scaled back to this weight.
    #     """
    #
    #     for bin_name, samples in self.samples_by_bin.items():
    #         bin_weight = sum(sample.sample_weight for sample in samples)
    #         if bin_weight == 0:
    #             continue
    #         if max_bin_weight is None:
    #             scale_factor = 1 / bin_weight
    #             # means each bin has equal possiblity
    #         elif bin_weight > max_bin_weight:
    #             scale_factor = max_bin_weight / bin_weight
    #         else:
    #             scale_factor = 1
    #         for sample in samples:
    #             sample.weight = np.float16(sample.sample_weight * scale_factor)
    #     self.rebuild_cdf()

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

    #
    # def rebuild_cdf(self, lbl_p=None):
    #     """Calculates the CDF used for fast random sampling for frames and
    #     segments, if balance labels is set each label has an equal chance of
    #     being chosen
    #     """
    #     if lbl_p is None:
    #         lbl_p = self.lbl_p
    #
    #     self.sample_cdf = []
    #     total = 0
    #     self.sample_label_cdf = {}
    #
    #     for sample in self.samples:
    #         sample_weight = sample.sample_weight
    #         if lbl_p and sample.label in lbl_p:
    #             sample_weight *= lbl_p[sample.label]
    #         total += sample_weight
    #
    #         self.sample_cdf.append(sample_weight)
    #
    #         cdf = self.sample_label_cdf.setdefault(sample.label, [])
    #         cdf.append(sample.sample_weight)
    #
    #     if len(self.sample_cdf) > 0:
    #         self.sample_cdf = [x / total for x in self.sample_cdf]
    #
    #     for key, cdf in self.sample_label_cdf.items():
    #         total = sum(cdf)
    #         self.sample_label_cdf[key] = [x / total for x in cdf]
    #
    #     if self.label_mapping:
    #         mapped_cdf = {}
    #         labels = list(self.label_mapping.keys())
    #         labels.sort()
    #         for label in labels:
    #             if label not in self.sample_label_cdf:
    #                 continue
    #             label_cdf = self.sample_label_cdf[label]
    #             new_label = self.label_mapping[label]
    #             if lbl_p and label in lbl_p:
    #                 label_cdf = np.float64(label_cdf)
    #                 label_cdf *= lbl_p[label]
    #             cdf = mapped_cdf.setdefault(new_label, [])
    #             cdf.extend(label_cdf)
    #
    #         for key, cdf in mapped_cdf.items():
    #             total = sum(cdf)
    #             mapped_cdf[key] = [x / total for x in cdf]
    #         self.sample_label_cdf = mapped_cdf

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
        self.label_mapping = groups
        samples_by_bin = {}
        samples_by_label = {}

        for lbl, remapped_lbl in groups.items():
            count = 0
            samples = self.samples_by_label[lbl]
            for s in samples:
                s.remapped_label = remapped_lbl
                samples_by_bin.setdefault(s.bin_id, {})[s.id] = s
                # .append(s)
            samples_by_label.setdefault(remapped_lbl, []).extend(samples)

        self.labels = list(groups.keys())
        self.labels.sort()
        self.samples_by_bin = samples_by_bin
        self.samples_by_label = samples_by_label
        #
        # if shuffle:
        #     np.random.shuffle(self.samples)
        # self.rebuild_cdf()

    def has_data(self):
        return len(self.samples_by_id) > 0

    #
    # def recalculate_segments(self, segment_type=SegmentType.ALL_RANDOM):
    #     self.samples_by_bin.clear()
    #     self.samples_by_label.clear()
    #     del self.samples[:]
    #     del self.samples
    #     self.samples = []
    #     self.samples_by_label = {}
    #     self.samples_by_bin = {}
    #     logging.info("%s generating segments  type %s", self.name, segment_type)
    #     start = time.time()
    #     empty_tracks = []
    #     filtered_stats = 0
    #
    #     for track in self.tracks:
    #         segment_frame_spacing = int(
    #             round(self.segment_spacing * track.frames_per_second)
    #         )
    #         segment_width = self.segment_length
    #         track.calculate_segments(
    #             segment_frame_spacing,
    #             segment_width,
    #             segment_type,
    #             segment_min_mass=segment_min_avg_mass,
    #         )
    #         filtered_stats = filtered_stats + track.filtered_stats["segment_mass"]
    #         if len(track.segments) == 0:
    #             empty_tracks.append(track)
    #             continue
    #         for sample in track.segments:
    #             self.add_clip_sample_mappings(sample)
    #
    #     self.rebuild_cdf()
    #     logging.info(
    #         "%s #segments %s filtered stats are %s took  %s",
    #         self.name,
    #         len(self.samples),
    #         filtered_stats,
    #         time.time() - start,
    #     )
    def remove_sample_by_id(self, id, bin_id):
        del self.samples_by_id[id]
        try:
            # not nessesarily there if splitting by clip hack
            del self.samples_by_bin[bin_id][id]
        except:
            pass

    def remove_sample(self, sample):
        del self.samples_by_id[sample.id]
        try:
            # not nessesarily there if splitting by clip hack
            del self.samples_by_bin[sample.bin_id][sample.id]
        except:
            pass

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


def filter_track(track_header, excluded_tags, filtered_stats={}):
    # some clips are banned for various reasons4
    if track_header.label is None:
        filtered_stats.setdefault("notags", 0)
        filtered_stats["notags"] += 1
        return True

    if track_header.label in excluded_tags:
        filtered_stats.setdefault("tags", 0)

        filtered_stats["tags"] += 1
        filter_tags = filtered_stats.setdefault("tag_names", set())
        filter_tags.add(track_header.label)

        return True

    if track_header.human_tags is not None:
        found_tags = [tag for tag in track_header.human_tags if tag in excluded_tags]
        if len(found_tags) > 0:
            filter_tags = filtered_stats.setdefault("tag_names", set())
            filter_tags |= set(found_tags)
            filtered_stats.setdefault("tags", 0)
            filtered_stats["tags"] += 1
            return True
    # always let the false-positives through as we need them even though they would normally
    # be filtered out.

    if track_header.regions_by_frame is None or len(track_header.regions_by_frame) == 0:
        filtered_stats.setdefault("no_data", 0)
        filtered_stats["no_data"] += 1
        logging.info("No region data")
        return True

    # dont think we need this gp 28/08/2023
    # if track_meta["human_tag"] == "false-positive":
    # return False

    # for some reason we get some records with a None confidence?
    if track_header.confidence <= 0.6:
        filtered_stats.setdefault("confidence", 0)
        filtered_stats["confidence"] += 1
        return True

    return False


def filter_clip(clip, filtered_stats={}):
    # remove tracks of trapped animals
    if (clip.events is not None and "trap" in clip.events.lower()) or (
        clip.trap is not None and "trap" in clip.trap.lower()
    ):
        self.filtered_stats["trap"] += 1
        logging.info("Filtered because in trap")
        return True
    return False
