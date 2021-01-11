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

from ml_tools.datasetstructures import TrackHeader, SegmentHeader, Camera
from ml_tools.trackdatabase import TrackDatabase
from ml_tools.preprocess import preprocess_segment


class TrackChannels:
    """ Indexes to channels in track. """

    thermal = 0
    filtered = 1
    flow_h = 2
    flow_v = 3
    mask = 4


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
        track_db: TrackDatabase,
        name="Dataset",
        config=None,
        use_segments=True,
        use_predictions=False,
        consecutive_segments=False,
    ):
        self.consecutive_segments = consecutive_segments
        self.camera_bins = {}
        self.use_segments = use_segments
        # database holding track data
        self.db = track_db

        # name of this dataset
        self.name = name

        # list of our tracks
        self.tracks = []
        self.tracks_by_label = {}
        self.tracks_by_bin = {}
        self.tracks_by_id = {}
        self.camera_names = set()
        self.cameras_by_id = {}

        # writes the frame motion into the center of the optical flow channels
        self.encode_frame_offsets_in_flow = False
        self.segment_cdf = []
        self.segment_label_cdf = {}
        self.segments = []
        self.segments_by_label = {}
        self.segments_by_id = {}

        self.frame_cdf = []
        self.frame_label_cdf = {}

        self.frame_samples = []
        self.clips_to_samples = {}
        self.frames_by_label = {}

        # list of label names
        self.labels = []

        # dictionary used to apply label remapping during track load
        self.label_mapping = None

        # this allows manipulation of data (such as scaling) during the sampling stage.
        self.enable_augmentation = False
        # how often to scale during augmentation
        self.scale_frequency = 0.50

        self.preloader_queue = None
        self.preloader_threads = None
        self.preloader_stop_flag = False
        # a copy of our entire dataset, if loaded.
        self.X = None
        self.y = None

        if config:
            self.min_frame_mass = config.build.train_min_mass

            # number of seconds each segment should be
            self.segment_length = config.build.segment_length
            # number of seconds segments are spaced apart
            self.segment_spacing = config.build.segment_spacing
            self.banned_clips = config.build.banned_clips
            self.included_labels = config.labels
            self.clip_before_date = config.build.clip_end_date
            self.segment_min_mass = config.build.test_min_mass
        else:
            self.min_frame_mass = 16
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

    @property
    def rows(self):
        return len(self.segments)

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

    def next_batch(self, n, disable_async=False, force_no_augmentation=False):
        """
        Returns a batch of n segments (X, y) from dataset.
        Applies augmentation and preprocessing automatically.
        :param n: number of segments
        :param disable_async: forces fetching of segment in this thread / process rather than collecting from
            an aync reader queue (if one exists)
        :param force_no_augmentation: forces augmentation off, may disable asyc loading.
        :return: X of shape [n, channels, height, width], y (labels) of shape [n]
        """

        # if async is enabled use it.
        if (
            not disable_async
            and self.preloader_queue is not None
            and not force_no_augmentation
        ):
            # get samples from queue
            batch_X = []
            batch_y = []
            for _ in range(n):
                X, y = self.preloader_queue.get()
                batch_X.append(X[0])
                batch_y.append(y[0])

            return np.asarray(batch_X), np.asarray(batch_y)

        segments = [self.sample_segment() for _ in range(n)]

        batch_X = []
        batch_y = []

        for segment in segments:
            data = self.fetch_segment(
                segment, augment=self.enable_augmentation and not force_no_augmentation
            )
            batch_X.append(data)
            batch_y.append(self.labels.index(segment.label))

            if np.isnan(data).any():
                logging.warning("NaN found in data from source: %r", segment.clip_id)

        # Half float should be fine here.  When using process based async loading we have to pickle the batch between
        # processes, so having it half the size helps a lot.  Also it reduces the memory required for the read buffers
        batch_X = np.float16(batch_X)
        batch_y = np.int32(batch_y)

        return batch_X, batch_y

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
        groups = []
        groups.append((["bird"], "bird"))
        groups.append((["rodent"], "rodent"))
        self.regroup(groups)

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
        if track_header.bin_id in self.tracks_by_bin:
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
        predictions = self.db.get_track_predictions(clip_id, track_id)
        if self.filter_track(clip_meta, track_meta):
            return False
        track_header = TrackHeader.from_meta(
            clip_id, clip_meta, track_meta, predictions
        )
        self.tracks.append(track_header)
        frames = self.db.get_track(clip_id, track_id)
        track_header.set_important_frames(
            labels, self.min_frame_mass, frame_data=frames
        )
        segment_frame_spacing = int(
            round(self.segment_spacing * track_header.frames_per_second)
        )
        segment_width = int(round(self.segment_length * track_header.frames_per_second))
        if track_header.num_sample_frames > segment_width / 3.0:
            track_header.calculate_segments(
                track_meta["mass_history"],
                segment_frame_spacing,
                segment_width,
                self.segment_min_mass,
                use_important=not self.consecutive_segments,
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

        if track_header.label not in self.tracks_by_label:
            self.labels.append(track_header.label)
            self.tracks_by_label[track_header.label] = []
        self.tracks_by_label[track_header.label].append(track_header)
        segs = self.segments_by_label.setdefault(track_header.label, [])
        segs.extend(track_header.segments)
        for seg in segs:
            self.segments_by_id[seg.id] = seg
        frames = self.frames_by_label.setdefault(track_header.label, [])
        samples = track_header.get_sample_frames()
        self.frame_samples.extend(samples)
        frames.extend(samples)
        camera = self.cameras_by_id.setdefault(
            track_header.camera_id, Camera(track_header.camera_id)
        )
        self.camera_names.add(track_header.camera_id)
        camera.add_track(track_header)

    def fetch_all(self):
        """
        Fetches all segments
        :return: X of shape [n,f,channels,height,width], y of shape [n]
        """
        X = np.float32([self.fetch_segment(segment) for segment in self.segments])
        y = np.int32([self.labels.index(segment.label) for segment in self.segments])
        return X, y

    def fetch_track(self, track: TrackHeader):
        """
        Fetches data for an entire track
        :param track: the track to fetch
        :return: segment data of shape [frames, channels, height, width]
        """
        data = self.db.get_track(track.clip_id, track.track_number, 0, track.frames)
        data = preprocess_segment(
            data,
            reference_level=track.frame_temp_median,
            frame_velocity=track.frame_velocity,
            encode_frame_offsets_in_flow=self.encode_frame_offsets_in_flow,
            default_inset=self.DEFAULT_INSET,
        )
        return data

    def fetch_segment(self, segment: SegmentHeader, augment=False):
        """
        Fetches data for segment.
        :param segment: The segment header to fetch
        :param augment: if true applies data augmentation
        :return: segment data of shape [frames, channels, height, width]
        """
        segment_width = self.segment_length * segment.track.frames_per_second
        # if we are requesting a segment smaller than the default segment size take it from the middle.
        unused_frames = segment.frames - segment_width
        if unused_frames < 0:
            raise Exception(
                "Maximum segment size for the dataset is {} frames, but requested {}".format(
                    segment.frames, segment_width
                )
            )
        first_frame = segment.start_frame + (unused_frames // 2)
        last_frame = segment.start_frame + (unused_frames // 2) + segment_width

        if augment and unused_frames > 0:
            # jitter first frame
            prev_frames = first_frame
            post_frames = segment.track.frames - 1 - last_frame
            max_jitter = max(5, unused_frames)
            jitter = np.clip(
                np.random.randint(-max_jitter, max_jitter), -prev_frames, post_frames
            )
        else:
            jitter = 0
        first_frame += jitter
        last_frame += jitter

        data = self.db.get_track(
            segment.clip_id, segment.track_number, first_frame, last_frame
        )

        if len(data) != segment_width:
            logging.error(
                "invalid segment length %d, expected %d", len(data), len(segment_width)
            )

        data = preprocess_segment(
            data,
            segment.track.frame_temp_median[first_frame:last_frame],
            segment.track.frame_velocity[first_frame:last_frame],
            augment=augment,
            default_inset=self.DEFAULT_INSET,
        )

        return data

    def sample_segment(self):
        """ Returns a random segment from weighted list. """
        if not self.segments:
            return None
        roll = random.random()
        index = bisect(self.segment_cdf, roll)
        return self.segments[index]

    def load_all(self, force=False):
        """ Loads all X and y into dataset if required. """
        if self.X is None or force:
            self.X, self.y = self.fetch_all()

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
        """ Removes any segments from track_headers where the segment has been deleted """
        segment_set = set(self.segments)

        # remove segments from tracks
        for track in self.tracks:
            segments = track.segments
            segments = [segment for segment in segments if (segment in segment_set)]
            track.segments = segments

    def get_normalisation_constants(self, n=None):
        """
        Gets constants required for normalisation from dataset.  If n is specified uses a random sample of n segments.
        Segment weight is not taken into account during this sampling.  Otherrwise the entire dataset is used.
        :param n: If specified calculates constants from n samples
        :return: normalisation constants
        """

        # note:
        # we calculate the standard deviation and mean using the moments as this allows the calculation to be
        # done piece at a time.  Otherwise we'd need to load the entire dataset into memory, which might not be
        # possiable.

        if len(self.segments) == 0:
            raise Exception("No segments in dataset.")

        sample = (
            self.segments
            if n is None or n >= len(self.segments)
            else random.sample(self.segments, n)
        )

        # fetch a sample to see what the dims are
        example = self.fetch_segment(self.segments[0])
        _, channels, height, width = example.shape

        # we use float64 as this accumulator will get very large!
        first_moment = np.zeros((channels, height, width), dtype=np.float64)
        second_moment = np.zeros((channels, height, width), dtype=np.float64)

        for segment in sample:
            data = np.float64(self.fetch_segment(segment))
            first_moment += np.mean(data, axis=0)
            second_moment += np.mean(np.square(data), axis=0)

        # reduce down to channel only moments, in the future per pixel normalisation would be a good idea.
        first_moment = np.sum(first_moment, axis=(1, 2)) / (
            len(sample) * width * height
        )
        second_moment = np.sum(second_moment, axis=(1, 2)) / (
            len(sample) * width * height
        )

        mu = first_moment
        var = second_moment + (mu ** 2) - (2 * mu * first_moment)

        normalisation_constants = [(mu[i], math.sqrt(var[i])) for i in range(channels)]

        return normalisation_constants

    def rebuild_cdf(self, lbl_p=None):
        """Calculates the CDF used for fast random sampling for frames and
        segments, if balance labels is set each label has an equal chance of
        being chosen
        """

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
        """ Calculates the CDF used for fast random sampling """
        self.segment_cdf = []
        total = 0
        self.segment_label_cdf = {}
        for segment in self.segments:
            seg_weight = segment.weight
            if lbl_p and segment.track.label in lbl_p:
                seg_weight *= lbl_p[segment.track.label]
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
        """ Returns the total weight for all segments of given label. """
        tracks = self.tracks_by_label.get(label)
        return sum(track.weight for track in tracks) if tracks else 0

    def get_label_segments_count(self, label):
        """ Returns the total weight for all segments of given class. """
        tracks = self.tracks_by_label.get(label, [])
        result = sum([len(track.segments) for track in tracks])
        return result

    def get_label_segments(self, label):
        """ Returns the total weight for all segments of given class. """
        result = []
        for track in self.tracks_by_label.get(label, []):
            result.extend(track.segments)
        return result

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
        new_labels = []
        tracks_by_bin = {}
        samples = []
        for g in groups:
            new_labels.append(g[1])
            count = 0
            for label in g[0]:
                lbl_samples = self.samples_for(label)
                count += len(lbl_samples)
                self.label_mapping[label] = g[1]
                samples.extend(lbl_samples)
                for sample in lbl_samples:
                    track = self.tracks_by_id[sample.unique_track_id]
                    tracks_by_bin[track.bin_id] = track
            counts.append(count)

        self.labels = new_labels
        self.tracks_by_bin = tracks_by_bin
        self.set_samples(samples)
        if self.use_segments:
            self.segments_by_id == {}
            for seg in samples:
                self.segments_by_id[seg.id] = seg

            if shuffle:
                np.random.shuffle(self.segments)
        elif shuffle:
            np.random.shuffle(self.frame_samples)
        self.rebuild_cdf()

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

    @property
    def sample_count(self):
        return len(self.samples())

    def start_async_load(self, buffer_size=128):
        """
        Starts async load process.
        """

        # threading has limitations due to global lock
        # but processor ends up slow on windows as the numpy array needs to be pickled across processes which is
        # 2ms per process..
        # this could be solved either by using linux (with forking, which is copy on write) or with a shared ctype
        # array.

        if self.PROCESS_BASED:
            self.preloader_queue = multiprocessing.Queue(buffer_size)
            self.preloader_threads = [
                multiprocessing.Process(
                    target=preloader, args=(self.preloader_queue, self)
                )
                for _ in range(self.WORKER_THREADS)
            ]
        else:
            self.preloader_queue = queue.Queue(buffer_size)
            self.preloader_threads = [
                threading.Thread(target=preloader, args=(self.preloader_queue, self))
                for _ in range(self.WORKER_THREADS)
            ]

        self.preloader_stop_flag = False
        for thread in self.preloader_threads:
            thread.start()

    def stop_async_load(self):
        """
        Stops async worker thread.
        """
        if self.preloader_threads is not None:
            for thread in self.preloader_threads:
                if hasattr(thread, "terminate"):
                    # note this will corrupt the queue, so reset it
                    thread.terminate()
                    self.preloader_queue = None
                else:
                    thread.exit()


# continue to read examples until queue is full
def preloader(q, dataset):
    """ add a segment into buffer """
    logging.info(
        " -started async fetcher for %s with augment=%s segment_width=%s",
        dataset.name,
        dataset.enable_augmentation,
        dataset.segment_width,
    )
    loads = 0
    timer = time.time()
    while not dataset.preloader_stop_flag:
        if not q.full():
            q.put(dataset.next_batch(1, disable_async=True))
            loads += 1
            if (time.time() - timer) > 1.0:
                # logging.debug("{} segments per seconds {:.1f}".format(dataset.name, loads / (time.time() - timer)))
                loads = 0
        else:
            time.sleep(0.1)


def dataset_db_path(config):
    return os.path.join(config.tracks_folder, "datasets.dat")
