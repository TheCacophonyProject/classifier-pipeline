"""
Author: Matthew Aitchison
Date: December 2017

Dataset used for training a tensorflow model from track data.

Tracks are broken into segments.  Filtered, and then passed to the trainer using a weighted random sample.

"""

import os
import datetime
from dateutil import parser

import numpy as np

from ml_tools.trackdatabase import TrackDatabase

class SegmentHeader():
    """ Header for segment. """

    def __init__(self, clip_id, track_number, offset, weight, label, avg_mass):
        # reference to clip this segment came from
        self.clip_id = clip_id
        # reference to track this segment came from
        self.track_number = track_number
        # first frame of this segment
        self.offset = offset
        # relative weight of the segment (higher is sampled more often)
        self.weight = weight
        # our label
        self.label = label.lower()
        self.avg_mass = avg_mass


    @property
    def name(self):
        """ Unique name of this segment. """
        return self.clip_id + '-' + self.track_number + '-' + str(self.offset)

    def __str__(self):
        return "offset:{0} weight:{1:.1f}".format(self.offset, self.weight)


class TrackHeader():
    """ Header for track. """

    def __init__(self, clip_id, track_number, label, start_time, duration, camera):
        # reference to clip this segment came from
        self.clip_id = clip_id
        # reference to track this segment came from
        self.track_number = track_number
        # list of segments that belong to this track
        self.segments = []
        # label for this track
        self.label = label
        # date and time of the start of the track
        self.start_time = start_time
        # duration in seconds
        self.duration = duration
        # camera this track came from
        self.camera = camera
        # name of the bin to assign this track to.
        self.bin = str(start_time.date())+'-'+str(camera)+'-'+label

    @property
    def name(self):
        """ Unique name of this track. """
        return TrackHeader.get_name(self.clip_id, self.track_number)

    @property
    def weight(self):
        """ Returns total weight for all segments in this track"""
        return sum(segment.weight for segment in self.segments)

    @staticmethod
    def get_name(clip_id, track_number):
        return str(clip_id) + '-' + str(track_number)

    @staticmethod
    def from_meta(clip_id, track_meta):
        """ Creates a track header from given metadata. """
        # kind of chacky way to get camera name from clip_id, in the future camera will be included in the metadata.
        camera = os.path.splitext(os.path.basename(clip_id))[0].split('-')[-1]
        result = TrackHeader(
            clip_id=clip_id, track_number=track_meta['id'], label=track_meta['tag'],
            start_time=parser.parse(track_meta['start_time']),
            duration=track_meta['duration'],
            camera=camera
        )
        return result


class Dataset():
    """
    Stores visit, clip, track, and segment information headers in memory, and allows track / segment streaming from
    disk.
    """

    def __init__(self, track_db: TrackDatabase, name="Dataset"):

        # database holding track data
        self.db = track_db

        # name of this dataset
        self.name = name

        # list of our tracks
        self.tracks = []
        self.track_by_name = {}
        self.tracks_by_label = {}
        self.tracks_by_bin = {}

        # cumulative probability distribution for segments.  Allows for super fast weighted random sampling.
        self.segment_cdf = []

        # segments list
        self.segments = []

        # list of label names
        self.labels = []

        # number of frames each segment should be
        self.segment_width = 27
        # number of frames segments are spaced apart
        self.segment_spacing = 9
        # minimum mass of a segment frame for it to be included
        self.segment_min_mass = None
        # minimum average frame mass for segment to be included
        self.segment_avg_mass = None

    def next_batch(self, n):
        """
        Returns a batch of n segments (X, y) from dataset.
        Applies augmentation and normalisation automatically.
        :param n: number of segment
        :return: numpy array of shape [n, channels, height, width]
        """

        segments = [self.sample_segment() for _ in range(n)]
        for segment in segments:
            data = self.fetch_segment(segment)

    def load_tracks(self, track_filter=None):
        """
        Loads track headers from track database with optional filter
        :return: number of tracks added.
        """
        counter = 0
        for clip_id, track_number in self.db.get_all_track_ids():
            if self.add_track(clip_id, track_number, track_filter):
                counter += 1
        return counter

    def add_tracks(self, tracks, track_filter=None):
        """
        Adds list of tracks to dataset
        :param tracks: list of TrackHeader
        :param track_filter: optional filter
        """
        for track in tracks:
            self.add_track(track.clip_id, track.track_number, track_filter)

    def add_track(self, clip_id, track_number, track_filter=None):
        """
        Creates segments for track and adds them to the dataset
        :param track_filter: if provided a function filter(clip_meta, track_meta) that returns true when a track should be ignored)
        :return: True if track was added, false if it was filtered out.
        """

        # make sure we don't already have this track
        if TrackHeader.get_name(clip_id, track_number) in self.tracks:
            return

        clip_meta = self.db.get_clip_meta(clip_id)
        track_meta = self.db.get_track_meta(clip_id, track_number)
        if track_filter and track_filter(clip_meta, track_meta):
            return False

        track_header = TrackHeader.from_meta(clip_id, track_meta)

        self.tracks.append(track_header)
        self.track_by_name[track_header.name] = track_header

        if track_header.label not in self.tracks_by_label:
            self.tracks_by_label[track_header.label] = []
        self.tracks_by_label[track_header.label].append(track_header)

        if track_header.bin not in self.tracks_by_bin:
            self.tracks_by_bin[track_header.bin] = []
        self.tracks_by_bin[track_header.bin].append(track_header)

        # scan through track looking for good segments to add to our datset
        mass_history = track_meta['mass_history']
        for i in range(len(mass_history) // self.segment_spacing):
            segment_start = i * self.segment_spacing
            mass_slice = mass_history[segment_start:segment_start + self.segment_width]
            segment_min_mass = np.min(mass_slice)
            segment_avg_mass = np.median(mass_slice)
            segment_frames = len(mass_slice)

            if segment_frames != self.segment_width:
                continue

            if self.segment_min_mass and segment_min_mass < self.segment_min_mass:
                continue

            if self.segment_avg_mass and segment_avg_mass < self.segment_avg_mass:
                continue

            segment = SegmentHeader(
                clip_id=clip_id, track_number=track_number, offset=segment_start,
                weight=1.0, label=track_meta['tag'], avg_mass=segment_avg_mass)

            self.segments.append(segment)
            track_header.segments.append(segment)

        return True

    def filter_segments(self, avg_mass):
        """
        Removes any segments with an average mass less than the given avg_mass
        :param avg_mass: segments with less avarage mass per frame than this will be removed from the dataset.
        :return: number of segments removed
        """

        filtered = 0
        new_segments = []

        for segment in self.segments:
            if segment.avg_mass >= avg_mass:
                new_segments.append(segment)
            else:
                filtered += 1

        self.segments = new_segments

        self._purge_track_segments()

        return filtered



    def fetch_segment(self, segment: SegmentHeader):
        """ Fetches data for segment"""
        pass


    def sample_segment(self):
        """ Returns a random segment from weighted list. """
        pass

    def balance_weights(self, weight_modifiers=None):
        """
        Adjusts weights so that every class is evenly represented.
        :param weight_modifiers: if specified is a dictionary mapping from label to weight modifier,
            where < 1 sampled less frequently, and > 1 is sampled more frequently.
        :return:
        """

        class_weight = {}
        mean_class_weight = 0

        for class_name in self.labels:
            class_weight[class_name] = self.get_class_weight(class_name)
            mean_class_weight += class_weight[class_name] / len(self.labels)

        scale_factor = {}
        for class_name in self.labels:
            modifier = 1.0 if weight_modifiers is None else weight_modifiers.get(class_name, 1.0)
            if class_weight[class_name] == 0:
                scale_factor[class_name] = 1.0
            else:
                scale_factor[class_name] = mean_class_weight / class_weight[class_name] * modifier

        for segment in self.segments:
            segment.weight *= scale_factor.get(segment.label, 1.0)

        self.rebuild_cdf()

    def balance_bins(self, max_bin_weight):
        """
        Adjusts weights so that bins with a number number of segments aren't sampled so frequently.
        :param max_bin_weight: bins with more weight than this number will be scaled back to this weight.
        """

        for bin_name, tracks in self.tracks_by_bin.items():
            bin_weight = sum(track.weight for track in tracks)
            if bin_weight > max_bin_weight:
                scale_factor =  max_bin_weight / bin_weight
                for track in tracks:
                    for segment in track.segments:
                        segment.weight *= scale_factor

        self.rebuild_cdf()


    def balance_resample(self, max_ratio=2.0, weight_modifiers=None):
        """ Removes segments until all classes are with given ratio """

        class_count = {}
        min_class_count = 9999999

        for class_name in self.labels:
            class_count[class_name] = self.get_class_segments_count(class_name)
            modifier = 1.0 if weight_modifiers is None else weight_modifiers.get(class_name, 1.0)
            if modifier > 0:
                min_class_count = min(min_class_count, class_count[class_name] / modifier)

        max_class_count = int(min_class_count * max_ratio)

        new_segments = []

        for class_name in self.labels:
            segments = self.get_class_segments(class_name)
            if len(segments) > max_class_count:
                # resample down
                modifier = 1.0 if weight_modifiers is None else weight_modifiers.get(class_name, 1.0)
                segments = np.random.choice(segments, int(max_class_count * modifier), replace=False).tolist()
            new_segments += segments

        self.segments = new_segments

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

    def rebuild_cdf(self):
        """ Calculates the CDF used for fast random sampling """
        self.segment_cdf = []
        prob = 0
        for segment in self.segments:
            prob += segment.weight
            self.segment_cdf.append(prob)
        normalizer = self.segment_cdf[-1]
        self.segment_cdf = [x / normalizer for x in self.segment_cdf]

    def get_class_weight(self, label):
        """ Returns the total weight for all segments of given label. """
        return sum(segment.weight for segment in self.segments if segment.label == label)

    def get_class_segments_count(self, label):
        """ Returns the total weight for all segments of given class. """
        return len(self.get_class_segments(label))

    def get_class_segments(self, label):
        """ Returns the total weight for all segments of given class. """
        return [segment for segment in self.segments if segment.label == label]



