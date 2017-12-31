"""
Author: Matthew Aitchison
Date: December 2017

Dataset used for training a tensorflow model from track data.

Tracks are broken into segments.  Filtered, and then passed to the trainer using a weighted random sample.

"""

import os

import numpy as np

from ml_tools.trackdatabase import TrackDatabase


class SegmentHeader():
    """ Header for segment. """

    def __init__(self, clip_id, track_id, offset, weight, tag, average_mass):
        # reference to clip this segment came from
        self.clip_id = clip_id
        # reference to track this segment came from
        self.track_id = track_id
        # first frame of this segment
        self.offset = offset
        # relative weight of the segment (higher is sampled more often)
        self.weight = weight
        self.tag = tag.lower()
        self.average_mass = average_mass

    @property
    def name(self):
        """ Unique name of this segment. """
        return self.clip_id + '-'+self.track_id+'-'+str(self.offset)

    def __str__(self):
        return "offset:{0} weight:{1:.1f}".format(self.offset, self.weight)


class TrackHeader():
    """ Header for track. """

    def __init__(self, clip_id, track_id):
        # reference to clip this segment came from
        self.clip_id = clip_id
        # reference to track this segment came from
        self.track_id = track_id
        # list of segments that belong to this track
        self.segments = []

    @property
    def name(self):
        """ Unique name of this track. """
        return TrackHeader.get_name(self.clip_id, self.track_id)

    @staticmethod
    def get_name(track_id, clip_id):
        return clip_id + '-' + track_id


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

        # tracks by name
        self.tracks = {}
        # segments by name
        self.segments = {}

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
        for clip_id, track_id in self.db.get_all_track_ids():
            if self.add_track(clip_id, track_id, track_filter):
                counter += 1
        return counter

    def add_track(self, clip_id, track_id, track_filter=None):
        """
        Creates segments for track and adds them to the dataset
        :param track_filter: if provided a function filter(clip_meta, track_meta) that returns true when a track should be ignored)
        :return: True if track was added, false if it was filtered out.
        """

        # make sure we don't already have this track
        if TrackHeader.get_name(clip_id, track_id) in self.tracks:
            return

        clip_meta = self.db.get_clip_meta(clip_id)
        track_meta = self.db.get_track_meta(clip_id, track_id)
        if track_filter and track_filter(clip_meta, track_meta):
            return False

        track_header = TrackHeader(clip_id, track_id)
        self.tracks[track_header.name] = track_header

        # scan through track looking for good segments to add to our datset
        mass_history = track_meta['mass_history']
        for i in range(len(mass_history) // self.segment_spacing):
            segment_start = i * self.segment_spacing
            mass_slice = mass_history[segment_start:segment_start + self.segment_width + 1]
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
                clip_id=clip_id, track_id=track_id, offset=segment_start,
                weight=1, tag=track_meta['tag'], average_mass=segment_avg_mass)

            self.segments[segment.name] = segment
            track_header.segments.append(segment)

        return True


    def fetch_segment(self, segment: SegmentHeader):
        """ Fetches data for segment"""
        pass


    def sample_segment(self):
        """ Returns a random segment from weighted list. """
        pass

