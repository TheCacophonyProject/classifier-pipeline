"""
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import datetime
import numpy as np
from collections import namedtuple


from ml_tools.tools import Rectangle
from ml_tools.dataset import TrackChannels
import track.region
from track.region import Region


class Track:
    """ Bounds of a tracked object over time. """

    # keeps track of which id number we are up to.
    _track_id = 1

    def __init__(self, clip_id, id=None):
        """
        Creates a new Track.
        :param id: id number for track, if not specified is provided by an auto-incrementer
        """

        # used to uniquely identify the track
        if not id:
            self._id = Track._track_id
            Track._track_id += 1
        else:
            self._id = id

        self.clip_id = clip_id
        # frame number this track starts at
        self.start_frame = None
        self.end_frame = None
        self.start_s = None
        self.end_s = None
        self.current_frame = 0
        self.frame_list = []
        # our bounds over time
        self.bounds_history = []
        # number frames since we lost target.
        self.frames_since_target_seen = 0
        # our current estimated horizontal velocity
        self.vel_x = 0
        # our current estimated vertical velocity
        self.vel_y = 0
        self.track_data = []
        # the tag for this track
        self.tag = "unknown"
        self.prev_frame = None
        self.include_filtered_channel = True
        self.confidence = None
        self.from_metadata = False

    def get_id(self):
        return self._id

    def load_track_meta(self, track_meta, frames_per_second, include_filtered_channel):
        self.from_metadata = True
        self._id = track_meta["id"]
        self.include_filtered_channel = include_filtered_channel
        self.track_data = []
        data = track_meta["data"]
        self.start_s = data["start_s"]
        self.end_s = data["end_s"]

        self.tag = data.get("tag", "unknown")
        self.confidence = data["confidence"]
        positions = data.get("positions")
        if not positions:
            return False
        self.bounds_history = []
        self.frame_list = []
        for position in positions:
            frame_number = round(position[0] * frames_per_second)
            if self.start_frame is None:
                self.start_frame = frame_number
            self.end_frame = frame_number
            region = Region.region_from_array(position[1], frame_number)
            self.bounds_history.append(region)
            self.frame_list.append(frame_number)

        return True

    def add_frame_from_region(self, region, buffer_frame):
        self.bounds_history.append(region)
        channels = buffer_frame.get_frame_channels(region, region.frame_number)
        self.track_data.append(channels)
        self.end_frame = region.frame_number
        self.current_frame += 1

    def add_frame(self, frame_number, buffer_frame, mass_delta_threshold):

        region = self.bounds_history[self.current_frame]
        prev_filtered = buffer_frame.get_previous_filtered(region, frame_number)
        channels = buffer_frame.get_frame_channels(region, frame_number)
        filtered = channels[TrackChannels.filtered]

        region.calculate_mass(filtered, mass_delta_threshold)
        region.calculate_variance(filtered, prev_filtered)

        if self.prev_frame and frame_number:
            frame_diff = frame_number - self.prev_frame - 1
            for _ in range(frame_diff):
                self.add_blank_frame()

        if len(self) >= 2:
            self.vel_x = self.bounds_history[-1].mid_x - self.bounds_history[-2].mid_x
            self.vel_y = self.bounds_history[-1].mid_y - self.bounds_history[-2].mid_y
        else:
            self.vel_x = self.vel_y = 0

        self.prev_frame = frame_number
        self.current_frame += 1

        if not self.include_filtered_channel:
            channels[TrackChannels.filtered] = 0

        self.track_data.append(channels)

    def add_blank_frame(self, buffer_frame=None):
        """ Maintains same bounds as previously, does not reset framce_since_target_seen counter """
        region = self.last_bound.copy()
        region.mass = 0
        region.pixel_variance = 0
        region.frame_number += 1
        self.bounds_history.append(region)
        if buffer_frame:
            channels = buffer_frame.get_frame_channels(region, region.frame_number)
            self.track_data.append(channels)
        self.vel_x = self.vel_y = 0

    def get_stats(self):
        """
        Returns statistics for this track, including how much it moves, and a score indicating how likely it is
        that this is a good track.
        :return: a TrackMovementStatistics record
        """

        if len(self) <= 1:
            return TrackMovementStatistics()

        # get movement vectors
        mass_history = [int(bound.mass) for bound in self.bounds_history]
        variance_history = [bound.pixel_variance for bound in self.bounds_history]
        mid_x = [bound.mid_x for bound in self.bounds_history]
        mid_y = [bound.mid_y for bound in self.bounds_history]
        delta_x = [mid_x[0] - x for x in mid_x]
        delta_y = [mid_y[0] - y for y in mid_y]
        vel_x = [cur - prev for cur, prev in zip(mid_x[1:], mid_x[:-1])]
        vel_y = [cur - prev for cur, prev in zip(mid_y[1:], mid_y[:-1])]

        movement = sum((vx ** 2 + vy ** 2) ** 0.5 for vx, vy in zip(vel_x, vel_y))
        max_offset = max((dx ** 2 + dy ** 2) ** 0.5 for dx, dy in zip(delta_x, delta_y))

        # the standard deviation is calculated by averaging the per frame variances.
        # this ends up being slightly different as I'm using /n rather than /(n-1) but that
        # shouldn't make a big difference as n = width*height*frames which is large.
        delta_std = float(np.mean(variance_history)) ** 0.5

        movement_points = (movement ** 0.5) + max_offset
        delta_points = delta_std * 25.0
        score = min(movement_points, 100) + min(delta_points, 100)

        stats = TrackMovementStatistics(
            movement=float(movement),
            max_offset=float(max_offset),
            average_mass=float(np.mean(mass_history)),
            median_mass=float(np.median(mass_history)),
            delta_std=float(delta_std),
            score=float(score),
        )

        return stats

    def smooth(self, frame_bounds: Rectangle):
        """
        Smooths out any quick changes in track dimensions
        :param frame_bounds The boundaries of the video frame.
        """
        if len(self.bounds_history) == 0:
            return

        new_bounds_history = []
        prev_frame = self.bounds_history[0]
        current_frame = self.bounds_history[0]
        next_frame = self.bounds_history[1]

        for i in range(len(self.bounds_history)):

            prev_frame = self.bounds_history[max(0, i - 1)]
            current_frame = self.bounds_history[i]
            next_frame = self.bounds_history[min(len(self.bounds_history) - 1, i + 1)]

            frame_x = current_frame.mid_x
            frame_y = current_frame.mid_y
            frame_width = (
                prev_frame.width + current_frame.width + next_frame.width
            ) / 3
            frame_height = (
                prev_frame.height + current_frame.height + next_frame.height
            ) / 3
            frame = Region(
                int(frame_x - frame_width / 2),
                int(frame_y - frame_height / 2),
                int(frame_width),
                int(frame_height),
            )
            frame.crop(frame_bounds)

            new_bounds_history.append(frame)

        self.bounds_history = new_bounds_history

    def trim(self):
        """
        Removes empty frames from start and end of track
        """
        mass_history = [int(bound.mass) for bound in self.bounds_history]
        start = 0
        while start < len(self) and mass_history[start] <= 2:
            start += 1
        end = len(self) - 1
        while end > 0 and mass_history[end] <= 2:
            end -= 1

        if end < start:
            self.start_frame = 0
            self.bounds_history = []
            self.track_data = []
        else:
            self.start_frame += start
            self.bounds_history = self.bounds_history[start : end + 1]
            self.track_data = self.track_data[start : end + 1]

    def get_track_region_score(self, region: Region):
        """
        Calculates a score between this track and a region of interest.  Regions that are close the the expected
        location for this track are given high scores, as are regions of a similar size.
        """
        expected_x = int(self.last_bound.mid_x + self.vel_x)
        expected_y = int(self.last_bound.mid_y + self.vel_y)

        distance = (
            (region.mid_x - expected_x) ** 2 + (region.mid_y - expected_y) ** 2
        ) ** 0.5

        # ratio of 1.0 = 20 points, ratio of 2.0 = 10 points, ratio of 3.0 = 0 points.
        # area is padded with 50 pixels so small regions don't change too much
        size_difference = (
            abs(region.area - self.last_bound.area) / (self.last_bound.area + 50)
        ) * 100

        return distance, size_difference

    def get_overlap_ratio(self, other_track, threshold=0.05):
        """
        Checks what ratio of the time these two tracks overlap.
        :param other_track: the other track to compare with
        :param threshold: how much frames must be overlapping to be counted
        :return: the ratio of frames that overlap
        """

        if len(self) == 0 or len(other_track) == 0:
            return 0.0

        start = max(self.start_frame, other_track.start_frame)
        end = min(self.end_frame, other_track.end_frame)

        frames_overlapped = 0

        for pos in range(start, end + 1):
            our_index = pos - self.start_frame
            other_index = pos - other_track.start_frame
            if (
                our_index >= 0
                and other_index >= 0
                and our_index < len(self)
                and other_index < len(other_track)
            ):
                our_bounds = self.bounds_history[our_index]
                other_bounds = other_track.bounds_history[other_index]
                overlap = our_bounds.overlap_area(other_bounds) / our_bounds.area
                if overlap >= threshold:
                    frames_overlapped += 1

        return frames_overlapped / len(self)

    @property
    def last_mass(self):
        return self.bounds_history[-1].mass

    @property
    def last_bound(self) -> Region:
        return self.bounds_history[-1]

    def __repr__(self):
        return "Track:{} frames".format(len(self))

    def __len__(self):
        return len(self.bounds_history)


TrackMovementStatistics = namedtuple(
    "TrackMovementStatistics",
    "movement max_offset score average_mass median_mass delta_std",
)
TrackMovementStatistics.__new__.__defaults__ = (0,) * len(
    TrackMovementStatistics._fields
)
