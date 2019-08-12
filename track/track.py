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

from ml_tools.tools import Rectangle, get_clipped_flow
from ml_tools.dataset import TrackChannels
import track.region
from track.region import Region

from ml_tools.tools import eucl_distance


class Track:
    """ Bounds of a tracked object over time. """

    # keeps track of which id number we are up to.
    _track_id = 1

    def __init__(self, clip_id, id=None):
        """
        Creates a new Track.
        :param id: id number for track, if not specified is provided by an auto-incrementer
        """

        if not id:
            self._id = Track._track_id
            Track._track_id += 1
        else:
            self._id = id

        self.clip_id = clip_id
        self.start_frame = None
        self.end_frame = None
        self.start_s = None
        self.end_s = None
        self.current_frame_num = None
        self.frame_list = []
        # our bounds over time
        self.bounds_history = []
        # number frames since we lost target.
        self.frames_since_target_seen = 0
        # our current estimated horizontal velocity
        self.vel_x = 0
        # our current estimated vertical velocity
        self.vel_y = 0
        # the tag for this track
        self.tag = "unknown"
        self.prev_frame_num = None
        self.include_filtered_channel = True
        self.confidence = None
        self.max_novelty = None
        self.avg_novelty = None

        self.from_metadata = False
        self.track_tags = None

    def get_id(self):
        return self._id

    def load_track_meta(
        self,
        track_meta,
        frames_per_second,
        include_filtered_channel,
        tag_precedence,
        min_confidence,
    ):
        self.from_metadata = True
        self._id = track_meta["id"]
        self.include_filtered_channel = include_filtered_channel
        data = track_meta["data"]
        self.start_s = data["start_s"]
        self.end_s = data["end_s"]
        self.track_tags = track_meta.get("TrackTags")
        tag = Track.get_best_human_tag(track_meta, tag_precedence, min_confidence)
        if tag:
            self.tag = tag["what"]
            self.confidence = tag["confidence"]
        else:
            return False

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
        self.current_frame_num = 0
        return True

    def add_frame_from_region(self, region):
        if self.prev_frame_num and region.frame_number:
            frame_diff = region.frame_number - self.prev_frame_num - 1
            for _ in range(frame_diff):
                self.add_blank_frame()

        self.bounds_history.append(region)
        self.end_frame = region.frame_number
        self.prev_frame_num = region.frame_number
        self.update_velocity()
        self.frames_since_target_seen = 0

    def update_velocity(self):
        if len(self.bounds_history) >= 2:
            self.vel_x = self.bounds_history[-1].mid_x - self.bounds_history[-2].mid_x
            self.vel_y = self.bounds_history[-1].mid_y - self.bounds_history[-2].mid_y
        else:
            self.vel_x = self.vel_y = 0

    def add_frame_for_existing_region(self, frame, mass_delta_threshold, prev_filtered):
        region = self.bounds_history[self.current_frame_num]
        if prev_filtered is not None:
            prev_filtered = region.subimage(prev_filtered)

        filtered = region.subimage(frame.filtered)
        region.calculate_mass(filtered, mass_delta_threshold)
        region.calculate_variance(filtered, prev_filtered)

        if self.prev_frame_num and frame.frame_number:
            frame_diff = frame.frame_number - self.prev_frame_num - 1
            for _ in range(frame_diff):
                self.add_blank_frame()

        self.update_velocity()

        self.prev_frame_num = frame.frame_number
        self.current_frame_num += 1
        self.frames_since_target_seen = 0

    def add_blank_frame(self, buffer_frame=None):
        """ Maintains same bounds as previously, does not reset framce_since_target_seen counter """
        region = self.last_bound.copy()
        region.mass = 0
        region.pixel_variance = 0
        region.frame_number += 1
        self.bounds_history.append(region)
        self.vel_x = self.vel_y = 0
        self.frames_since_target_seen += 1

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
        variance_history = [
            bound.pixel_variance
            for bound in self.bounds_history
            if bound.pixel_variance
        ]
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
        else:
            self.start_frame += start
            self.bounds_history = self.bounds_history[start : end + 1]

    def get_track_region_score(self, region: Region, moving_vel_thresh):
        """
        Calculates a score between this track and a region of interest.  Regions that are close the the expected
        location for this track are given high scores, as are regions of a similar size.
        """

        if abs(self.vel_x) + abs(self.vel_y) >= moving_vel_thresh:
            expected_x = int(self.last_bound.mid_x + self.vel_x)
            expected_y = int(self.last_bound.mid_y + self.vel_y)
            distance = eucl_distance(
                (expected_x, expected_y), (region.mid_x, region.mid_y)
            )
        else:
            expected_x = int(self.last_bound.x + self.vel_x)
            expected_y = int(self.last_bound.y + self.vel_y)
            distance = eucl_distance((expected_x, expected_y), (region.x, region.y))
            distance += eucl_distance(
                (
                    expected_x + self.last_bound.width,
                    expected_y + self.last_bound.height,
                ),
                (region.x + region.width, region.y + region.height),
            )
            distance /= 2.0

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

    def crop_by_region_at_trackframe(self, frame, track_frame_number, clip_flow=True):
        bounds = self.bounds_history[track_frame_number]
        return self.crop_by_region(frame, bounds)

    def crop_by_region(self, frame, region, clip_flow=True):
        thermal = region.subimage(frame.thermal)
        filtered = region.subimage(frame.filtered)
        if frame.flow is not None:
            flow_h = region.subimage(frame.flow_h)
            flow_v = region.subimage(frame.flow_v)
            if clip_flow and not frame.flow_clipped:
                flow_h = get_clipped_flow(flow_h)
                flow_v = get_clipped_flow(flow_v)
        else:
            flow_h = None
            flow_v = None

        mask = region.subimage(frame.mask).copy()
        # make sure only our pixels are included in the mask.
        mask[mask != region.id] = 0
        mask[mask > 0] = 1

        # stack together into a numpy array.
        # by using int16 we lose a little precision on the filtered frames, but not much (only 1 bit)
        if flow_h is not None and flow_v is not None:
            return np.int16(np.stack((thermal, filtered, flow_h, flow_v, mask), axis=0))
        else:
            empty = np.zeros(filtered.shape)
            return np.int16(np.stack((thermal, filtered, empty, empty, mask), axis=0))
        return frame

    @property
    def frames(self):
        return self.end_frame + 1 - self.start_frame

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

    @classmethod
    def get_best_human_tag(cls, track_meta, tag_precedence, min_confidence=-1):
        """ returns highest precidence non AI tag from the metadata """

        track_tags = track_meta.get("TrackTags", [])
        track_tags = [
            tag
            for tag in track_tags
            if not tag.get("automatic", False)
            and tag.get("confidence") > min_confidence
        ]

        if not track_tags:
            return None

        tag = None
        default_prec = tag_precedence.get("default", 100)
        best = None
        for track_tag in track_tags:
            ranking = cls.tag_ranking(track_tag, tag_precedence, default_prec)

            # if 2 track_tags have same confidence ignore both
            if ranking == best:
                tag = None
            elif best is None or ranking < best:
                best = ranking
                tag = track_tag
        return tag

    @staticmethod
    def tag_ranking(track_tag, precedence, default_prec):
        """ returns a ranking of tags based of what they are and confidence """

        what = track_tag.get("what")
        confidence = 1 - track_tag.get("confidence", 0)
        prec = precedence.get(what, default_prec)
        return prec + confidence


TrackMovementStatistics = namedtuple(
    "TrackMovementStatistics",
    "movement max_offset score average_mass median_mass delta_std",
)
TrackMovementStatistics.__new__.__defaults__ = (0,) * len(
    TrackMovementStatistics._fields
)
