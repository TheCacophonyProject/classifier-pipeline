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

import logging
import math
import numpy as np
from collections import namedtuple

from ml_tools.tools import Rectangle
from track.region import Region
from .kalman import Kalman
from ml_tools.tools import eucl_distance_sq
from ml_tools.datasetstructures import get_segments, SegmentHeader, SegmentType
import cv2
import logging
from track.tracker import Tracker


class RegionTracker(Tracker):
    # number of frames required before using kalman estimation
    MIN_KALMAN_FRAMES = 18

    # THERMAL VALUES
    # GP Need to put in config per camera type
    # MAX_DISTANCE = 2000
    #
    # TRACKER_VERSION = 1
    # BASE_DISTANCE_CHANGE = 450
    # # minimum region mass change
    # MIN_MASS_CHANGE = 20
    # # enforce mass growth after X seconds
    # RESTRICT_MASS_AFTER = 1.5
    # # amount region mass can change
    # MASS_CHANGE_PERCENT = 0.55

    # IR VALUES
    BASE_DISTANCE_CHANGE = 11250

    # minimum region mass change
    MIN_MASS_CHANGE = 20 * 4
    # enforce mass growth after X seconds
    RESTRICT_MASS_AFTER = 1.5
    # amount region mass can change
    MASS_CHANGE_PERCENT = 0.55

    # MAX_DISTANCE = 2000
    MAX_DISTANCE = 30752
    BASE_VELOCITY = 8
    VELOCITY_MULTIPLIER = 10

    def __init__(self, id, tracking_config, crop_rectangle=None):
        self.track_id = id
        self.clear_run = 0
        self.kalman_tracker = Kalman()
        self._frames_since_target_seen = 0
        self.frames = 0
        self._blank_frames = 0
        self._last_bound = None
        self.crop_rectangle = crop_rectangle
        self._tracking = False
        self.type = tracking_config.type
        self.min_mass_change = tracking_config.params.get(
            "min_mass_change", RegionTracker.MIN_MASS_CHANGE
        )
        self.max_distance = tracking_config.params.get(
            "max_distance", RegionTracker.MAX_DISTANCE
        )
        self.base_distance_change = tracking_config.params.get(
            "base_distance_change", RegionTracker.BASE_DISTANCE_CHANGE
        )
        self.restrict_mass_after = tracking_config.params.get(
            "restrict_mass_after", RegionTracker.RESTRICT_MASS_AFTER
        )
        self.mass_change_percent = tracking_config.params.get(
            "mass_change_percent", RegionTracker.MASS_CHANGE_PERCENT
        )
        self.velocity_multiplier = tracking_config.params.get(
            "velocity_multiplier", RegionTracker.VELOCITY_MULTIPLIER
        )
        self.base_velocity = tracking_config.params.get(
            "base_velocity", RegionTracker.BASE_VELOCITY
        )
        self.max_blanks = tracking_config.params.get("max_blanks", 18)

    @property
    def tracking(self):
        return self._tracking

    @property
    def last_bound(self):
        return self._last_bound

    def get_size_change(self, current_area, region: Region):
        """
        Gets a value representing the difference in regions sizes
        """

        # ratio of 1.0 = 20 points, ratio of 2.0 = 10 points, ratio of 3.0 = 0 points.
        # area is padded with 50 pixels so small regions don't change too much
        size_difference = abs(region.area - current_area) / (current_area + 50)

        return size_difference

    def match(self, regions, track):
        scores = []
        avg_mass = track.average_mass()
        max_distances = self.get_max_distance_change(track)
        for region in regions:
            size_change = self.get_size_change(track.average_area(), region)
            distances = self.last_bound.average_distance(region)

            max_size_change = get_max_size_change(track, region)
            max_mass_change = self.get_max_mass_change_percent(track, avg_mass)

            logging.debug(
                "Track %s %s has max size change %s, distances %s to region %s size change %s max distance %s",
                track,
                track.last_bound,
                max_size_change,
                distances,
                region,
                size_change,
                max_distances,
            )
            # only for thermal
            if type == "thermal":
                # GP should figure out good values for the 3 distances rather than the mean
                distances = [np.mean(distances)]
                max_distances = max_distances[:1]
            else:
                distances = [(distances[0] + distances[2]) / 2]
                max_distances = max_distances[:1]

            if max_mass_change and abs(avg_mass - region.mass) > max_mass_change:
                logging.debug(
                    "track %s region mass %s deviates too much from %s for region %s",
                    track.get_id(),
                    region.mass,
                    avg_mass,
                    region,
                )
                continue
            skip = False
            for distance, max_distance in zip(distances, max_distances):
                if max_distance is None:
                    continue
                if distance > max_distance:
                    logging.debug(
                        "track %s distance score %s bigger than max distance %s for region %s",
                        track.get_id(),
                        distance,
                        max_distance,
                        region,
                    )
                    skip = True
                    break
                    # continue
            if skip:
                continue

            if size_change > max_size_change:
                logging.debug(
                    "track % size_change %s bigger than max size_change %s for region %s",
                    track.get_id(),
                    size_change,
                    max_size_change,
                    region,
                )
                continue
            # only for thermal
            if type == "ir":
                distance_score = np.mean(distances)
            else:
                # GP should figure out good values for the 3 distances rather than the mean
                distance_score = distances[0]

            scores.append((distance_score, track, region))
        return scores

    def add_region(self, region):
        self.frames += 1
        if region.blank:
            self._blank_frames += 1
            self._frames_since_target_seen += 1
            stop_tracking = min(
                2 * (self.frames - self._frames_since_target_seen),
                self.max_blanks,
            )
            self._tracking = self._frames_since_target_seen < stop_tracking
        else:
            if self._frames_since_target_seen != 0:
                self.clear_run = 0
            self.clear_run += 1
            self._tracking = True
            self.kalman_tracker.correct(region)
            self._frames_since_target_seen = 0

        prediction = self.kalman_tracker.predict()
        self.predicted_mid = (prediction[0][0], prediction[1][0])
        self._last_bound = region

    @property
    def blank_frames(self):
        return self._blank_frames

    @property
    def frames_since_target_seen(self):
        return self._frames_since_target_seen

    @property
    def nonblank_frames(self):
        return self.frames - self._blank_frames

    def predicted_velocity(self):
        if (
            self.last_bound is None
            or self.nonblank_frames <= RegionTracker.MIN_KALMAN_FRAMES
        ):
            return (0, 0)
        pred_vel_x = self.predicted_mid[0] - self.last_bound.centroid[0]
        pred_vel_y = self.predicted_mid[1] - self.last_bound.centroid[1]

        return (pred_vel_x, pred_vel_y)

    def add_blank_frame(self):
        kalman_amount = (
            self.frames
            - RegionTracker.MIN_KALMAN_FRAMES
            - self._frames_since_target_seen * 2
        )

        if kalman_amount > 0:
            region = Region(
                int(self.predicted_mid[0] - self.last_bound.width / 2.0),
                int(self.predicted_mid[1] - self.last_bound.height / 2.0),
                self.last_bound.width,
                self.last_bound.height,
                centroid=[self.predicted_mid[0], self.predicted_mid[1]],
            )
            if self.crop_rectangle:
                region.crop(self.crop_rectangle)
        else:
            region = self.last_bound.copy()
        region.blank = True
        region.mass = 0
        region.pixel_variance = 0
        region.frame_number = self.last_bound.frame_number + 1

        self.add_region(region)
        return region

    def tracker_version(self):
        return f"RegionTracker-{RegionTracker.TRACKER_VERSION}"

    def get_max_distance_change(self, track):
        x, y = track.velocity
        # x = max(x, 2)
        # y = max(y, 2)
        if len(track) == 1:
            x = self.base_velocity
            y = self.base_velocity
        x = self.velocity_multiplier * x
        y = self.velocity_multiplier * y
        velocity_distance = x * x + y * y

        pred_vel = track.predicted_velocity()
        logging.debug(
            "%s velo %s pred vel %s vel distance %s",
            track,
            track.velocity,
            track.predicted_velocity(),
            velocity_distance,
        )
        pred_distance = pred_vel[0] * pred_vel[0] + pred_vel[1] * pred_vel[1]
        pred_distance = max(velocity_distance, pred_distance)
        max_distance = self.base_distance_change + max(velocity_distance, pred_distance)
        # max top left, max between predicted and region, max between right bottom
        distances = [max_distance, None, max_distance]
        return distances

    def get_max_mass_change_percent(self, track, average_mass):
        if self.mass_change_percent is None:
            return None
        if len(track) > self.restrict_mass_after * track.fps:
            vel = track.velocity
            mass_percent = self.mass_change_percent
            if np.sum(np.abs(vel)) > 5:
                # faster tracks can be a bit more deviant
                mass_percent = mass_percent + 0.1
            return max(
                self.min_mass_change,
                average_mass * mass_percent,
            )
        else:
            return None


def get_max_size_change(track, region):
    exiting = region.is_along_border and not track.last_bound.is_along_border
    entering = not exiting and track.last_bound.is_along_border
    region_percent = 1.5
    if len(track) < 5:
        # may increase at first
        region_percent = 2
    vel = np.sum(np.abs(track.velocity))
    if entering or exiting:
        region_percent = 2
        if vel > 10:
            region_percent *= 3
    elif vel > 10:
        region_percent *= 2
    return region_percent


class Track:
    """Bounds of a tracked object over time."""

    # keeps track of which id number we are up to.
    _track_id = 1

    # Percentage increase that is considered jitter, e.g. if a region gets
    # 30% bigger or smaller
    JITTER_THRESHOLD = 0.3
    # must change atleast 5 pixels to be considered for jitter
    MIN_JITTER_CHANGE = 5

    def __init__(
        self,
        clip_id,
        id=None,
        fps=9,
        tracking_config=None,
        crop_rectangle=None,
        tracker_version=None,
    ):
        """
        Creates a new Track.
        :param id: id number for track, if not specified is provided by an auto-incrementer
        """
        self.in_trap = False
        self.trap_reported = False
        self.trigger_frame = None
        self.direction = 0
        self.trap_tag = None
        if not id:
            self._id = Track._track_id
            Track._track_id += 1
        else:
            self._id = id
        self.clip_id = clip_id
        self.start_frame = None
        self.start_s = None
        self.end_s = None
        self.fps = fps
        self.current_frame_num = None
        self.frame_list = []
        # our bounds over time
        self.bounds_history = []
        # number frames since we lost target.

        self.vel_x = []
        self.vel_y = []
        # the tag for this track
        self.tag = "unknown"
        self.prev_frame_num = None
        self.confidence = None
        self.max_novelty = None
        self.avg_novelty = None

        self.from_metadata = False
        self.tags = None
        self.predictions = None
        self.predicted_class = None
        self.predicted_confidence = None

        self.all_class_confidences = None
        self.prediction_classes = None

        self.crop_rectangle = crop_rectangle

        self.predictions = None
        self.predicted_tag = None
        self.predicted_confidence = None

        self.all_class_confidences = None
        self.prediction_classes = None
        self.tracker_version = tracker_version

        self.tracker = None
        if tracking_config is not None:
            self.tracker = self.get_tracker(tracking_config)
        # self.tracker = RegionTracker(
        #     self.get_id(), tracking_config, self.crop_rectangle
        # )

    def get_tracker(self, tracking_config):
        tracker = tracking_config.tracker
        if tracker == "RegionTracker":
            return RegionTracker(self.get_id(), tracking_config, self.crop_rectangle)
        else:
            raise Exception(f"Cant find for tracker {tracker}")

    @property
    def blank_frames(self):
        if self.tracker is None:
            return 0
        return self.tracker.blank_frames

    @property
    def tracking(self):
        return self.tracker.tracking

    @property
    def frames_since_target_seen(self):
        return self.tracker.frames_since_target_seen

    def match(self, regions):
        return self.tracker.match(regions, self)

    def get_segments(
        self,
        # frame_temp_median,
        segment_width,
        segment_frame_spacing=9,
        repeats=1,
        min_frames=0,
        segment_frames=None,
        segment_type=SegmentType.ALL_RANDOM,
        from_last=None,
        max_segments=None,
        ffc_frames=None,
        dont_filter=False,
    ):
        if from_last is not None:
            if from_last == 0:
                return []
            regions = np.array(self.bounds_history[-from_last:])
            start_frame = regions[0].frame_number
        else:
            start_frame = self.start_frame
            regions = np.array(self.bounds_history)

        # frame_temp_median = np.uint16(frame_temp_median)
        segments = []
        if segment_frames is not None:
            mass_history = np.uint16([region.mass for region in regions])
            for frames in segment_frames:
                relative_frames = frames - self.start_frame
                mass_slice = mass_history[relative_frames]
                segment_mass = np.sum(mass_slice)
                segment = SegmentHeader(
                    self.clip_id,
                    self._id,
                    start_frame=start_frame,
                    frames=len(frames),
                    weight=1,
                    mass=segment_mass,
                    label=None,
                    regions=regions[relative_frames],
                    # frame_temp_median=frame_temp_median[relative_frames],
                    frame_indices=frames,
                )
                segments.append(segment)
        else:
            segments, _ = get_segments(
                self.clip_id,
                self._id,
                start_frame,
                segment_frame_spacing=segment_frame_spacing,
                segment_width=segment_width,
                regions=regions,
                ffc_frames=ffc_frames,
                repeats=repeats,
                # frame_temp_median=frame_temp_median,
                min_frames=min_frames,
                segment_frames=None,
                segment_type=segment_type,
                max_segments=max_segments,
                dont_filter=dont_filter,
            )
        return segments

    @classmethod
    def from_region(cls, clip, region, tracker_version=None, tracking_config=None):
        track = cls(
            clip.get_id(),
            fps=clip.frames_per_second,
            tracker_version=tracker_version,
            crop_rectangle=clip.crop_rectangle,
            tracking_config=tracking_config,
        )
        track.start_frame = region.frame_number
        track.start_s = region.frame_number / float(clip.frames_per_second)
        track.add_region(region)
        return track

    def get_id(self):
        return self._id

    def add_prediction_info(self, track_prediction):
        logging.warn("TODO add prediction info needs to be implemented")
        return

    def load_track_meta(
        self,
        track_meta,
        frames_per_second,
        tag_precedence=None,
        min_confidence=0.8,
    ):
        self.tracker_version = track_meta.get("tracker_version", "unknown")
        self.from_metadata = True
        self._id = track_meta["id"]
        extra_info = track_meta
        if "data" in track_meta:
            extra_info = track_meta["data"]
        if "start_s" in extra_info:
            self.start_s = extra_info["start_s"]
            self.end_s = extra_info["end_s"]
        else:
            self.start_s = extra_info["start"]
            self.end_s = extra_info["end"]
        self.fps = frames_per_second

        self.tags = track_meta.get("tags")
        tag = Track.get_best_human_tag(self.tags, tag_precedence, min_confidence)
        if tag:
            self.tag = tag["what"]
            self.confidence = tag["confidence"]

        positions = track_meta.get("positions")
        if not positions:
            return False
        self.bounds_history = []
        self.frame_list = []
        for i, position in enumerate(positions):
            if isinstance(position, list):
                region = Region.region_from_array(position[1])
                if region.frame_number is None:
                    frame_number = round(position[0] * frames_per_second)

                    region.frame_number = frame_number
            else:
                region = Region.region_from_json(position)
                if region.frame_number is None:
                    if "frameTime" in position:
                        if i == 0:
                            region.frame_number = position["frameTime"] * 9
                        else:
                            region.frame_number = (
                                self.bounds_history[0].frame_number + i
                            )
                    else:
                        raise Exception("No frame number info for track")
            if self.start_frame is None:
                self.start_frame = region.frame_number
            # self.end_frame = region.frame_number
            self.bounds_history.append(region)
            self.frame_list.append(region.frame_number)
        self.current_frame_num = 0
        return True

    #
    # def add_frame(self, frame):
    #     if self.stable:
    #         ok, bbox = self.tracker.update(frame.thermal)
    #         if ok:
    #             r = Region.from_ltwh(bbox[0], bbox[1], bbox[2], bbox[3])
    #
    #             r.crop(self.crop_rectangle)
    #
    #             sub_filtered = r.subimage(frame.filtered)
    #             r.calculate_mass(sub_filtered, 1)
    #             r.frame_number = frame.frame_number
    #             self.add_region(r, frame)
    #
    #         else:
    #             self.add_blank_frame()
    #
    def add_region(self, region):
        if self.prev_frame_num and region.frame_number:
            frame_diff = region.frame_number - self.prev_frame_num - 1
            for _ in range(frame_diff):
                self.add_blank_frame()
        self.tracker.add_region(region)
        self.bounds_history.append(region)
        # self.end_frame = region.frame_number
        self.prev_frame_num = region.frame_number
        self.update_velocity()

    def update_velocity(self):
        if len(self.bounds_history) >= 2:
            self.vel_x.append(
                self.bounds_history[-1].centroid[0]
                - self.bounds_history[-2].centroid[0]
            )
            self.vel_y.append(
                self.bounds_history[-1].centroid[1]
                - self.bounds_history[-2].centroid[1]
            )
        else:
            self.vel_x.append(0)
            self.vel_y.append(0)

    def crop_regions(self):
        if self.crop_rectangle is None:
            logging.info("No crop rectangle to crop with")
            return
        for region in self.bounds_history:
            region.crop(self.crop_rectangle)

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
        if self.tracker:
            self.tracker.add_region(region)

        self.update_velocity()
        self.prev_frame_num = frame.frame_number
        self.current_frame_num += 1

    def average_area(self):
        """Average mass of last 5 frames that weren't blank"""
        avg_area = 0
        count = 0
        for i in range(len(self.bounds_history)):
            bound = self.bounds_history[-i - 1]
            if not bound.blank:
                avg_area += bound.area
                count += 1
            if count == 5:
                break
        if count == 0:
            return 0
        return avg_area / count

    def average_mass(self):
        """Average mass of last 5 frames that weren't blank"""
        avg_mass = 0
        count = 0
        for i in range(len(self.bounds_history)):
            bound = self.bounds_history[-i - 1]
            if not bound.blank:
                avg_mass += bound.mass
                count += 1
            if count == 5:
                break
        if count == 0:
            return 0
        return avg_mass / count

    def add_blank_frame(self):
        """Maintains same bounds as previously, does not reset framce_since_target_seen counter"""
        if self.tracker:
            region = self.tracker.add_blank_frame()
        self.bounds_history.append(region)
        self.prev_frame_num = region.frame_number
        self.update_velocity()

    def get_stats(self):
        """
        Returns statistics for this track, including how much it moves, and a score indicating how likely it is
        that this is a good track.
        :return: a TrackMovementStatistics record
        """

        if len(self) <= 1:
            return TrackMovementStatistics()
        # get movement vectors only from non blank regions
        non_blank = [bound for bound in self.bounds_history if not bound.blank]
        mass_history = [int(bound.mass) for bound in non_blank]
        variance_history = [
            bound.pixel_variance for bound in non_blank if bound.pixel_variance
        ]
        movement = 0
        max_offset = 0

        frames_moved = 0
        avg_vel = 0
        first_point = self.bounds_history[0].mid
        for i, (vx, vy) in enumerate(zip(self.vel_x, self.vel_y)):
            region = self.bounds_history[i]
            if not region.blank:
                avg_vel += abs(vx) + abs(vy)
            if i == 0:
                continue

            if region.blank or self.bounds_history[i - 1].blank:
                continue
            if region.has_moved(self.bounds_history[i - 1]) or region.is_along_border:
                distance = (vx**2 + vy**2) ** 0.5
                movement += distance
                offset = eucl_distance_sq(first_point, region.mid)
                max_offset = max(max_offset, offset)
                frames_moved += 1
        avg_vel = avg_vel / len(mass_history)
        # the standard deviation is calculated by averaging the per frame variances.
        # this ends up being slightly different as I'm using /n rather than /(n-1) but that
        # shouldn't make a big difference as n = width*height*frames which is large.
        max_offset = math.sqrt(max_offset)
        delta_std = float(np.mean(variance_history)) ** 0.5
        jitter_bigger = 0
        jitter_smaller = 0
        for i, bound in enumerate(self.bounds_history[1:]):
            prev_bound = self.bounds_history[i]
            if prev_bound.is_along_border or bound.is_along_border:
                continue
            height_diff = bound.height - prev_bound.height
            width_diff = prev_bound.width - bound.width
            thresh_h = max(
                Track.MIN_JITTER_CHANGE, prev_bound.height * Track.JITTER_THRESHOLD
            )
            thresh_v = max(
                Track.MIN_JITTER_CHANGE, prev_bound.width * Track.JITTER_THRESHOLD
            )
            if abs(height_diff) > thresh_h:
                if height_diff > 0:
                    jitter_bigger += 1
                else:
                    jitter_smaller += 1
            elif abs(width_diff) > thresh_v:
                if width_diff > 0:
                    jitter_bigger += 1
                else:
                    jitter_smaller += 1

        movement_points = (movement**0.5) + max_offset
        delta_points = delta_std * 25.0
        jitter_percent = int(
            round(100 * (jitter_bigger + jitter_smaller) / float(self.frames))
        )

        blank_percent = int(round(100.0 * self.blank_frames / self.frames))
        score = (
            min(movement_points, 100)
            + min(delta_points, 100)
            + (100 - jitter_percent)
            + (100 - blank_percent)
        )
        stats = TrackMovementStatistics(
            movement=float(movement),
            max_offset=float(max_offset),
            average_mass=float(np.mean(mass_history)),
            median_mass=float(np.median(mass_history)),
            delta_std=float(delta_std),
            score=float(score),
            region_jitter=jitter_percent,
            jitter_bigger=jitter_bigger,
            jitter_smaller=jitter_smaller,
            blank_percent=blank_percent,
            frames_moved=frames_moved,
            mass_std=float(np.std(mass_history)),
            average_velocity=float(avg_vel),
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

            frame_x = current_frame.centroid[0]
            frame_y = current_frame.centroid[1]
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
        median_mass = np.median(mass_history)
        filter_mass = 0.005 * median_mass
        filter_mass = max(filter_mass, 2)
        start = 0
        logging.debug(
            "Triming track with median % and filter mass %s", median_mass, filter_mass
        )
        while start < len(self) and mass_history[start] <= filter_mass:
            start += 1
        end = len(self) - 1

        while end > 0 and mass_history[end] <= filter_mass:
            if self.tracker and self.frames_since_target_seen > 0:
                self.tracker._frames_since_target_seen -= 1
                self.tracker._blank_frames -= 1
            end -= 1
        if end < start:
            self.bounds_history = []
            self.vel_x = []
            self.vel_y = []
            if self.tracker:
                self.tracker._blank_frames = 0
        else:
            self.start_frame += start
            self.bounds_history = self.bounds_history[start : end + 1]
            self.vel_x = self.vel_x[start : end + 1]
            self.vel_y = self.vel_y[start : end + 1]
        self.start_s = self.start_frame / float(self.fps)

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
                if our_bounds.area == 0:
                    continue
                other_bounds = other_track.bounds_history[other_index]
                overlap = our_bounds.overlap_area(other_bounds) / our_bounds.area
                if overlap >= threshold:
                    frames_overlapped += 1

        return frames_overlapped / len(self)

    def set_end_s(self, fps):
        if len(self) == 0:
            self.end_s = self.start_s
            return
        self.end_s = (self.end_frame + 1) / fps

    def predicted_velocity(self):
        return self.tracker.predicted_velocity()

    def update_trapped_state(self):
        if self.in_trap:
            return self.in_trap
        min_frames = 2
        if len(self.bounds_history) < min_frames:
            return False
        self.in_trap = all(r.in_trap for r in self.bounds_history[-min_frames:])
        return self.in_trap

    @property
    def end_frame(self):
        if len(self.bounds_history) == 0:
            return self.start_frame
        return self.bounds_history[-1].frame_number

    @property
    def nonblank_frames(self):
        return self.end_frame + 1 - self.start_frame - self.blank_frames

    @property
    def frames(self):
        return self.end_frame + 1 - self.start_frame

    @property
    def last_mass(self):
        return self.bounds_history[-1].mass

    @property
    def velocity(self):
        return self.vel_x[-1], self.vel_y[-1]

    @property
    def last_bound(self) -> Region:
        return self.bounds_history[-1]

    def __repr__(self):
        return "Track: {} frames# {}".format(self.get_id(), len(self))

    def __len__(self):
        return len(self.bounds_history)

    def start_and_end_in_secs(self):
        if self.end_s is None:
            if len(self) == 0:
                self.end_s = self.start_s
            else:
                self.end_s = (self.end_frame + 1) / self.fps

        return (self.start_s, self.end_s)

    def get_metadata(self, predictions_per_model=None):
        track_info = {}
        start_s, end_s = self.start_and_end_in_secs()

        track_info["id"] = self.get_id()
        if self.in_trap:
            track_info["trap_triggered"] = self.in_trap
            track_info["trigger_frame"] = self.trigger_frame
            if self.trap_tag is not None:
                track_info["trap_tag"] = self.trap_tag
        track_info["tracker_version"] = self.tracker_version
        track_info["start_s"] = round(start_s, 2)
        track_info["end_s"] = round(end_s, 2)
        track_info["num_frames"] = len(self)
        track_info["frame_start"] = self.start_frame
        track_info["frame_end"] = self.end_frame
        track_info["positions"] = self.bounds_history
        prediction_info = []
        if predictions_per_model:
            for model_id, predictions in predictions_per_model.items():
                prediction = predictions.prediction_for(self.get_id())
                if prediction is None:
                    continue
                prediciont_meta = prediction.get_metadata()
                prediciont_meta["model_id"] = model_id
                prediction_info.append(prediciont_meta)
        track_info["predictions"] = prediction_info
        return track_info

    @classmethod
    def get_best_human_tag(cls, track_tags, tag_precedence, min_confidence=-1):
        """returns highest precidence non AI tag from the metadata"""
        if track_tags is None:
            return None
        track_tags = [
            tag
            for tag in track_tags
            if not tag.get("automatic", False)
            and tag.get("confidence") >= min_confidence
        ]

        if not track_tags:
            return None

        tag = None
        if tag_precedence is None:
            default_prec = 100
            tag_precedence = {}
        else:
            default_prec = tag_precedence.get("default", 100)
        best = None
        for track_tag in track_tags:
            ranking = cls.tag_ranking(track_tag, tag_precedence, default_prec)
            # if 2 track_tags have same confidence ignore both
            if tag and ranking == best and track_tag["what"] != tag["what"]:
                tag = None
            elif best is None or ranking < best:
                best = ranking
                tag = track_tag
        return tag

    @staticmethod
    def tag_ranking(track_tag, precedence, default_prec):
        """returns a ranking of tags based of what they are and confidence"""

        what = track_tag.get("what")
        confidence = 1 - track_tag.get("confidence", 0)
        prec = precedence.get(what, default_prec)
        return prec + confidence


TrackMovementStatistics = namedtuple(
    "TrackMovementStatistics",
    "movement max_offset score average_mass median_mass delta_std region_jitter jitter_smaller jitter_bigger blank_percent frames_moved mass_std, average_velocity",
)
TrackMovementStatistics.__new__.__defaults__ = (0,) * len(
    TrackMovementStatistics._fields
)
