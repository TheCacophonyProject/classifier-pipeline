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
import math
import numpy as np
from collections import namedtuple

from ml_tools.tools import Rectangle
from track.region import Region
from kalman.kalman import Kalman
from ml_tools.tools import eucl_distance
from ml_tools.datasetstructures import get_segments, SegmentHeader
import cv2


class Track:
    """Bounds of a tracked object over time."""

    # keeps track of which id number we are up to.
    _track_id = 1
    # number of frames required before using kalman estimation
    MIN_KALMAN_FRAMES = 18
    # Percentage increase that is considered jitter, e.g. if a region gets
    # 30% bigger or smaller
    JITTER_THRESHOLD = 0.3
    # must change atleast 5 pixels to be considered for jitter
    MIN_JITTER_CHANGE = 5

    def __init__(
        self, clip_id, id=None, fps=9, crop_rectangle=None, tracker_version=None
    ):
        """
        Creates a new Track.
        :param id: id number for track, if not specified is provided by an auto-incrementer
        """

        if not id:
            self._id = Track._track_id
            Track._track_id += 1
        else:
            self._id = id
        self.tracker = None
        self.clip_id = clip_id
        self.start_frame = None
        self.end_frame = None
        self.start_s = None
        self.end_s = None
        self.fps = fps
        self.current_frame_num = None
        self.frame_list = []
        # our bounds over time
        self.bounds_history = []
        # number frames since we lost target.
        self.frames_since_target_seen = 0
        self.blank_frames = 0

        self.vel_x = []
        self.vel_y = []
        # the tag for this track
        self.tag = "unknown"
        self.prev_frame_num = None
        self.confidence = None
        self.max_novelty = None
        self.avg_novelty = None

        self.from_metadata = False
        self.track_tags = None
        self.kalman_tracker = Kalman()
        self.predictions = None
        self.predicted_class = None
        self.predicted_confidence = None

        self.all_class_confidences = None
        self.prediction_classes = None

        self.predicted_mid = None
        self.crop_rectangle = crop_rectangle

        self.predictions = None
        self.predicted_tag = None
        self.predicted_confidence = None

        self.all_class_confidences = None
        self.prediction_classes = None
        self.tracker_version = tracker_version
        self.stable = False

    def get_segments(
        self,
        ffc_frames,
        frame_temp_median,
        segment_width,
        segment_frame_spacing=9,
        repeats=1,
        min_frames=0,
        segment_frames=None,
    ):

        regions = np.array(self.bounds_history)
        frame_temp_median = np.uint16(frame_temp_median)
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
                    start_frame=self.start_frame,
                    frames=len(frames),
                    weight=1,
                    mass=segment_mass,
                    label=None,
                    regions=regions[relative_frames],
                    frame_temp_median=frame_temp_median[relative_frames],
                    frame_indices=frames,
                )
                segments.append(segment)
        else:
            has_mass = any([region.mass for region in regions if region.mass > 0])
            segments, _ = get_segments(
                self.clip_id,
                self._id,
                self.start_frame,
                segment_frame_spacing,
                segment_width,
                regions=regions,
                ffc_frames=ffc_frames,
                repeats=repeats,
                frame_temp_median=frame_temp_median,
                min_frames=min_frames,
                segment_frames=None,
                ignore_mass=not has_mass,
            )
        return segments

    @classmethod
    def from_region(cls, clip, region, tracker_version=None, frame=None):
        track = cls(
            clip.get_id(),
            fps=clip.frames_per_second,
            tracker_version=tracker_version,
            crop_rectangle=clip.crop_rectangle,
        )
        track.start_frame = region.frame_number
        track.start_s = region.frame_number / float(clip.frames_per_second)
        track.add_region(region, frame)
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
        tag_precedence,
        min_confidence,
    ):
        self.from_metadata = True
        self._id = track_meta["id"]
        extra_info = track_meta
        if "data" in track_meta:
            extra_info = track_meta["data"]

        self.start_s = extra_info["start_s"]
        self.end_s = extra_info["end_s"]
        self.fps = frames_per_second
        self.predicted_tag = extra_info.get("tag")
        self.all_class_confidences = extra_info.get("all_class_confidences", None)
        self.predictions = extra_info.get("predictions")

        self.track_tags = track_meta.get("TrackTags")
        self.prediction_classes = extra_info.get("classes")
        tag = Track.get_best_human_tag(self.track_tags, tag_precedence, min_confidence)
        if tag:
            self.tag = tag["what"]
            self.confidence = tag["confidence"]

        positions = extra_info.get("positions")
        if not positions:
            return False
        self.bounds_history = []
        self.frame_list = []
        for position in positions:
            if isinstance(position, list):
                frame_number = round(position[0] * frames_per_second)
                region = Region.region_from_array(position[1], frame_number)
            else:
                region = Region.region_from_json(position)

            if self.start_frame is None:
                self.start_frame = region.frame_number
            self.end_frame = region.frame_number
            self.bounds_history.append(region)
            self.frame_list.append(region.frame_number)
        self.current_frame_num = 0
        return True

    def add_frame(self, frame):
        if self.stable:
            ok, bbox = self.tracker.update(frame.thermal)
            if ok:
                r = Region.from_ltwh(bbox[0], bbox[1], bbox[2], bbox[3])

                r.crop(self.crop_rectangle)

                sub_filtered = r.subimage(frame.filtered)
                r.calculate_mass(sub_filtered, 1)
                r.frame_number = frame.frame_number
                self.add_region(r, frame)

            else:
                self.add_blank_frame()

    def add_region(self, region, frame):
        if self.tracker is None:
            self.tracker = cv2.TrackerCSRT_create()
        if not self.stable:
            ok = self.tracker.init(frame.thermal, region.to_ltwh())

        if self.prev_frame_num and region.frame_number:
            frame_diff = region.frame_number - self.prev_frame_num - 1
            for _ in range(frame_diff):
                self.add_blank_frame()

        self.bounds_history.append(region)
        self.end_frame = region.frame_number
        self.prev_frame_num = region.frame_number
        self.update_velocity()
        self.frames_since_target_seen = 0
        self.kalman_tracker.correct(region)
        prediction = self.kalman_tracker.predict()

        self.predicted_mid = (prediction[0][0], prediction[1][0])
        if len(self) > 10 and not self.stable:
            stable = True
            for r in self.bounds_history[-10:]:
                if r.blank:
                    print("blank???")
                    stable = False
                    break
                w_diff = region.width - r.width
                h_diff = region.height - r.height
                if w_diff > 10 or h_diff > 10:
                    stable = False
                    break
                    # print("not stable", w_diff, h_diff)
            print("setting stable for track", self, stable)
            self.stable = stable

    def update_velocity(self):
        if len(self.bounds_history) >= 2:
            self.vel_x.append(
                self.bounds_history[-1].mid_x - self.bounds_history[-2].mid_x
            )
            self.vel_y.append(
                self.bounds_history[-1].mid_y - self.bounds_history[-2].mid_y
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
        self.update_velocity()
        self.prev_frame_num = frame.frame_number
        self.current_frame_num += 1
        self.frames_since_target_seen = 0
        self.kalman_tracker.correct(region)

    def average_mass(self):
        """Average mass of last 3 frames that weren't blank"""
        avg_mass = 0
        count = 0
        for i in range(len(self.bounds_history)):
            bound = self.bounds_history[-i - 1]
            if not bound.blank:
                avg_mass += bound.mass
                count += 1
            if count == 3:
                break
        if count == 0:
            return 0
        return avg_mass / count

    def add_blank_frame(self, buffer_frame=None):
        """Maintains same bounds as previously, does not reset framce_since_target_seen counter"""
        if self.frames > Track.MIN_KALMAN_FRAMES:
            region = Region(
                int(self.predicted_mid[0] - self.last_bound.width / 2.0),
                int(self.predicted_mid[1] - self.last_bound.height / 2.0),
                self.last_bound.width,
                self.last_bound.height,
            )
            if self.crop_rectangle:
                region.crop(self.crop_rectangle)
        else:
            region = self.last_bound.copy()
        region.blank = True
        region.mass = 0
        region.pixel_variance = 0
        region.frame_number = self.last_bound.frame_number + 1
        self.bounds_history.append(region)
        self.prev_frame_num = region.frame_number
        self.update_velocity()
        self.blank_frames += 1
        self.frames_since_target_seen += 1
        prediction = self.kalman_tracker.predict()
        self.predicted_mid = (prediction[0][0], prediction[1][0])

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
                distance = (vx ** 2 + vy ** 2) ** 0.5
                movement += distance
                offset = eucl_distance(first_point, region.mid)
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

        movement_points = (movement ** 0.5) + max_offset
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
            if self.frames_since_target_seen > 0:
                self.frames_since_target_seen -= 1
                self.blank_frames -= 1
            end -= 1
        if end < start:
            self.start_frame = 0
            self.bounds_history = []
            self.vel_x = []
            self.vel_y = []
            self.blank_frames = 0
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
        self.end_s = (self.end_frame + 1) / fps

    def predicted_velocity(self):
        prev = self.last_bound
        if prev is None or self.nonblank_frames <= Track.MIN_KALMAN_FRAMES:
            return (0, 0)
        pred_vel_x = self.predicted_mid[0] - prev.mid_x
        pred_vel_y = self.predicted_mid[1] - prev.mid_y

        return (pred_vel_x, pred_vel_y)

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
            self.end_s = (self.end_frame + 1) / self.fps

        return (self.start_s, self.end_s)

    def get_metadata(self, predictions_per_model=None):
        track_info = {}
        start_s, end_s = self.start_and_end_in_secs()

        track_info["id"] = self.get_id()
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
