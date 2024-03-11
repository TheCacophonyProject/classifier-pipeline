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

import attr

from .defaultconfig import (
    DefaultConfig,
    deep_copy_map_if_key_not_exist,
)
from .trackingmotionconfig import TrackingMotionConfig
from track.cliptrackextractor import ClipTrackExtractor
from track.track import RegionTracker


@attr.s
class TrackingConfig(DefaultConfig):
    tracker = attr.ib()
    params = attr.ib()
    type = attr.ib()
    motion = attr.ib()
    edge_pixels = attr.ib()
    # dilation_pixels = attr.ib()
    min_dimension = attr.ib()
    frame_padding = attr.ib()
    track_smoothing = attr.ib()
    denoise = attr.ib()

    high_quality_optical_flow = attr.ib()
    flow_threshold = attr.ib()
    max_tracks = attr.ib()
    track_overlap_ratio = attr.ib()
    min_duration_secs = attr.ib()
    track_min_offset = attr.ib()
    track_min_mass = attr.ib()
    aoi_min_mass = attr.ib()
    aoi_pixel_variance = attr.ib()
    cropped_regions_strategy = attr.ib()
    enable_track_output = attr.ib()
    min_tag_confidence = attr.ib()
    moving_vel_thresh = attr.ib()
    min_moving_frames = attr.ib()
    max_blank_percent = attr.ib()
    max_mass_std_percent = attr.ib()

    max_jitter = attr.ib()
    # used to provide defaults
    filters = attr.ib()
    areas_of_interest = attr.ib()
    # filter regions out by mass and variance before matching to a track
    filter_regions_pre_match = attr.ib()
    min_hist_diff = attr.ib()

    @classmethod
    def load(cls, tracking):
        if tracking is None:
            return None
        trackers = {}
        for type, raw_tracker in tracking.items():
            if raw_tracker is None:
                raw_tracker = {}
            tracker = TrackingConfig.load_type(raw_tracker, type)
            trackers[tracker.type] = tracker
        return trackers

    @classmethod
    def load_type(cls, tracking, type):
        defaults = cls.get_type_defaults(type)
        deep_copy_map_if_key_not_exist(defaults.as_dict(), tracking)
        return cls(
            tracker=tracking["tracker"],
            params=tracking["params"],
            type=type,
            motion=TrackingMotionConfig.load(tracking.get("motion")),
            min_dimension=tracking["min_dimension"],
            edge_pixels=tracking["edge_pixels"],
            # dilation_pixels=tracking["dilation_pixels"],
            frame_padding=tracking["frame_padding"],
            track_smoothing=tracking["track_smoothing"],
            denoise=tracking["denoise"],
            high_quality_optical_flow=tracking["high_quality_optical_flow"],
            flow_threshold=tracking["flow_threshold"],
            max_tracks=tracking["max_tracks"],
            moving_vel_thresh=tracking["filters"]["moving_vel_thresh"],
            track_overlap_ratio=tracking["filters"]["track_overlap_ratio"],
            min_duration_secs=tracking["filters"]["min_duration_secs"],
            track_min_offset=tracking["filters"]["track_min_offset"],
            track_min_mass=tracking["filters"]["track_min_mass"],
            cropped_regions_strategy=tracking["areas_of_interest"][
                "cropped_regions_strategy"
            ],
            aoi_min_mass=tracking["areas_of_interest"]["min_mass"],
            aoi_pixel_variance=tracking["areas_of_interest"]["pixel_variance"],
            enable_track_output=tracking["enable_track_output"],
            min_tag_confidence=tracking["min_tag_confidence"],
            min_moving_frames=tracking["min_moving_frames"],
            max_blank_percent=tracking["max_blank_percent"],
            max_jitter=tracking["max_jitter"],
            filters=tracking["filters"],
            areas_of_interest=tracking["areas_of_interest"],
            max_mass_std_percent=tracking["max_mass_std_percent"],
            filter_regions_pre_match=tracking["filter_regions_pre_match"],
            min_hist_diff=tracking["min_hist_diff"],
        )

    @classmethod
    def get_defaults(cls):
        default_tracking = {}
        default_tracking["thermal"] = cls.get_type_defaults("thermal")
        default_tracking["IR"] = cls.get_type_defaults("IR")
        return default_tracking

    @classmethod
    def get_type_defaults(cls, type):
        default_tracking = cls(
            motion=TrackingMotionConfig.get_defaults(),
            edge_pixels=1,
            frame_padding=4,
            min_dimension=0,
            # dilation_pixels=2,
            track_smoothing=False,
            denoise=True,
            high_quality_optical_flow=False,
            flow_threshold=40,
            max_tracks=None,
            filters={
                "track_overlap_ratio": 0.5,
                "min_duration_secs": 0,
                "track_min_offset": 4.0,
                "track_min_mass": 2.0,
                "moving_vel_thresh": 4,
            },
            areas_of_interest={
                "min_mass": 4.0,
                "pixel_variance": 2.0,
                "cropped_regions_strategy": "cautious",
            },
            # defaults provided in dictionaries, placesholders to stop init complaining
            aoi_min_mass=4.0,
            aoi_pixel_variance=2.0,
            cropped_regions_strategy="cautious",
            track_min_offset=4.0,
            track_min_mass=2.0,
            track_overlap_ratio=0.5,
            min_duration_secs=0,
            min_tag_confidence=0.8,
            enable_track_output=True,
            moving_vel_thresh=4,
            min_moving_frames=2,
            max_blank_percent=30,
            max_mass_std_percent=RegionTracker.MASS_CHANGE_PERCENT,
            max_jitter=20,
            tracker="RegionTracker",
            type="thermal",
            params={
                "base_distance_change": 450,
                "min_mass_change": 20,
                "restrict_mass_after": 1.5,
                "mass_change_percent": 0.55,
                "max_distance": 2000,
                "max_blanks": 18,
                "velocity_multiplier": 2,
                "base_velocity": 2,
            },
            filter_regions_pre_match=True,
            min_hist_diff=None,
        )
        if type == "IR":
            # default_tracking.min_hist_diff = 0.95
            default_tracking.filters["min_duration_secs"] = 0
            default_tracking.min_duration_secs = 0
            default_tracking.filter_regions_pre_match = False
            default_tracking.areas_of_interest["pixel_variance"] = 0
            default_tracking.areas_of_interest["min_mass"] = 0
            default_tracking.filters["track_min_offset"] = 7
            default_tracking.track_min_offset = 20
            default_tracking.min_dimension = 10
            default_tracking.min_tracks = None
            default_tracking.frame_padding = 10
            default_tracking.edge_pixels = 0
            default_tracking.tracker = "RegionTracker"
            default_tracking.type = "IR"
            default_tracking.params = {
                "base_distance_change": 12000,
                "min_mass_change": None,
                "restrict_mass_after": 1.5,
                "mass_change_percent": None,
                "max_distance": 30752,
                "max_blanks": 18,
                "velocity_multiplier": 8,
                "base_velocity": 10,
            }
        return default_tracking

    #
    # def load_trackers(raw):
    #     if raw is None:
    #         return None
    #     trackers = {}
    #     for raw_tracker in raw.values():
    #         tracker = TrackingConfig.load(raw_tracker)
    #         trackers[tracker.type] = tracker
    #     return trackers

    def validate(self):
        return True

    def as_dict(self):
        return attr.asdict(self)

    def rescale(self, scale):
        # adjust numbers if we rescale frame sizes
        self.frame_padding = int(scale * self.frame_padding)
        self.min_dimension = int(scale * self.min_dimension)
        if self.params["base_distance_change"]:
            self.params["base_distance_change"] *= scale
        if self.params["min_mass_change"]:
            self.params["min_mass_change"] *= scale
        if self.params["max_distance"]:
            self.params["max_distance"] *= scale
        if self.params["base_velocity"]:
            self.params["base_velocity"] *= scale
        self.track_min_offset *= scale
        self.track_min_mass *= scale
        self.aoi_min_mass *= scale


#
#
# @attr.s
# class TrackerConfig(DefaultConfig):
#
#     tracker = attr.ib()
#     params = attr.ib()
#     type = attr.ib()
#
#     @classmethod
#     def load(cls, raw):
#         defaults = cls.get_defaults()
#         deep_copy_map_if_key_not_exist(defaults.as_dict(), raw)
#
#         return cls(
#             tracker=raw["tracker"],
#             params=raw["params"],
#             type=raw["type"],
#         )
#
#     def as_dict(self):
#         return attr.asdict(self)
#
#     @classmethod
#     def get_defaults(cls):
#         return cls(
#             tracker="RegionTracker",
#             type="IR",
#             params={
#                 "base_distance_change": 11250,
#                 "min_mass_change": 20 * 4,
#                 "restrict_mass_after": 1.5,
#                 "mass_change_percent": 0.55,
#                 "max_distance": 30752,
#             },
#         )
#
#     def validate(self):
#         return True
