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

import ml_tools.config
from track.trackextractor import TrackExtractor


@attr.s
class TrackingConfig:

    background_calc = attr.ib()
    temp_thresh = attr.ib()
    dynamic_thresh = attr.ib()
    delta_thresh = attr.ib()
    ignore_frames = attr.ib()
    threshold_percentile = attr.ib()
    static_background_threshold = attr.ib()
    max_mean_temperature_threshold = attr.ib()
    max_temperature_range_threshold = attr.ib()
    edge_pixels = attr.ib()
    dilation_pixels = attr.ib()
    frame_padding = attr.ib()
    track_smoothing = attr.ib()
    remove_track_after_frames = attr.ib()
    high_quality_optical_flow = attr.ib()
    min_threshold = attr.ib()
    max_threshold = attr.ib()
    flow_threshold = attr.ib()
    max_tracks = attr.ib()
    track_overlap_ratio = attr.ib()
    min_duration_secs = attr.ib()
    track_min_offset = attr.ib()
    track_min_delta = attr.ib()
    track_min_mass = attr.ib()
    aoi_min_mass = attr.ib()
    aoi_pixel_variance = attr.ib()
    cropped_regions_strategy = attr.ib()
    verbose = attr.ib()
    moving_vel_thresh = attr.ib()

    @classmethod
    def load(cls, tracking):
        return cls(
            background_calc=ml_tools.config.parse_options_param(
                "background_calc",
                tracking["background_calc"],
                [TrackExtractor.PREVIEW, "stats"],
            ),
            dynamic_thresh=tracking["preview"]["dynamic_thresh"],
            temp_thresh=tracking["preview"]["temp_thresh"],
            delta_thresh=tracking["preview"]["delta_thresh"],
            ignore_frames=tracking["preview"]["ignore_frames"],
            threshold_percentile=tracking["stats"]["threshold_percentile"],
            static_background_threshold=tracking["static_background_threshold"],
            max_mean_temperature_threshold=tracking["max_mean_temperature_threshold"],
            max_temperature_range_threshold=tracking["max_temperature_range_threshold"],
            edge_pixels=tracking["edge_pixels"],
            dilation_pixels=tracking["dilation_pixels"],
            frame_padding=tracking["frame_padding"],
            track_smoothing=tracking["track_smoothing"],
            remove_track_after_frames=tracking["remove_track_after_frames"],
            high_quality_optical_flow=tracking["high_quality_optical_flow"],
            min_threshold=tracking["stats"]["min_threshold"],
            max_threshold=tracking["stats"]["max_threshold"],
            flow_threshold=tracking["flow_threshold"],
            max_tracks=tracking["max_tracks"],
            moving_vel_thresh=tracking["filters"]["moving_vel_thresh"],
            track_overlap_ratio=tracking["filters"]["track_overlap_ratio"],
            min_duration_secs=tracking["filters"]["min_duration_secs"],
            track_min_offset=tracking["filters"]["track_min_offset"],
            track_min_delta=tracking["filters"]["track_min_delta"],
            track_min_mass=tracking["filters"]["track_min_mass"],
            cropped_regions_strategy=tracking["areas_of_interest"][
                "cropped_regions_strategy"
            ],
            aoi_min_mass=tracking["areas_of_interest"]["min_mass"],
            aoi_pixel_variance=tracking["areas_of_interest"]["pixel_variance"],
            verbose=tracking["verbose"],
        )

    def as_dict(self):
        return attr.asdict(self)
