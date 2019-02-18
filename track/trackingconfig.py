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

from collections import namedtuple
import ml_tools.config
from track.trackextractor import TrackExtractor

TrackingConfigTuple = namedtuple(
    "tracking",
    [
        "background_calc",
        "temp_thresh",
        "delta_thresh",
        "ignore_frames",
        "threshold_percentile",
        "static_background_threshold",
        "max_mean_temperature_threshold",
        "max_temperature_range_threshold",
        "edge_pixels",
        "frame_padding",
        "cropped_regions_strategy",
        "track_smoothing",
        "remove_track_after_frames",
        "high_quality_optical_flow",
        "min_threshold",
        "max_threshold",
        "flow_threshold",
        "max_tracks",
        "track_overlap_ratio",
        "min_duration_secs",
        "track_min_offset",
        "track_min_delta",
        "track_min_mass",
        "verbose",
    ],
)

class TrackingConfig(TrackingConfigTuple):

    @classmethod
    def load(cls, tracking):
        config = cls(
            background_calc=ml_tools.config.parse_options_param("background_calc", tracking["background_calc"],[TrackExtractor.PREVIEW, "stats"]),
            temp_thresh=tracking["preview"]["temp_thresh"],
            delta_thresh=tracking["preview"]["delta_thresh"],
            ignore_frames=tracking["preview"]["ignore_frames"],
            threshold_percentile=tracking["stats"]["threshold_percentile"],
            static_background_threshold=tracking["static_background_threshold"],
            max_mean_temperature_threshold=tracking["max_mean_temperature_threshold"],
            max_temperature_range_threshold=tracking["max_temperature_range_threshold"],
            edge_pixels=tracking["edge_pixels"],
            frame_padding=tracking["frame_padding"],
            cropped_regions_strategy=tracking["cropped_regions_strategy"],
            track_smoothing=tracking["track_smoothing"],
            remove_track_after_frames=tracking["remove_track_after_frames"],
            high_quality_optical_flow=tracking["high_quality_optical_flow"],
            min_threshold=tracking["stats"]["min_threshold"],
            max_threshold=tracking["stats"]["max_threshold"],
            flow_threshold=tracking["flow_threshold"],
            max_tracks=tracking["max_tracks"],
            track_overlap_ratio=tracking["filters"]["track_overlap_ratio"],
            min_duration_secs=tracking["filters"]["min_duration_secs"],
            track_min_offset=tracking["filters"]["track_min_offset"],
            track_min_delta=tracking["filters"]["track_min_delta"],
            track_min_mass=tracking["filters"]["track_min_mass"],
            verbose=tracking["verbose"],
        )
        return config


