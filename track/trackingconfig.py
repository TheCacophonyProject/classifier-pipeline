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

configTuple = namedtuple(
    "tracking",
    [
        "enable_compression",
        "include_filtered_channel",
        "threshold_percentile",
        "static_background_threshold",
        "max_mean_temperature_threshold",
        "max_temperature_range_threshold",
        "frame_padding",
        "cropped_regions_strategy",
        "track_smoothing",
        "remove_track_after_frames",
    ],
)

class TrackingConfig(configTuple):

    @classmethod
    def load(cls, tracking):
        return cls(enable_compression = tracking["enable_compression"],
            include_filtered_channel=tracking["include_filtered_channel"],
            threshold_percentile=tracking["threshold_percentile"],
            static_background_threshold=tracking["static_background_threshold"],
            max_mean_temperature_threshold=tracking["max_mean_temperature_threshold"],
            max_temperature_range_threshold=tracking["max_temperature_range_threshold"],
            frame_padding=tracking["frame_padding"],
            cropped_regions_strategy=tracking["cropped_regions_strategy"],
            track_smoothing=tracking["track_smoothing"],
            remove_track_after_frames=tracking["remove_track_after_frames"],
        )
