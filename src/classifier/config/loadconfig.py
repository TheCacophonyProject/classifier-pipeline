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

from .defaultconfig import DefaultConfig


@attr.s
class LoadConfig(DefaultConfig):
    EXCLUDED_TAGS = ["poor tracking", "part", "untagged", "unidentified"]

    DEFAULT_GROUPS = {
        0: [
            "bird",
            "false-positive",
            "hedgehog",
            "possum",
            "rodent",
            "mustelid",
            "cat",
            "kiwi",
            "dog",
            "leporidae",
            "human",
            "insect",
            "pest",
        ],
        1: ["unidentified", "other"],
        2: ["part", "bad track"],
        3: ["default"],
    }

    enable_compression = attr.ib()
    include_filtered_channel = attr.ib()
    preview = attr.ib()
    tag_precedence = attr.ib()
    cache_to_disk = attr.ib()
    high_quality_optical_flow = attr.ib()
    excluded_tags = attr.ib()

    @classmethod
    def load(cls, config):
        return cls(
            enable_compression=config["enable_compression"],
            include_filtered_channel=config["include_filtered_channel"],
            preview=config["preview"],
            tag_precedence=config["tag_precedence"],
            cache_to_disk=config["cache_to_disk"],
            high_quality_optical_flow=config["high_quality_optical_flow"],
            excluded_tags=config["excluded_tags"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            enable_compression=False,
            include_filtered_channel=True,
            preview=None,
            tag_precedence=LoadConfig.DEFAULT_GROUPS,
            cache_to_disk=False,
            high_quality_optical_flow=True,
            excluded_tags=LoadConfig.EXCLUDED_TAGS,
        )

    def validate(self):
        return True
