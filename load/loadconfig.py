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


@attr.s
class LoadConfig:

    enable_compression = attr.ib()
    include_filtered_channel = attr.ib()
    preview = attr.ib()

    @classmethod
    def load(cls, extract):
        return cls(
            enable_compression=extract["enable_compression"],
            include_filtered_channel=extract["include_filtered_channel"],
            preview=extract["preview"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            enable_compression=False, include_filtered_channel=False, preview="tracking"
        )

    def validate(self):
        return True
