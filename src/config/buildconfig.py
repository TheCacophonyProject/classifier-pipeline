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
import os
import logging
from os import path
from .defaultconfig import DefaultConfig


@attr.s
class BuildConfig(DefaultConfig):
    test_clips_folder = attr.ib()
    banned_clips = attr.ib()
    segment_length = attr.ib()
    segment_spacing = attr.ib()
    segment_min_avg_mass = attr.ib()
    min_frame_mass = attr.ib()
    filter_by_lq = attr.ib()
    max_segments = attr.ib()

    @classmethod
    def load(cls, build):
        return cls(
            test_clips_folder=build["test_clips_folder"],
            banned_clips=load_banned_clips_file(build["banned_clips"]),
            segment_length=build["segment_length"],
            segment_spacing=build["segment_spacing"],
            segment_min_avg_mass=build["segment_min_avg_mass"],
            min_frame_mass=build["min_frame_mass"],
            filter_by_lq=build["filter_by_lq"],
            max_segments=build["max_segments"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            test_clips_folder=None,
            banned_clips=None,
            segment_length=25,
            segment_spacing=1,
            segment_min_avg_mass=10,
            min_frame_mass=10,
            filter_by_lq=False,
            max_segments=5,
        )

    def validate(self):
        return True

    def test_clips(self):
        if not self.test_clips_folder or not path.exists(self.test_clips_folder):
            return None
        if not os.path.isdir(self.test_clips_folder):
            return None
        test_clips = []
        for f in os.listdir(self.test_clips_folder):
            file_path = path.join(self.test_clips_folder, f)
            if path.isfile(file_path) and path.splitext(file_path)[1] == ".list":
                with open(file_path) as stream:
                    for line in stream:
                        if line.strip() == "":
                            continue
                        try:
                            clips = line.split(",")
                            # print("line is", line, "clips are", clips)
                            for clip in clips:
                                test_clips.append(int(clip.strip()))
                        except:
                            logging.warn(
                                "Could not parse clip_id %s from %s",
                                line,
                                file_path,
                            )
        return test_clips


def load_banned_clips_file(filename):
    if not filename or not os.path.exists(filename):
        return None
    if not path.isfile(filename):
        return None
    files = []

    with open(filename) as stream:
        for line in stream:
            files.append(line.strip())
    return files
