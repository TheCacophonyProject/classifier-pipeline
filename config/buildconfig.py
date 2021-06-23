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
import dateutil.parser
import os
import logging
from os import path
from .defaultconfig import DefaultConfig


@attr.s
class BuildConfig(DefaultConfig):
    banned_clips_file = attr.ib()
    banned_clips = attr.ib()
    test_clips_folder = attr.ib()
    clip_end_date = attr.ib()
    cap_bin_weight = attr.ib()
    use_previous_split = attr.ib()
    excluded_trap = attr.ib()
    label_weights = attr.ib()
    test_min_mass = attr.ib()
    train_min_mass = attr.ib()
    max_validation_set_track_duration = attr.ib()
    test_set_count = attr.ib()
    test_set_bins = attr.ib()
    segment_length = attr.ib()
    segment_spacing = attr.ib()
    previous_split = attr.ib()
    max_segments_per_track = attr.ib()
    max_frames_per_track = attr.ib()

    @classmethod
    def load(cls, build):
        return cls(
            banned_clips_file=build["banned_clips_file"],
            test_clips_folder=build["test_clips_folder"],
            clip_end_date=dateutil.parser.parse(build["clip_end_date"])
            if build["clip_end_date"]
            else None,
            cap_bin_weight=build["cap_bin_weight"],
            use_previous_split=build["use_previous_split"],
            banned_clips=load_banned_clips_file(build["banned_clips_file"]),
            excluded_trap=build["excluded_trap"],
            label_weights=build["label_weights"],
            test_min_mass=build["test_min_mass"],
            train_min_mass=build["train_min_mass"],
            max_validation_set_track_duration=build[
                "max_validation_set_track_duration"
            ],
            test_set_count=build["test_set_count"],
            test_set_bins=build["test_set_bins"],
            segment_length=build["segment_length"],
            segment_spacing=build["segment_spacing"],
            previous_split=build["previous_split"],
            max_segments_per_track=build["max_segments_per_track"],
            max_frames_per_track=build["max_frames_per_track"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            test_clips_folder=None,
            banned_clips_file=None,
            clip_end_date=None,
            cap_bin_weight=1.5,
            use_previous_split=True,
            banned_clips=None,
            excluded_trap=True,
            label_weights=None,
            test_min_mass=30,
            train_min_mass=20,
            max_validation_set_track_duration=3 * 60,
            test_set_count=300,
            test_set_bins=10,
            segment_length=3,
            segment_spacing=1,
            previous_split="template.dat",
            max_segments_per_track=None,
            max_frames_per_track=None,
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
                                test_clips.append(int(clip))
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
