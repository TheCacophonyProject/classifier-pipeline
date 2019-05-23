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
import os.path as path

import ml_tools.config
from ml_tools.defaultconfig import DefaultConfig
from ml_tools.previewer import Previewer


@attr.s
class EvaluateConfig(DefaultConfig):

    show_extended_evaluation = attr.ib()
    new_visit_threshold = attr.ib()
    null_tags = attr.ib()

    @classmethod
    def load(cls, classify, base_folder):
        return cls(
            show_extended_evaluation=classify["show_extended_evaluation"],
            new_visit_threshold=classify["new_visit_threshold"],
            null_tags=classify["null_tags"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            show_extended_evaluation=False,
            new_visit_threshold=180,
            null_tags=["false-positive", "none", "no-tag"],
        )

    def validate(self):
        return True
