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

import os.path as path

import attr

import ml_tools.config
from ml_tools.defaultconfig import DefaultConfig
from ml_tools.previewer import Previewer


@attr.s
class ClassifyConfig(DefaultConfig):

    model = attr.ib()
    meta_to_stdout = attr.ib()
    preview = attr.ib()
    classify_folder = attr.ib()
    cache_to_disk = attr.ib()

    @classmethod
    def load(cls, classify, base_folder):
        return cls(
            model=classify["model"],
            meta_to_stdout=classify["meta_to_stdout"],
            preview=ml_tools.config.parse_options_param(
                "preview", classify["preview"], Previewer.PREVIEW_OPTIONS
            ),
            classify_folder=path.join(base_folder, classify["classify_folder"]),
            cache_to_disk=classify["cache_to_disk"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            meta_to_stdout=False,
            model=None,
            preview="none",
            classify_folder="classify",
            cache_to_disk=True,
        )

    def validate(self):
        if self.model is None:
            raise KeyError("model not found in configuration file")
