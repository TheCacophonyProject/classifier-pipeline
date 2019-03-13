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
import os.path as path

import ml_tools.config
from ml_tools.previewer import Previewer

ClassifyConfigTuple = namedtuple(
    "classify",
    [
        "model",
        "meta_to_stdout",
        "preview",
        "classify_folder",
    ],
)

class ClassifyConfig(ClassifyConfigTuple):

    @classmethod
    def load(cls, classify, base_folder):
        config = cls(model=classify["model"],
            meta_to_stdout=classify["meta_to_stdout"],
            preview=ml_tools.config.parse_options_param("preview", classify["preview"], Previewer.PREVIEW_OPTIONS),
            classify_folder=path.join(base_folder, classify["classify_folder"]),
        )
        return config


