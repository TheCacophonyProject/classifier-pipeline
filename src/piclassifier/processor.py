"""
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2020, The Cacophony Project

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

from abc import ABC, abstractmethod

from .service import SnapshotService


class Processor(ABC):
    def __init__(
        self,
    ):
        model_labels = {}
        if self.classifier is not None:
            model_labels[self.classifier.id] = self.classifier.labels
        if self.fp_model is not None:
            model_labels[self.fp_model.id] = self.fp_model.labels

        self.service = SnapshotService(
            self.get_recent_frame,
            self.headers,
            self.take_snapshot,
            model_labels,
            self.get_thumbnail,
        )

    @abstractmethod
    def take_snapshot(self): ...

    @abstractmethod
    def process_frame(self, lepton_frame): ...

    @abstractmethod
    def get_recent_frame(self, last_frame=None): ...

    @abstractmethod
    def disconnected(self): ...

    @abstractmethod
    def skip_frame(self): ...

    @property
    @abstractmethod
    def res_x(self): ...

    @property
    @abstractmethod
    def res_y(self): ...

    @property
    @abstractmethod
    def output_dir(self): ...
