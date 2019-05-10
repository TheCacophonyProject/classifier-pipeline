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

from os import path

import attr

import ml_tools.config

@attr.s
class TrainConfig:
    hyper_params = attr.ib()
    train_dir = attr.ib()
    epochs = attr.ib()

    @classmethod
    def load(cls, raw, base_data_folder):
        return cls(
            hyper_params=raw["hyper_params"],
            train_dir=path.join(base_data_folder, raw.get("train_dir", "train")),
            epochs=raw["epochs"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            hyper_params={
                "batch_size": 16,
                "learning_rate": 0.0004,
                "learning_rate_decay": 1.0,
                "l2_reg": 0,
                "label_smoothing": 0.1,
                "keep_prob": 0.2,
                "batch_norm": True,
                "lstm_units": 256,
                "enable_flow": True,
                "augmentation": True,
                "thermal_threshold": 10,
                "scale_frequency": 0.5,
            },
            train_dir="train",
            epochs=30,
        )
        
    def validate(self):
        return True
