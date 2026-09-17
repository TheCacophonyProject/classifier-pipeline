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

from dataclasses import dataclass
from typing import Any

from .defaultconfig import DefaultConfig
from pathlib import Path


@dataclass(slots=True)
class TrainConfig(DefaultConfig):
    hyper_params: Any
    train_dir: Any
    epochs: Any
    resnet_params: Any
    use_gru: Any
    label_probabilities: Any
    type: Any
    LABEL_PROBABILITIES = {
        "bird": 20,
        "possum": 20,
        "rodent": 20,
        "hedgehog": 20,
        "cat": 5,
        "insect": 1,
        "leporidae": 5,
        "mustelid": 5,
        "false-positive": 1,
        "wallaby": 5,
        "vehicle": 1,
        "human": 1,
    }

    @classmethod
    def load(cls, raw, base_data_folder):
        resent_config = None
        if raw.get("resnet_params"):
            resent_config = ResnetConfig.load(raw.get("resnet_params"))
        return cls(
            type=raw["type"],
            resnet_params=resent_config,
            hyper_params=raw["hyper_params"],
            train_dir=Path(base_data_folder) / raw.get("train_dir", "train"),
            epochs=raw["epochs"],
            use_gru=raw["use_gru"],
            label_probabilities=raw["label_probabilities"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            type="thermal",
            hyper_params={},
            resnet_params=None,
            train_dir=Path("train"),
            epochs=60,
            use_gru=True,
            label_probabilities=TrainConfig.LABEL_PROBABILITIES,
        )

    def validate(self):
        return True


@dataclass(slots=True)
class ResnetConfig:
    #  resnet
    num_filters: Any
    kernel_size: Any
    conv_stride: Any
    block_sizes: Any
    block_strides: Any
    bottleneck: Any
    resnet_size: Any
    first_pool_size: Any
    first_pool_stride: Any

    @classmethod
    def load(cls, raw):
        return cls(
            num_filters=raw.get("num_filters"),
            kernel_size=raw.get("kernel_size"),
            conv_stride=raw.get("conv_stride"),
            block_sizes=raw.get("block_sizes"),
            block_strides=raw.get("block_strides", [1, 2, 2, 2]),
            bottleneck=raw.get("bottleneck"),
            resnet_size=raw.get("resnet_size"),
            first_pool_size=raw.get("first_pool_size"),
            first_pool_stride=raw.get("first_pool_stride"),
        )
