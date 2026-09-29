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
from enum import Enum

from dataclasses import dataclass, asdict
from typing import Any
import logging

from config import config
from .defaultconfig import DefaultConfig


class PreviewType(Enum):
    NONE = "none"
    RAW = "raw"
    CLASSIFIED = "classified"
    TRACKING = "tracking"
    BOXES = "boxes"

    @classmethod
    def options(cls):
        return [None] + [preview_type.value for preview_type in cls]


@dataclass(slots=True)
class ClassifyConfig(DefaultConfig):
    models: Any
    meta_to_stdout: Any
    preview: Any
    cache_to_disk: Any

    @classmethod
    def load(cls, classify):
        return cls(
            models=ClassifyConfig.load_models(classify.get("models")),
            meta_to_stdout=classify["meta_to_stdout"],
            preview=config.parse_options_param(
                "preview", classify["preview"], PreviewType.options()
            ),
            cache_to_disk=classify["cache_to_disk"],
        )

    def load_models(raw):
        if raw is None:
            return None

        models = []
        for model in raw:
            models.append(ModelConfig.load(model))

        return models

    @classmethod
    def get_defaults(cls):
        return cls(
            models=None,
            meta_to_stdout=False,
            preview="none",
            cache_to_disk=False,
        )

    def validate(self):
        if self.models is None:
            return
        for model in self.models:
            model.validate()


@dataclass(slots=True)
class ModelConfig:
    DEFAULT_SCORE = 0
    id: Any
    name: Any
    type: Any

    model_file: Any
    model_weights: Any

    wallaby: Any
    tag_scores: Any
    ignored_tags: Any
    thumbnail_model: Any
    reclassify: Any
    submodel: Any
    run_over_network: Any
    port: Any = 8123

    @classmethod
    def load(cls, raw):
        model = cls(
            id=raw["id"],
            name=raw["name"],
            type=raw.get("type"),
            model_file=raw["model_file"],
            model_weights=raw.get("model_weights"),
            wallaby=raw.get("wallaby", False),
            tag_scores=load_scores(raw.get("tag_scores", {})),
            ignored_tags=raw.get("ignored_tags", []),
            thumbnail_model=raw.get("thumbnail_model", False),
            reclassify=raw.get("reclassify", None),
            submodel=raw.get("submodel", False),
            run_over_network=raw.get("run_over_network", False),
            port=raw.get("port", 8123),
        )
        return model

    def validate(self):
        if not path.exists(self.model_file):
            logging.warn(f"{self.model_file} does not exist")
            # raise ValueError(f"{self.model_file} does not exist")

    def as_dict(self):
        return asdict(self)


def load_scores(scores):
    scores.setdefault("default", ModelConfig.DEFAULT_SCORE)
    return scores
