from abc import ABC, abstractmethod

import json
import logging

from ml_tools.hyperparams import HyperParams
from pathlib import Path


class Interpreter(ABC):
    def __init__(self, model_file):
        self.load_json(model_file)

    def load_json(self, filename):
        """Loads model and parameters from file."""
        filename = Path(filename)
        filename = filename.with_suffix(".txt")
        logging.info("Loading metadata from %s", filename)
        stats = json.load(open(filename, "r"))

        self.labels = stats["labels"]
        self.params = HyperParams()
        self.params.update(stats.get("hyperparams", {}))

    @abstractmethod
    def shape(self):
        """Prediction shape"""
        ...
