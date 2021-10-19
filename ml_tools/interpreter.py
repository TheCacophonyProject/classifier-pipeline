import json
import logging

from ml_tools.hyperparams import HyperParams


class Interpreter:
    def __init__(self, model_file):
        self.load_json(model_file)

    def load_json(self, filename):
        """Loads model and parameters from file."""
        logging.info("Loading metadata from %s.txt", filename)
        stats = json.load(open(filename + ".txt", "r"))

        self.model_name = stats["name"]
        self.model_description = stats["description"]
        self.labels = stats["labels"]
        self.params = HyperParams()
        self.params.update(stats["hyperparams"])
