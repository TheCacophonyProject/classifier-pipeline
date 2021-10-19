import json


class Interpreter:
    def __init__(self, model_file):
        self.load_json(model_file)

    def load_json(self, filename):
        """Loads model and parameters from file."""
        stats = json.load(open(filename + ".txt", "r"))

        self.model_name = stats["name"]
        self.model_description = stats["description"]
        self.labels = stats["labels"]
        self.params = HyperParams()
        self.params.update(stats["hyperparams"])
