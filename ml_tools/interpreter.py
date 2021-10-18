import json
class Interpreter:
    def load_json(self, filename):
        """Loads model and parameters from file."""
        stats = json.load(open(filename + ".txt", "r"))

        self.MODEL_NAME = stats["name"]
        self.MODEL_DESCRIPTION = stats["description"]
        self.labels = stats["labels"]
        self.params = HyperParams()
        self.params.update(stats["hyperparams"])
