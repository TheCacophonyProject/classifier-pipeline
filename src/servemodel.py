from flask import Flask
from flask import request
from flask import Response
import numpy as np
import sys
from ml_tools.interpreter import LiteInterpreter
from config.config import Config
import logging
from ml_tools.logs import init_logging
from config.thermalconfig import ThermalConfig


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 407200


@app.route("/predict", methods=["POST"])
def main():
    data = request.data
    input_data = np.frombuffer(data, dtype=np.float32)
    input_data = input_data.reshape(160, 160, 3)
    prediction = np.squeeze(interpreter.predict([input_data]))
    response = Response(prediction.tostring(), mimetype="application/octet-stream")
    return response


def startup_classifier():
    # classifies an empty frame to force loading of the model into memory
    num_inputs, in_shape = interpreter.shape()
    if num_inputs > 1:
        zero_input = []
        for shape in in_shape:
            zero_input.append(np.zeros((1, *shape[1:]), np.float32))
    else:
        zero_input = np.zeros((1, *in_shape[1:]), np.float32)
    interpreter.predict(zero_input)


def get_model():

    thermal_config = ThermalConfig.load_from_file()

    if not thermal_config.motion.run_classifier:
        logging.info("Classifier isn't configured to run in config")
        return None

    config = Config.load_from_file()
    network_model = [
        model for model in config.classify.models if model.run_over_network
    ]
    if len(network_model) == 0:
        logging.info("No network model configured in classifier.yaml")
        return None
    if len(network_model) > 1:
        logging.info("Got multiple network models using first")
    return network_model[0]


if __name__ == "__main__":
    init_logging()

    network_model = get_model()
    if network_model is None:
        sys.exit(0)
    interpreter = LiteInterpreter(network_model.model_file)
    startup_classifier()
    app.run(port=network_model.port, debug=True)  # Set debug=False for production
