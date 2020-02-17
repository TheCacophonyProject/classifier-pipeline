#!/usr/bin/python3
import absl.logging
import argparse
from datetime import datetime, timedelta
import logging
import os
import psutil
import socket
import time

# fixes logging not showing up in tensorflow

from cptv import Frame, CPTVReader
import numpy as np
import json

from config.config import Config
from config.thermalconfig import ThermalConfig
from .cptvrecorder import CPTVRecorder
from ml_tools.logs import init_logging
from ml_tools import tools
from .motiondetector import MotionDetector
from .piclassifier import PiClassifier
from service import SnapshotService
from .telemetry import Telemetry

SOCKET_NAME = "/var/run/lepton-frames"
VOSPI_DATA_SIZE = 160
TELEMETRY_PACKET_COUNT = 4


class NeuralInterpreter:

    def __init__(self, model_name):
        from openvino.inference_engine import IENetwork, IECore
        print(model_name)
        device = "MYRIAD"
        model_xml = model_name +".xml"
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        ie = IECore()
        net = IENetwork(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        net.batch_size = 1
        self.exec_net = ie.load_network(network=net, device_name=device)

        # self.interpreter.allocate_tensors()
        # input_details = self.interpreter.get_input_details()

        # self.in_values = {}
        # for detail in input_details:
        #     self.in_values[detail["name"]] = detail["index"]
        # output_details = self.interpreter.get_output_details()

        # self.out_values = {}
        # for detail in output_details:
        #     self.out_values[detail["name"]] = self.interpreter.get_tensor(
        #         detail["index"]
        #     )

        self.load_json(model_name)
        # self.accuracy = self.out_values["accuracy"]
        # self.prediction = self.out_values["prediction"]
        # # interpreter.set_tensor(in_values["X"], input_data)

    def classify_frame_with_novelty(self, input_x):
        print("neural classify")
        input_x = np.array([[input_x]])
        input_x = input_x.reshape((1,48,1,5,48))
        res = self.exec_net.infer(inputs={self.input_blob: input_x})
        res = res[self.out_blob]
        print(res)
        return res[0][0], res[0][1], None

    def load_json(self, filename):
        """ Loads model and parameters from file. """
        stats = json.load(open(filename + ".txt", "r"))

        self.MODEL_NAME = stats["name"]
        self.MODEL_DESCRIPTION = stats["description"]
        self.labels = stats["labels"]
        self.eval_score = stats["score"]
        self.params = stats["hyperparams"]


class LiteInterpreter:
    # import tflite_runtime.interpreter as tflite

    def __init__(self, model_name):
        print(model_name)
        self.interpreter = tflite.Interpreter(model_path=model_name + ".tflite")

        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()

        self.in_values = {}
        for detail in input_details:
            self.in_values[detail["name"]] = detail["index"]
        output_details = self.interpreter.get_output_details()

        self.out_values = {}
        for detail in output_details:
            self.out_values[detail["name"]] = self.interpreter.get_tensor(
                detail["index"]
            )

        self.load_json(model_name)
        self.accuracy = self.out_values["accuracy"]
        self.prediction = self.out_values["prediction"]
        # interpreter.set_tensor(in_values["X"], input_data)

    def run(self, input_x):
        self.interpreter.set_tensor(self.in_values["X"], [[input_x]])
        self.interpreter.invoke()

    def classify_frame_with_novelty(self, input_x):
        self.run(input_x)
        return self.prediction, 5, None

    def load_json(self, filename):
        """ Loads model and parameters from file. """
        stats = json.load(open(filename + ".txt", "r"))

        self.MODEL_NAME = stats["name"]
        self.MODEL_DESCRIPTION = stats["description"]
        self.labels = stats["labels"]
        self.eval_score = stats["score"]
        self.params = stats["hyperparams"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cptv", help="a CPTV file to send", default=None)
    args = parser.parse_args()
    return args


def get_classifier(config):
    print("get classifier")
    model_name, model_type = os.path.splitext(config.classify.model)
    if model_type == ".tflite":
        return LiteInterpreter(model_name)
    elif model_type == ".xml":
        print("getting neural")
        return NeuralInterpreter(model_name)

    else:
        return get_full_classifier(config)


def get_full_classifier(config):
    from ml_tools.model import Model

    """
    Returns a classifier object, which is created on demand.
    This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
    """
    t0 = datetime.now()
    logging.info("classifier loading")
    classifier = Model(
        train_config=config.train,
        session=tools.get_session(disable_gpu=not config.use_gpu),
    )
    classifier.load(config.classify.model)
    logging.info("classifier loaded ({})".format(datetime.now() - t0))

    return classifier


# Links to socket and continuously waits for 1 connection
def main():
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    init_logging()
    args = parse_args()

    config = Config.load_from_file()
    thermal_config = ThermalConfig.load_from_file()
    proccesor = None
    if thermal_config.motion.run_classifier:
        classifier = get_classifier(config)
        proccesor = PiClassifier(config, thermal_config, classifier)
    else:
        proccesor = MotionDetector(
            config.res_x,
            config.res_y,
            thermal_config,
            config.tracking.dynamic_thresh,
            CPTVRecorder(thermal_config),
        )
    if args.cptv:
        with open(args.cptv, "rb") as f:
            reader = CPTVReader(f)
            for frame in reader:
                proccesor.process_frame(frame)

        proccesor.disconnected()
        return

    service = SnapshotService(proccesor)
    try:
        os.unlink(SOCKET_NAME)
    except OSError:
        if os.path.exists(SOCKET_NAME):
            raise

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sock.bind(SOCKET_NAME)
    sock.listen(1)
    while True:
        logging.info("waiting for a connection")
        connection, client_address = sock.accept()
        logging.info("connection from %s", client_address)
        try:
            handle_connection(connection, proccesor)
        finally:
            # Clean up the connection
            connection.close()


def handle_connection(connection, processor):
    img_dtype = np.dtype("uint16")
    # big endian > little endian <
    # lepton3 is big endian while python is little endian

    thermal_frame = np.empty((processor.res_y, processor.res_x), dtype=img_dtype)
    while True:
        data = connection.recv(400 * 400 * 2 + TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE)

        if not data:
            logging.info("disconnected from camera")
            processor.disconnected()
            return

        if len(data) > processor.res_y * processor.res_x * 2:
            telemetry = Telemetry.parse_telemetry(
                data[: TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE]
            )

            thermal_frame = np.frombuffer(
                data, dtype=img_dtype, offset=TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE
            ).reshape(processor.res_y, processor.res_x)
        else:
            telemetry = Telemetry()
            telemetry.last_ffc_time = timedelta(milliseconds=time.time())
            telemetry.time_on = timedelta(
                milliseconds=time.time(), seconds=MotionDetector.FFC_PERIOD.seconds + 1
            )
            thermal_frame = np.frombuffer(data, dtype=img_dtype, offset=0).reshape(
                processor.res_y, processor.res_x
            )

        # swap from big to little endian
        lepton_frame = Frame(
            thermal_frame.byteswap(), telemetry.time_on, telemetry.last_ffc_time
        )
        t_max = np.amax(lepton_frame.pix)
        t_min = np.amin(lepton_frame.pix)
        if t_max > 10000 or t_min == 0:
            logging.warning(
                "received frame has odd values skipping thermal frame max {} thermal frame min {} cpu % {} memory % {}".format(
                    t_max, t_min, psutil.cpu_percent(), psutil.virtual_memory()[2]
                )
            )
            # this frame has bad data probably from lack of CPU
            processor.skip_frame()
            continue
        processor.process_frame(lepton_frame)
