#!/usr/bin/python3
import absl.logging
import argparse
from datetime import datetime
import logging
import os
import psutil
import socket

# fixes logging not showing up in tensorflow

from cptv import CPTVReader
import numpy as np

from config.config import Config
from config.thermalconfig import ThermalConfig
from .cptvrecorder import CPTVRecorder
from .headerinfo import HeaderInfo
from ml_tools.logs import init_logging
from ml_tools import tools
from .motiondetector import MotionDetector
from .piclassifier import PiClassifier
from service import SnapshotService
from .cameras import lepton3

SOCKET_NAME = "/var/run/lepton-frames"
VOSPI_DATA_SIZE = 160
TELEMETRY_PACKET_COUNT = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cptv", help="a CPTV file to send", default=None)
    args = parser.parse_args()
    return args


def get_classifier(config):
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

    if args.cptv:
        return parse_cptv(args.cptv, config, thermal_config)

    try:
        os.unlink(SOCKET_NAME)
    except OSError:
        if os.path.exists(SOCKET_NAME):
            raise

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_NAME)
    sock.listen(1)
    while True:
        logging.info("waiting for a connection")
        connection, client_address = sock.accept()
        logging.info("connection from %s", client_address)
        try:
            handle_connection(connection, config, thermal_config)
        finally:
            # Clean up the connection
            connection.close()


def parse_cptv(cptv_file, config, thermal_config):
    with open(cptv_file, "rb") as f:
        reader = CPTVReader(f)

        headers = HeaderInfo(
            res_x=reader.x_resolution,
            res_y=reader.y_resolution,
            fps=9,
            brand="",
            model="",
            frame_size=reader.x_resolution * reader.y_resolution * 2,
            pixel_bits=16,
        )
        processor = get_processor(config, thermal_config, headers)
        for frame in reader:
            processor.process_frame(frame)

        processor.disconnected()


def get_processor(config, thermal_config, headers):
    if thermal_config.motion.run_classifier:
        classifier = get_classifier(config)
        return PiClassifier(config, thermal_config, classifier, headers)

    return MotionDetector(
        thermal_config,
        config.tracking.dynamic_thresh,
        CPTVRecorder(thermal_config, headers),
        headers,
    )


def handle_headers(connection):
    headers = ""
    line = ""
    while True:
        data = connection.recv(1).decode()

        line += data
        if data == "\n":
            if line.strip() == "":
                break

            headers += line
            line = ""
    return HeaderInfo.parse_header(headers)


def handle_connection(connection, config, thermal_config):
    headers = handle_headers(connection)
    logging.debug("parsed camera headers", headers)
    processor = get_processor(config, thermal_config, headers)
    service = SnapshotService(processor)

    raw_frame = lepton3.Lepton3(headers)

    while True:
        data = connection.recv(
            headers.frame_size + raw_frame.get_telemetry_size(), socket.MSG_WAITALL
        )
        if not data:
            logging.info("disconnected from camera")
            processor.disconnected()
            service.quit()
            return

        frame = raw_frame.parse(data)

        t_max = np.amax(frame.pix)
        t_min = np.amin(frame.pix)
        if t_max > 10000 or t_min == 0:
            logging.warning(
                "received frame has odd values skipping thermal frame max {} thermal frame min {} cpu % {} memory % {}".format(
                    t_max, t_min, psutil.cpu_percent(), psutil.virtual_memory()[2]
                )
            )
            # this frame has bad data probably from lack of CPU
            processor.skip_frame()
            continue
        processor.process_frame(frame)
