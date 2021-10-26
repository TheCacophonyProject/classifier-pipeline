#!/usr/bin/python3
import argparse
from datetime import datetime
import logging
import os
import psutil
import socket
import time

# fixes logging not showing up in tensorflow

from cptv import CPTVReader
import numpy as np
import json

from config.config import Config
from config.thermalconfig import ThermalConfig
from .cptvrecorder import CPTVRecorder
from .headerinfo import HeaderInfo
from ml_tools.logs import init_logging
from ml_tools import tools
from .motiondetector import MotionDetector
from .piclassifier import PiClassifier, run_classifier
from .cameras import lepton3
import multiprocessing

SOCKET_NAME = "/var/run/lepton-frames"
VOSPI_DATA_SIZE = 160
TELEMETRY_PACKET_COUNT = 4
STOP_SIGNAL = "stop"

SKIP_SIGNAL = "skip"


# TODO abstract interpreter class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cptv", help="a CPTV file to send", default=None)
    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument(
        "--thermal-config-file", help="Path to pi-config file (config.toml) to use"
    )

    args = parser.parse_args()
    return args


# Links to socket and continuously waits for 1 connection
def main():
    init_logging()
    args = parse_args()

    config = Config.load_from_file(args.config_file)
    thermal_config = ThermalConfig.load_from_file(args.thermal_config_file)

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
        except:
            logging.error("Error with connection", exc_info=True)
            # return
        finally:
            # Clean up the connection
            try:
                connection.close()
            except:
                pass


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
        process_queue = multiprocessing.Queue()

        p = get_processor(process_queue, config, thermal_config, headers)
        p.start()
        for frame in reader:
            process_queue.put(frame)
        process_queue.put(STOP_SIGNAL)
        p.join()


def get_processor(process_queue, config, thermal_config, headers):
    p_processor = multiprocessing.Process(
        target=run_classifier,
        args=(
            process_queue,
            config,
            thermal_config,
            headers,
            thermal_config.motion.run_classifier,
        ),
    )
    return p_processor


def handle_headers(connection):
    headers = ""
    left_over = None
    while True:
        data = connection.recv(4096).decode()
        headers += data
        done = headers.find("\n\n")
        if done > -1:
            headers = headers[:done]
            left_over = headers[done + 2 :].encode()
            break
    return HeaderInfo.parse_header(headers), left_over


def handle_connection(connection, config, thermal_config):
    headers, extra_b = handle_headers(connection)
    logging.info("parsed camera headers %s", headers)
    process_queue = multiprocessing.Queue()

    processor = get_processor(process_queue, config, thermal_config, headers)
    processor.start()

    edge = config.tracking.edge_pixels
    crop_rectangle = tools.Rectangle(
        edge, edge, headers.res_x - 2 * edge, headers.res_y - 2 * edge
    )
    raw_frame = lepton3.Lepton3(headers)
    read = 0
    try:
        while True:
            if extra_b is not None:
                data = extra_b + connection.recv(
                    headers.frame_size - len(extra_b), socket.MSG_WAITALL
                )
                extra_b = None
            else:
                data = connection.recv(headers.frame_size, socket.MSG_WAITALL)

            if not data:
                logging.info("disconnected from camera")
                process_queue.put(STOP_SIGNAL)
                break
            try:
                message = data[:5].decode("utf-8")
                if message == "clear":
                    logging.info("processing error from camera")
                    process_queue.put(STOP_SIGNAL)
                    break
            except:
                pass
            read += 1
            frame = raw_frame.parse(data)
            cropped_frame = crop_rectangle.subimage(frame.pix)
            t_max = np.amax(cropped_frame)
            t_min = np.amin(cropped_frame)
            # seems to happen if pi is working hard
            if t_min == 0:
                logging.warning(
                    "received frame has odd values skipping thermal frame max {} thermal frame min {} cpu % {} memory % {}".format(
                        t_max, t_min, psutil.cpu_percent(), psutil.virtual_memory()[2]
                    )
                )
                process_queue.put(SKIP_SIGNAL)
            elif read < 100:
                process_queue.put(SKIP_SIGNAL)
            else:
                process_queue.put(frame)
    finally:
        time.sleep(5)
        # give it a moment to close down properly
        processor.terminate()
