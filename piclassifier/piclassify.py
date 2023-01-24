#!/usr/bin/python3
import argparse
from datetime import datetime
import logging
import os
import psutil
import socket
import time
import cv2

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
from .cameras.irframe import IRFrame
import multiprocessing
from cptv import Frame


SOCKET_NAME = "/var/run/lepton-frames"
VOSPI_DATA_SIZE = 160
TELEMETRY_PACKET_COUNT = 4
STOP_SIGNAL = "stop"

SKIP_SIGNAL = "skip"


# TODO abstract interpreter class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="a test file to send", default=None)
    parser.add_argument(
        "-p",
        "--preview-type",
        help="Create MP4 previews of this type",
    )
    parser.add_argument("-c", "--config-file", help="Path to config file to use")

    parser.add_argument(
        "--thermal-config-file", help="Path to pi-config file (config.toml) to use"
    )
    parser.add_argument(
        "--ir", action="count", help="Path to pi-config file (config.toml) to use"
    )
    args = parser.parse_args()
    return args


# Links to socket and continuously waits for 1 connection
def main():
    init_logging()
    args = parse_args()

    config = Config.load_from_file(args.config_file)
    if args.file:
        return parse_file(
            args.file, config, args.thermal_config_file, args.preview_type
        )
    if args.ir:
        while True:
            try:
                ir_camera(config, args.thermal_config_file)
            except:
                logging.error("Error reading camera", exc_info=True)
                time.sleep(10)
        return
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
            handle_connection(connection, config, args.thermal_config_file)
        except:
            logging.error("Error with connection", exc_info=True)
            # return
        finally:
            # Clean up the connection
            try:
                connection.close()
            except:
                pass


def parse_file(file, config, thermal_config_file, preview_type):
    _, ext = os.path.splitext(file)

    if ext == ".cptv":
        parse_cptv(file, config, thermal_config_file, preview_type)
    else:
        parse_ir(file, config, thermal_config_file, preview_type)


def parse_ir(file, config, thermal_config_file, preview_type):

    thermal_config = ThermalConfig.load_from_file(thermal_config_file, "IR")
    count = 0
    vidcap = cv2.VideoCapture(file)
    while True:
        success, image = vidcap.read()
        if not success:
            break
        # gray = cv2.resize(gray, (640, 480))
        if count == 0:
            res_y, res_x = image.shape[:2]
            headers = HeaderInfo(
                res_x=res_x,
                res_y=res_y,
                fps=10,
                brand=None,
                model="IR",
                frame_size=res_y * res_x,
                pixel_bits=8,
                serial="",
                firmware="",
            )

            pi_classifier = PiClassifier(
                config,
                thermal_config,
                headers,
                thermal_config.motion.run_classifier,
                0,
                preview_type,
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            pi_classifier.motion_detector._background._background = np.float32(gray)
            pi_classifier.motion_detector._background._frames = 1000
            count += 1
            # assume this has been run over 1000 frames
            continue
        # frame = Frame(image, None, None, None, None)
        # frame.received_at = time.time()
        pi_classifier.process_frame(image, time.time())
        count += 1
    vidcap.release()
    pi_classifier.disconnected()


def parse_cptv(file, config, thermal_config_file, preview_type):
    with open(file, "rb") as f:
        reader = CPTVReader(f)

        headers = HeaderInfo(
            res_x=reader.x_resolution,
            res_y=reader.y_resolution,
            fps=9,
            brand=reader.brand.decode() if reader.brand else None,
            model=reader.model.decode() if reader.model else None,
            frame_size=reader.x_resolution * reader.y_resolution * 2,
            pixel_bits=16,
            serial="",
            firmware="",
        )
        thermal_config = ThermalConfig.load_from_file(
            thermal_config_file, headers.model
        )
        pi_classifier = PiClassifier(
            config,
            thermal_config,
            headers,
            thermal_config.motion.run_classifier,
            0,
            preview_type,
        )
        for frame in reader:
            if frame.background_frame:
                pi_classifier.motion_detector._background._background = frame.pix
                continue
            pi_classifier.process_frame(frame, time.time())
        pi_classifier.disconnected()


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


def ir_camera(config, thermal_config_file):
    FPS = 10
    logging.info("Starting ir video capture")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    try:
        res_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        res_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        headers = HeaderInfo(
            res_x=int(res_x),
            res_y=int(res_y),
            fps=FPS,
            brand=None,
            model="IR",
            frame_size=res_y * res_x,
            pixel_bits=8,
            serial="",
            firmware="",
        )
        thermal_config = ThermalConfig.load_from_file(
            thermal_config_file, headers.model
        )
        process_queue = multiprocessing.Queue()

        processor = get_processor(process_queue, config, thermal_config, headers)
        processor.start()
        while True:
            returned, frame = cap.read()

            if not returned:
                logging.info("no frame from video capture")
                process_queue.put(STOP_SIGNAL)

                break
            process_queue.put((frame, time.time()))
    finally:
        time.sleep(5)
        processor.terminate()


def handle_connection(connection, config, thermal_config_file):
    headers, extra_b = handle_headers(connection)
    thermal_config = ThermalConfig.load_from_file(thermal_config_file, headers.model)
    logging.info(
        "parsed camera headers %s running with config %s", headers, thermal_config
    )

    process_queue = multiprocessing.Queue()

    processor = get_processor(process_queue, config, thermal_config, headers)
    processor.start()

    edge = config.tracking["thermal"].edge_pixels
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
            frame.received_at = time.time()
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
                process_queue.put((frame, time.time()))
    finally:
        time.sleep(5)
        # give it a moment to close down properly
        processor.terminate()
