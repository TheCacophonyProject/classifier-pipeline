#!/usr/bin/python3
import argparse
from datetime import datetime, timedelta
import logging
import os
import psutil
import socket
import time
import cv2
import json
import sys

# fixes logging not showing up in tensorflow

from cptv import CPTVReader
import numpy as np
from threading import Thread

from config.timewindow import WindowStatus, TimeWindow, RelAbsTime
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
from .eventreporter import log_event
from track.irtrackextractor import IRTrackExtractor
from track.cliptrackextractor import ClipTrackExtractor

from piclassifier.monitorconfig import monitor_file


SOCKET_NAME = "/var/run/lepton-frames"
VOSPI_DATA_SIZE = 160
TELEMETRY_PACKET_COUNT = 4
STOP_SIGNAL = "stop"

SKIP_SIGNAL = "skip"
SNAPSHOT_SIGNAL = "snap"

restart_pending = False
connected = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="a test file to send", default=None)
    parser.add_argument(
        "-p",
        "--preview-type",
        help="Create MP4 previews of this type",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        help="Path to config file to use",
    )

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
    thermal_config = ThermalConfig.load_from_file(args.thermal_config_file)
    monitor_thread = Thread(
        target=monitor_file, args=(file_changed, thermal_config.config_file)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    thermal_config.recorder.rec_window.set_location(
        *thermal_config.location.get_lat_long(use_default=True),
        thermal_config.location.altitude,
    )

    if args.file:
        return parse_file(args.file, config, thermal_config, args.preview_type)

    process_queue = multiprocessing.Queue()

    # get a cloned window so we dont update it
    snapshot_thread = Thread(
        target=take_snapshots,
        args=(
            thermal_config.recorder.rec_window.clone(),
            process_queue,
        ),
    )
    snapshot_thread.daemon = True
    snapshot_thread.start()
    if args.ir or thermal_config.device_setup.ir:
        while True:
            if restart_pending:
                break
            try:
                read = ir_camera(config, thermal_config, process_queue)
                if read == 0:
                    logging.error("Error reading camera try again in 10")
                    time.sleep(10)
                else:
                    log_event("camera-disconnected", f"read {read} frames")
            except Exception as ex:
                log_event("camera-disconnected", ex)
                logging.error("Error reading camera try again in 10", exc_info=True)
                time.sleep(10)
        return

    try:
        os.unlink(SOCKET_NAME)
    except OSError:
        if os.path.exists(SOCKET_NAME):
            raise
    logging.info("running as thermal")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_NAME)
    sock.settimeout(3 * 60)  # 3 minutes
    sock.listen(1)

    global connected
    while True:
        if restart_pending:
            sock.close()
            break
        logging.info("waiting for a connection")
        try:
            connection, client_address = sock.accept()
            connected = True
            logging.info("connection from %s", client_address)
            log_event("camera-connected", {"type": ClipTrackExtractor.TYPE})
            handle_connection(
                connection, config, args.thermal_config_file, process_queue
            )
        except socket.timeout:
            logging.error("Socket %s timeout error", SOCKET_NAME, exc_info=True)
            return

        except Exception as ex:
            log_event("camera-disconnected", ex)
            logging.error("Error with connection", exc_info=True)
        finally:
            # Clean up the connection
            try:
                connection.close()
            except:
                pass
        connected = False


def file_changed(event):
    logging.info("Received file changed event %s restarting", event)
    global restart_pending
    restart_pending = True
    if not connected:
        logging.info("Not conencted so closing")
        os._exit(0)


def parse_file(file, config, thermal_config, preview_type):
    _, ext = os.path.splitext(file)
    thermal_config.recorder.rec_window = rec_window = TimeWindow(
        RelAbsTime(""),
        RelAbsTime(""),
    )

    if ext == ".cptv":
        parse_cptv(file, config, thermal_config.config_file, preview_type)
    else:
        parse_ir(file, config, thermal_config, preview_type)


def parse_ir(file, config, thermal_config, preview_type):
    from piclassifier import irmotiondetector

    irmotiondetector.MIN_FRAMES = 0
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
                model=IRTrackExtractor.TYPE,
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
            pi_classifier.motion_detector._background.update_background(
                gray, learning_rate=1
            )
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
    headers = b""
    left_over = None
    while True:

        logging.info("Getting header info")
        data = connection.recv(4096)

        if not data:
            raise Exception("Disconnected from camera while getting headers")
        headers += data
        done = headers.find(b"\n\n")
        if done > -1:
            left_over = headers[done + 2 :]
            headers = headers[:done]
            if left_over[:5] == b"clear":
                left_over = left_over[5:]
            break
    headers = headers.decode()
    return HeaderInfo.parse_header(headers), left_over


def ir_camera(config, thermal_config, process_queue):
    FPS = 10
    logging.info("Starting ir video capture")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    frames = 0
    try:
        res_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        res_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        headers = HeaderInfo(
            res_x=int(res_x),
            res_y=int(res_y),
            fps=FPS,
            brand=None,
            model=IRTrackExtractor.TYPE,
            frame_size=res_y * res_x,
            pixel_bits=8,
            serial="",
            firmware="",
        )

        global connected
        connected = True
        processor = get_processor(process_queue, config, thermal_config, headers)
        processor.start()
        drop_frame = None
        dropped = 0
        start_dropping = None

        while True:
            if restart_pending:
                logging.info("Restarting as config changed")
                process_queue.put(STOP_SIGNAL)
                # give it time to clean up
                processor.join(5)
                if processor.is_alive():
                    logging.info("Killing process")
                    try:
                        processor.kill()
                    except:
                        pass
                break
            returned, frame = cap.read()
            if not processor.is_alive():
                logging.info("Processor stopped, restarting %s", processor.is_alive())
                processor = get_processor(
                    process_queue, config, thermal_config, headers
                )
                processor.start()
            if not returned:
                logging.info("no frame from video capture")
                process_queue.put(STOP_SIGNAL)
                break
            frames += 1
            if frames == 1:
                log_event("camera-connected", {"type": IRTrackExtractor.TYPE})
            if drop_frame is not None and (frames - start_dropping) % drop_frame == 0:
                logging.info("Dropping frame due to slow processing")
                dropped += 1
            else:
                process_queue.put((frame, time.time()))
            qsize = process_queue.qsize()
            if qsize > headers.fps * 4 and (
                drop_frame is None or frames > (start_dropping + drop_frame)
            ):
                # drop every 9th frame
                if drop_frame is None:
                    drop_frame = 9
                else:
                    drop_frame = drop_frame - 1
                # drop first frame
                start_dropping = frames + 1
                logging.info("Dropping every %s frame as qsize %s", drop_frame, qsize)
            elif qsize < headers.fps * 3:
                drop_frame = None
                start_dropping = None
    finally:
        if processor is not None:
            time.sleep(5)
            processor.kill()
    return frames


def next_snapshot(window, prev_window_type=None):
    current_status = None
    if prev_window_type is None:
        current_status = window.window_status()
    if window.non_stop:
        if prev_window_type is not None:
            window.next_window()
        return (window.start.dt, WindowStatus.non_stop)
    if current_status == WindowStatus.before or (
        prev_window_type == WindowStatus.after
    ):
        started = window.next_start()
        return (window.next_start(), WindowStatus.before)
    elif not window.non_stop and (
        current_status == WindowStatus.inside or prev_window_type == WindowStatus.before
    ):
        started = window.next_start()
        if current_status is not None and started - datetime.now() < timedelta(
            minutes=30
        ):
            logging.info("Started inside window within 30 mins")
            return (started, WindowStatus.before)

        return (window.next_end(), WindowStatus.inside)
    else:
        # next windowtimes
        window.next_window()
        return (window.next_start(), WindowStatus.before)


def take_snapshots(window, process_queue):
    if window.non_stop:
        window.start.dt = datetime.now()
        window.end.dt = datetime.now()
    next_snap = next_snapshot(window, None)
    while True:
        if next_snap is None:
            snap_time = datetime.now()
        else:
            snap_time = next_snap[0] - timedelta(minutes=2)
        time_until = (snap_time - datetime.now()).total_seconds()
        if time_until > 0:
            logging.info("Taking snapshot at %s", snap_time)
            time.sleep(time_until)
        logging.info("Sending snapshot signal")
        process_queue.put(SNAPSHOT_SIGNAL)
        next_snap = next_snapshot(window, next_snap[1])


def handle_connection(connection, config, thermal_config_file, process_queue):
    headers, extra_b = handle_headers(connection)
    thermal_config = ThermalConfig.load_from_file(thermal_config_file, headers.model)
    logging.info(
        "parsed camera headers %s running with config %s", headers, thermal_config
    )

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
            if restart_pending:
                logging.info("Restarting as config changed")
                process_queue.put(STOP_SIGNAL)
                # give it time to clean up
                processor.join(5)
                if processor.is_alive():
                    logging.info("Killing process")
                    try:
                        processor.kill()
                    except:
                        pass
                break
            if not processor.is_alive():
                logging.info("Processor stopped restarting")
                processor = get_processor(
                    process_queue, config, thermal_config, headers
                )
                processor.start()
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
                message = data[:5]
                if message == b"clear":
                    logging.info(
                        "processing error from camera"
                    )  # TODO Check if this is handled properly.
                    process_queue.put(STOP_SIGNAL)
                    break
            except:
                pass
            read += 1
            frame = raw_frame.parse(data)
            frame.received_at = time.time()
            cropped_frame = crop_rectangle.subimage(frame.pix)
            t_min = np.amin(cropped_frame)
            # seems to happen if pi is working hard
            if t_min == 0:
                logging.warning(
                    "received frame has odd values skipping thermal frame min {} cpu % {} memory % {}".format(
                        t_min, psutil.cpu_percent(), psutil.virtual_memory()[2]
                    )
                )
                log_event("bad-thermal-frame", f"Bad Pixel of {t_min}")
                process_queue.put(SKIP_SIGNAL)
            elif read < 100:
                process_queue.put(SKIP_SIGNAL)
            else:
                process_queue.put((frame, time.time()))

    finally:
        time.sleep(5)
        # give it a moment to close down properly
        processor.terminate()
