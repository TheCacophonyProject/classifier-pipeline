#!/usr/bin/python3
import argparse
from datetime import datetime, timedelta
import logging
import os
import psutil
import socket
import time

import numpy as np
from threading import Thread

from config.config import Config
from config.thermalconfig import ThermalConfig
from .headerinfo import HeaderInfo
from ml_tools.logs import init_logging
import multiprocessing
from .eventreporter import log_event
from piclassifier.monitorconfig import monitor_file
from pathlib import Path
from piclassifier import utils
from .signals import STOP_SIGNAL, SKIP_SIGNAL, SNAPSHOT_SIGNAL, PARSING_FILE, PARSED

SOCKET_NAME = "/var/run/lepton-frames"
VOSPI_DATA_SIZE = 160
TELEMETRY_PACKET_COUNT = 4

restart_pending = False
connected = False
ready_to_record = False

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
        "--seed",
        type=int,
        default=None,
        help="Seed to use for randomness, this will make predictions the same every run on a file",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="When running a file through specify the frame rate you want it to run at, otherwise it runs as fast as the cpu can",
    )

    args = parser.parse_args()
    return args


# Links to socket and continuously waits for 1 connection
def main():
    init_logging()
    args = parse_args()

    config = Config.load_from_file(args.config_file)
    thermal_config = ThermalConfig.load_from_file(args.thermal_config_file)

    thermal_config.recorder.rec_window.set_location(
        *thermal_config.location.get_lat_long(use_default=True),
        thermal_config.location.altitude,
    )
    if args.file:
        return parse_file(
            args.file, config, thermal_config, args.preview_type, args.fps, args.seed
        )


    process_queue = multiprocessing.Queue()
    response_queue = multiprocessing.Queue()

    # TODO this will break things if we ever have different resolution or FPS
    headers = default_headers()
    processor = get_processor(
        process_queue, response_queue, config, thermal_config, headers
    )
    processor.start()

    monitor_thread = Thread(
        target=monitor_file, args=(file_changed, thermal_config.config_file)
    )
    monitor_thread.daemon = True
    monitor_thread.start()


    # get a cloned window so we dont update it
    if not thermal_config.recorder.use_low_power_mode:
        snapshot_thread = Thread(
            target=take_snapshots,
            args=(
                thermal_config.recorder.rec_window.clone(),
                process_queue,
                thermal_config.recorder.output_dir,
            ),
        )
        snapshot_thread.daemon = True
        snapshot_thread.start()
    try:
        os.unlink(SOCKET_NAME)
    except OSError:
        if os.path.exists(SOCKET_NAME):
            raise
    logging.info("running as thermal")

    # try not run classifier unless we are inside a recording window
    # enable_network_classifier = (
    #     model is not None and thermal_config.motion.run_classifier
    # )

    # will start this up later, if tc2-agent is offloading recordings this can overload the system
    # best to wait until we get frames
    # if thermal_config.recorder.rec_window.inside_window() and enable_network_classifier:
    #     success = utils.toggle_network_classifier(model.run_over_network)
    #     if not success:
    #         raise Exception("Could not start up network classifier")
    # if not enable_network_classifier:
    utils.toggle_network_classifier(False)

    success = utils.startup_postprocessor(thermal_config.motion.postprocess)
    if not success and thermal_config.motion.postprocess:
        raise Exception("Could not start up postprocessor")


    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_NAME)
    sock.settimeout(1 * 60)
    sock.listen(1)
    global connected

    while True:
        if restart_pending:
            sock.close()
            logging.info("Restart pending exiting")
            break
        logging.info("waiting for a connection")
        try:
            connection, client_address = sock.accept()
            connected = True
            logging.info("connection from %s", client_address)
            log_event("camera-connected", {"type": "thermal"})

            handle_connection(
                processor,
                connection,
                config,
                args.thermal_config_file,
                process_queue,
                response_queue,
            )


        except socket.timeout:
            logging.error("Socket %s timeout error", SOCKET_NAME, exc_info=True)
            continue

        except Exception as ex:
            log_event("camera-disconnected", ex)
            logging.error("Error with connection", exc_info=True)
        finally:
            # Clean up the connection
            try:
                connection.close()
            except:
                pass
    if processor.is_alive:
        logging.info("Stopping processor because restart was pending")
        process_queue.put(STOP_SIGNAL)
        # give it time to clean up, seems to take a while if classifier is running
        processor.join(50)
        if processor.is_alive():
            logging.info("Killing process")
            try:
                utils.kill_process_with_timeout(processor)
            except:
                pass


def file_changed(event):
    logging.info("Received file changed event %s restarting", event)
    global restart_pending
    restart_pending = True
    if not connected:
        logging.info("Not connected so closing")
        os._exit(0)


def parse_file(file, config, thermal_config, preview_type, fps, seed):
    from config.timewindow import TimeWindow, RelAbsTime

    thermal_config.recorder.rec_window = rec_window = TimeWindow(
        RelAbsTime(""), RelAbsTime(""), None, None, 0
    )

    parse_cptv(file, config, thermal_config, preview_type, fps, seed)


def parse_cptv(file, config, thermal_config, preview_type, fps, seed):
    from .piclassifier import PiClassifier

    # this doesnt matter since it will get read from file later
    telemetry_size = 160 * 4
    headers = HeaderInfo(
        res_x=160,
        res_y=120,
        fps=9,
        brand=None,
        model=None,
        frame_size=120 * 160 * 2 + telemetry_size,
        pixel_bits=16,
        serial="",
        firmware="",
        source=file,
    )
    pi_classifier = PiClassifier(
        config,
        thermal_config,
        headers,
        thermal_config.motion.run_classifier,
        preview_type,
    )
    pi_classifier.parse_file(file, fps, seed)
    pi_classifier.service.quit()
    print("ALL DONE")


def get_processor(process_queue, response_queue, config, thermal_config, headers):
    from .piclassifier import run_classifier

    p_processor = multiprocessing.Process(
        target=run_classifier,
        args=(
            process_queue,
            response_queue,
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
    return HeaderInfo.parse_header(headers.decode()), left_over


def next_snapshot(window, prev_window_type=None):
    from config.timewindow import WindowStatus

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
        return (window.next_start(), WindowStatus.before)
    elif not window.non_stop and (
        current_status == WindowStatus.inside or prev_window_type == WindowStatus.before
    ):
        started = window.next_start()
        if (
            current_status is not None
            and abs((started - datetime.now()).total_seconds()) < 60 * 30
        ):
            logging.info("Started inside window within 30 mins")
            return (started, WindowStatus.before)

        return (window.next_end(), WindowStatus.inside)
    else:
        # next windowtimes
        window.next_window()
        return (window.next_start(), WindowStatus.before)


def take_snapshots(window, process_queue, output_dir):
    if window.non_stop:
        window.start.dt = datetime.now()
        window.end.dt = datetime.now()
    next_snap = next_snapshot(window, None)
    while True:
        delete_stale_thumbnails(output_dir)
        if next_snap is None:
            snap_time = datetime.now()
        else:
            snap_time = next_snap[0] - timedelta(minutes=2)
        time_until = (snap_time - datetime.now()).total_seconds()
        if time_until > 0:
            logging.info("Taking snapshot at %s", snap_time)
            time.sleep(time_until)
        while not ready_to_record:
            time.sleep(10)
        logging.info("Sending snapshot signal")
        process_queue.put(SNAPSHOT_SIGNAL)
        next_snap = next_snapshot(window, next_snap[1])


def delete_stale_thumbnails(output_dir):
    # delete all but latest clip thumbnail
    logging.info("Deleting stale thumnbnails")
    thumbnail_dir = Path(output_dir) / "thumbnails"
    thumbnail_dir.mkdir(exist_ok=True)
    for f in thumbnail_dir.iterdir():
        if f.is_file:
            f.unlink()

    # if needed can keep the last thumbnail taken, probably not nessesary
    # Need to make sure that new files are kept before the last thumb kept here
    # perhaps a metadata file or read file creation date


#     files = list(thumbnail_dir.glob(f"*.npy"))
#     files = sorted(files, key=lambda f: thumb_clip_id(f.name), reverse=True)
#     keep_id = None
#     for f in files:
#         clip_id = thumb_clip_id(f.name)
#         if keep_id is None:
#             if clip_id == -1:
#                 keep_id = 0
#                 # should delete files where clip id coult not be parsed
#             else:
#                 keep_id = clip_id
#                 logging.info("Keeping %s", keep_id)

#         if clip_id != keep_id:
#             logging.info("Deleting %s file %s", clip_id, f)
#             f.unlink()


# def thumb_clip_id(filename):
#     try:
#         hyphen = filename.index("-")
#         clip_id = filename[:hyphen]
#         return int(clip_id)
#     except:
# return -1


import fcntl, termios, struct

def bytes_queued(sock):
    buf = struct.pack('i', 0)
    return struct.unpack('i', fcntl.ioctl(sock.fileno(), termios.FIONREAD, buf))[0]

def handle_connection(
    processor, connection, config, thermal_config_file, process_queue, response_queue
):
    from .cameras import lepton3
    from ml_tools.rectangle import Rectangle
    from queue import Empty

    # sometimes the headers are never received
    connection.settimeout(20)
    headers, extra_b = handle_headers(connection)
    connection.settimeout(None)

    thermal_config = ThermalConfig.load_from_file(thermal_config_file, headers.model)
    logging.info(
        "parsed camera headers %s running with config %s", headers, thermal_config
    )
    process_queue.put(headers)
    global ready_to_record
    ready_to_record = True

    edge = config.tracking["thermal"].edge_pixels
    crop_rectangle = Rectangle(
        edge, edge, headers.res_x - 2 * edge, headers.res_y - 2 * edge
    )
    raw_frame = lepton3.Lepton3(headers)
    read = 0
    parsing_file = False
    try:
        while True:
            if restart_pending:
                logging.info("Restarting as config changed")
                break

            if not processor.is_alive():
                # this potentially loops on indefinately on an error if the error is to do with the headers
                logging.info("Processor stopped restarting")
                processor = get_processor(
                    process_queue, response_queue, config, thermal_config, headers
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
                break
            try:
                message = data[:5]
                if message == b"clear":
                    logging.info(
                        "processing error from camera"
                    )  # TODO Check if this is handled properly.
                    break
            except:
                pass
            read += 1
            # if read % 90:
            #     print_memory_usage()
            if parsing_file:
                # need to keep reading from data socket in the mean time so just do a quick check
                try:
                    message = response_queue.get(False, 0)
                    if message == PARSED:
                        parsing_file = False
                        logging.info("Finished parsing file")
                        process_queue.put(headers)
                except Empty:
                    pass
                continue
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
            else:
                process_queue.put((frame, time.time()))

            if process_queue.qsize() > 20:
                # check if there is a reason frames have slowed down
                try:
                    message = response_queue.get(False, 0)
                    if message == PARSING_FILE:
                        parsing_file = True
                        clear_queue(process_queue)
                        logging.info("Parsing file so will not process any more frames")
                except Empty:
                    pass

    except:
        logging.error("Error handling connection",exc_info=True)
    finally:
        ready_to_record = False
        if processor.is_alive:
            logging.info("Stopping processor because there was an issue in frame handling")

            process_queue.put(STOP_SIGNAL)
            # give it time to clean up, seems to take a while if classifier is running
            processor.join(50)
            if processor.is_alive():
                logging.info("Killing process")
                try:
                    utils.kill_process_with_timeout(processor)
                except:
                    pass
        clear_queue(process_queue)
        clear_queue(response_queue)

def clear_queue(q):
    """Removes all items from a multiprocessing Queue."""
    from queue import Empty

    try:
        while True:
            q.get_nowait()
    except Empty:
        pass


def default_headers():
    telemetry_size = 160 * 4

    headers = HeaderInfo(
        res_x=160,
        res_y=120,
        fps=9,
        brand=None,
        model=None,
        frame_size=120 * 160 * 2 + telemetry_size,
        pixel_bits=16,
        serial="",
        firmware="",
    )
    return headers



def print_memory_usage():
    process = psutil.Process(os.getpid())
    main_rss = process.memory_info().rss
    main_uss = process.memory_full_info().uss
    total_rss = main_rss
    total_uss = main_uss
    logging.info(
        "Memory usage pid %d (%s) %.1fMB rss %.1fMB uss",
        process.pid,
        process.name(),
        main_rss / (1024 * 1024),
        main_uss / (1024 * 1024),
    )
    children = process.children(recursive=True)
    for child in children:
        try:
            child_rss = child.memory_info().rss
            child_uss = child.memory_full_info().uss
            total_rss += child_rss
            total_uss += child_uss
            logging.info(
                "Memory usage pid %d (%s) %.1fMB rss %.1fMB uss",
                child.pid,
                child.name(),
                child_rss / (1024 * 1024),
                child_uss / (1024 * 1024),
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    swap = psutil.swap_memory()
    logging.info(
        "Memory usage main %.1fMB total (with %d sub processes) %.1fMB uss %.1fMB swap used %.1fMB of %.1fMB (%.1f%%)",
        main_rss / (1024 * 1024),
        len(children),
        total_rss / (1024 * 1024),
        total_uss / (1024 * 1024),
        swap.used / (1024 * 1024),
        swap.total / (1024 * 1024),
        swap.percent,
    )

