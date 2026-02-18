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

from config.timewindow import WindowStatus, TimeWindow, RelAbsTime
from config.config import Config
from config.thermalconfig import ThermalConfig
from .headerinfo import HeaderInfo
from ml_tools.logs import init_logging
from ml_tools.rectangle import Rectangle
from .piclassifier import PiClassifier, run_classifier
from .cameras import lepton3
import multiprocessing
from .eventreporter import log_event
from piclassifier.monitorconfig import monitor_file
from pathlib import Path
import subprocess
from piclassifier import utils

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
        return parse_file(
            args.file, config, thermal_config, args.preview_type, args.fps
        )

    process_queue = multiprocessing.Queue()

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

    # start relevenet services

    model = None
    for model_config in config.classify.models:
        if model_config.type != "RandomForest":
            model = model_config
            break

    # try not run classifier unless we are inside a recording window
    enable_network_classifier = thermal_config.motion.postprocess or (
        model is not None and thermal_config.motion.run_classifier
    )

    if thermal_config.recorder.rec_window.inside_window() and enable_network_classifier:
        success = utils.startup_network_classifier(model.run_over_network)
        if not success:
            raise Exception("Could not start up network classifier")
    else:
        utils.stop_network_classifier()

    success = utils.startup_postprocessor(thermal_config.motion.postprocess)
    if not success and thermal_config.motion.postprocess:
        raise Exception("Could not start up postprocessor")
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
            log_event("camera-connected", {"type": "thermal"})
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
        logging.info("Not connected so closing")
        os._exit(0)


def parse_file(file, config, thermal_config, preview_type, fps):
    _, ext = os.path.splitext(file)
    thermal_config.recorder.rec_window = rec_window = TimeWindow(
        RelAbsTime(""), RelAbsTime(""), None, None, 0
    )

    if ext == ".cptv":
        parse_cptv(file, config, thermal_config.config_file, preview_type, fps)
    else:
        parse_ir(file, config, thermal_config, preview_type, fps)


def parse_ir(file, config, thermal_config, preview_type, fps):
    from piclassifier import irmotiondetector
    import cv2

    irmotiondetector.MIN_FRAMES = 0
    count = 0
    vidcap = cv2.VideoCapture(file)
    while True:
        if fps is not None:
            time.sleep(1 / fps)
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


def preview_socket(headers, frame_queue):
    import yaml

    # convert casing
    python_dic = headers.__dict__
    go_dic = {}
    for k, v in python_dic.items():
        new_key = f"{k[0].upper()}{k[1:]}"
        try:
            under_index = new_key.index("_")
            new_key = f"{new_key[:under_index]}{new_key[under_index+1].upper()}{new_key[under_index+2:]}"
        except:
            pass
        go_dic[new_key] = v
    header_bytes = yaml.dump(go_dic).encode()
    header_bytes += b"\nclear"

    while True:
        try:
            # connect to management socket
            frameSocket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            frameSocket.connect("/var/spool/managementd")
            logging.info("Connected to management interface")
            frameSocket.send(header_bytes)
            telemetry_bytes = bytearray(640)
            # if we need this can add the correct info
            while True:
                frame = frame_queue.get()
                if isinstance(frame, str):
                    if frame == STOP_SIGNAL:
                        return
                if frame is None:
                    logging.info("Disconnected")
                    break
                frame_bytes = frame.pix.byteswap().tobytes()
                frame_bytes = telemetry_bytes + frame_bytes
                frameSocket.send(frame_bytes)
        except:
            # logging.error("Failed to connect to /var/spool/managementd", exc_info=True)
            try:
                # empty the queue
                items = frame_queue.qsize()
                items = max(items, 1)
                for _ in range(items):
                    item = frame_queue.get(100)
                    if isinstance(item, str):
                        if item == STOP_SIGNAL:
                            return
            except:
                pass
            # could not connect wait a few seconds
            time.sleep(2)


def parse_cptv(file, config, thermal_config_file, preview_type, fps):
    from cptv import Frame
    from cptv_rs_python_bindings import CptvReader

    reader = CptvReader(str(file))
    header = reader.get_header()
    telemetry_size = 160 * 4
    headers = HeaderInfo(
        res_x=header.x_resolution,
        res_y=header.y_resolution,
        fps=9,
        brand=header.brand if header.brand else None,
        model=header.model if header.model else None,
        frame_size=header.x_resolution * header.y_resolution * 2 + telemetry_size,
        pixel_bits=16,
        serial="",
        firmware="",
    )

    frame_queue = multiprocessing.Queue()
    preview_process = multiprocessing.Process(
        target=preview_socket,
        args=(
            headers,
            frame_queue,
        ),
    )
    preview_process.start()
    try:
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
        while True:
            frame = reader.next_frame()

            if frame is None:
                break
            # to get extra properties and allow pickling convert to cptv.Frame
            frame = Frame(
                frame.pix,
                timedelta(milliseconds=frame.time_on),
                timedelta(milliseconds=frame.last_ffc_time),
                frame.temp_c,
                frame.last_ffc_temp_c,
                frame.background_frame,
            )

            frame_queue.put(frame)

            frame.ffc_imminent = False
            frame.ffc_status = 0

            if frame.background_frame:
                pi_classifier.motion_detector._background._background = frame.pix
                continue
            pi_classifier.process_frame(frame, time.time())
            if fps is not None:
                time.sleep(1.0 / fps)

        pi_classifier.disconnected()
        frame_queue.put(STOP_SIGNAL)
        preview_process.join(7)
        if preview_process.is_alive():
            logging.info("Killing preview process")
            try:
                kill_process_with_timeout(preview_process)
            except:
                pass
    except Exception as ex:
        pi_classifier.disconnected()
        logging.error("EXception all done")
        frame_queue.put(STOP_SIGNAL)
        preview_process.join(7)
        if preview_process.is_alive():
            logging.info("Killing preview process")
            try:
                kill_process_with_timeout(preview_process)
            except:
                pass
        raise ex


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
        logging.info("Received %s",len(data))
        if not data:
            raise Exception("Disconnected from camera while getting headers")
        headers += data
        done = headers.find(b"\n\n")
        if done > -1:
            logging.info("Headers %s done %s ",headers,done)
            # need the clear message
            left_over = headers[done + 2 :]
            headers = headers[:done]

            # ensure we handle the clear message
            if len(left_over) < 5:
                left_over += connection.recv(5-len(left_over))
                
            if left_over[:5] == b"clear":
                left_over = left_over[5:]
            break
    header_s = headers.decode()
    logging.info("header is %s ",header_s)
    return HeaderInfo.parse_header(header_s), left_over


def ir_camera(config, thermal_config, process_queue):
    import cv2

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
            model="IR",
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
                logging.info(
                    "Restarting as config changed QSize is %s", process_queue.qsize()
                )
                process_queue.put(STOP_SIGNAL)
                # give it time to clean up
                processor.join(5)
                if processor.is_alive():
                    logging.info("Killing process")
                    try:
                        kill_process_with_timeout(processor)
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
                log_event("camera-connected", {"type": "IR"})
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
            kill_process_with_timeout(processor)

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


def kill_process_with_timeout(process, timeout=30):
    # for some reason process.kill hangs sometimes
    kill_thread = Thread(target=kill_process, args=(process,))
    kill_thread.start()
    try:
        kill_thread.join(timeout)
    except:
        logging.error("Kill thread didnt terminate", exc_info=True)


def kill_process(process):
    logging.info("Killing process %s", process.pid)
    try:
        parent = psutil.Process(process.pid)
        children = parent.children()
        for child in children:
            if child.is_running:
                kill_process(child)
        psutil.wait_procs(children, timeout=5)
        if parent.is_running:
            try:
                parent.kill()
            except:
                logging.error("Could not kill process %s ", parent.pid, exc_info=True)
            parent.wait(5)

    except:
        logging.error("Could not kill process", exc_info=True)


def get_uint32(raw, offset):
    return (
        raw[offset + 1]
        | (raw[offset] << 8)
        | (raw[offset + 3] << 16)
        | (raw[offset + 2] << 24)
    )


gzip_header = bytes([0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF])

STATIC_LZ77_DYNAMIC_BLOCK_HEADER = bytes(
    [
        5,
        224,
        3,
        152,
        109,
        105,
        182,
        6,
        106,
        190,
        255,
        24,
        107,
        70,
        196,
        222,
        153,
        89,
        85,
        231,
        182,
        109,
        219,
        182,
        109,
        219,
        182,
        109,
        219,
        182,
        109,
        219,
        182,
        109,
        119,
        223,
        83,
        200,
        220,
        59,
        34,
        230,
        26,
        227,
        235,
        7,
    ]
)


import zlib
import io
import struct
import gzip


def decompress(decompressor, data, read_header=False):
    
    fp = io.BytesIO(data)
    if not read_header:
        result = gzip._read_gzip_header(fp)
        if result is None:
            logging.info("Couldn't read header")
            return data, b"", read_header
        data = data[fp.tell() :]
        read_header = True
        logging.info("Read header")
    try:
        decompressed = decompressor.decompress(data)
    except:
        logging.error("Error decompressing ",exc_info=True)
        return data,b"",read_header
    unused_data = decompressor.unused_data[8:].lstrip(b"\x00")

    # print("Tell is no0w ", fp.tell()," Unused data is " , len(decompressor.unused_data), " decompressed is ",len(decompressed))
    if not decompressor.eof or len(decompressor.unused_data) < 8:
        print("Reach eof")
        # 1/0
        return unused_data, decompressed,read_header
        raise EOFError(
            "Compressed file ended before the end-of-stream " "marker was reached"
        )
    crc, length = struct.unpack("<II", decompressor.unused_data[:8])

    if crc != zlib.crc32(decompressed):
        logging.error("CRC error")
        return unused_data, decompressed,read_header

        raise Exception("CRC check failed")
    if length != (len(decompressed) & 0xFFFFFFFF):
        raise Exception("Incorrect length of data produced")
    return unused_data, decompressed,read_header


def medium_power(thermal_config, config, connection, headers, extra_b):
    from cptv_rs_python_bindings import CptvStreamReader

    from ml_tools.imageprocessing import normalize
    import cv2
    import zlib
    logging.info("GOt header running medium power extra size %s: %s",len(extra_b),extra_b[:50])
    # pi_classifier = PiClassifier(
    #     config,
    #     thermal_config,
    #     headers,
    #     thermal_config.motion.run_classifier,
    #     0,
    #     None,
    # )
    reader = CptvStreamReader()
    decompressor = zlib.decompressobj(wbits=-zlib.MAX_WBITS)

    # decompressed_chunk = decompressor.decompress(gzip_header)
    # decompressed_chunk = decompressor.decompress(STATIC_LZ77_DYNAMIC_BLOCK_HEADER

    u8_data = None
    frame_i = 0
    read_header = False
    connection.settimeout(5)
    data = b""
    finished = False
    logging.info("Headers frame size is %s extra b size is %s",headers.frame_size,len(extra_b))
    f = open("/home/pi/streamed/raw.gz","wb")
    while not finished:
        byte_data = b""
        try:
            if extra_b is not None:
                byte_data = extra_b + connection.recv(headers.frame_size - len(extra_b))#,socket.MSG_WAITALL)
                extra_b = None
            else:
                byte_data = connection.recv(headers.frame_size)#,socket.MSG_WAITALL)
        except TimeoutError as e:
            logging.info("TImed out")
            time.sleep(1)
            # continue
        except:
            logging.error("No data resetting data",exc_info=True)
            byte_data = b""
            data = b""
            time.sleep(1)
            continue
        # if len(byte_data)==0 and  not read_header :
        #     continue
        # if len(byte_data)==0 and read_header:
        #     logging.info("Finished closing file")
        if len(byte_data)==0:
            continue
        
        f.write(byte_data)
        if byte_data[-5:] == b"clear":
            logging.info("Received clear finished file")
            finished = True
            f.close()
        else:
            data = data + byte_data


        logging.info("Have data %s %s  ",len(data),data[:10])
        try:
            data, decompressed_chunk,read_header = decompress(decompressor, data, read_header)
        except:
            logging.error("Error decompressing ", exc_info=True)
            time.sleep(1)
            continue

        logging.info("Have decompressed %s left over data is %s", len(decompressed_chunk), len(data))
        if len(decompressed_chunk) ==0:
            continue
        

        if u8_data is None:
            u8_data = np.frombuffer(decompressed_chunk, dtype=np.uint8)
        else:
            u8_data = np.concatenate(
                (u8_data, np.frombuffer(decompressed_chunk, dtype=np.uint8)), axis=0
            )

        logging.info("Loading frames wtih %s",len(u8_data))
        while True:
            result = reader.next_frame_from_data(u8_data)
            if result is not None:
                frame, used = result
                u8_data = u8_data[used:]
                logging.info("Loaded a frame from gz data")
                frame_i += 1
                normed, _ = normalize(frame.pix, new_max=255)
                normed = np.uint8(normed)
                cv2.imwrite(f"/home/pi/streamed/frame{frame_i}.png", normed)
            else:
                # need more data
                logging.info("Need more data have %s", len(u8_data))
                break

        

def handle_connection(connection, config, thermal_config_file, process_queue):
    headers, extra_b = handle_headers(connection)
    thermal_config = ThermalConfig.load_from_file(thermal_config_file, headers.model)

    if True or headers.medium_power:
        medium_power(thermal_config, config, connection, headers, extra_b)

        # do things
    logging.info(
        "parsed camera headers %s running with config %s", headers, thermal_config
    )

    processor = get_processor(process_queue, config, thermal_config, headers)
    processor.start()

    edge = config.tracking["thermal"].edge_pixels
    crop_rectangle = Rectangle(
        edge, edge, headers.res_x - 2 * edge, headers.res_y - 2 * edge
    )
    raw_frame = lepton3.Lepton3(headers)
    read = 0
    try:
        while True:
            if restart_pending:
                logging.info("Restarting as config changed")
                break
            if not processor.is_alive():
                # this potentially loops on indefinately on an error if the error is to do with the headers
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

    finally:
        if processor.is_alive:
            process_queue.put(STOP_SIGNAL)
            # give it time to clean up, seems to take a while if classifier is running
            processor.join(50)
            if processor.is_alive():
                logging.info("Killing process")
                try:
                    kill_process_with_timeout(processor)
                except:
                    pass
