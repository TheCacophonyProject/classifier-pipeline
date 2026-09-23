import socket

from piclassifier.headerinfo import HeaderInfo
import os
import time
import logging
import sys
from multiprocessing import Queue, Process
import numpy as np
from pathlib import Path
from datetime import datetime

fmt = "%(asctime)s %(process)d %(thread)s:%(levelname)7s %(message)s"

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
)
SOCKET_NAME = "/var/run/lepton-frames"
start = time.time()


WRITE_CPTV = True
PROCESS_LOAD = True
TEST = len(sys.argv) > 1
if TEST:
    MODEL_PATH = "./thermal-model/converted_model.tflite"
else:
    MODEL_PATH = "/home/pi/tflite/converted_model.tflite"
MODEL_PATH = "/home/pi/tflite/converted_model.tflite"

# MODEL_PATH = "/home/gp/cacophony/classifier-data/thermal-training/2026Aug/v11/160/qat/singleExclude160QAT.tflite"
def parse_cptv(cptv_file, frame_queue):
    from cptv_rs_python_bindings import CptvReader
    from cptv import Frame

    reader = CptvReader(cptv_file)
    while True:
        frame = reader.next_frame()
        if frame is None:
            break
        py_frame = Frame(
            frame.pix,
            frame.time_on,
            frame.last_ffc_time,
            frame.temp_c,
            frame.last_ffc_temp_c,
        )
        frame_queue.put((py_frame, time.time()))
        time.sleep(1 / 9)
    frame_queue.put(CLEAR_SIGNAL)

    frame_queue.put(STOP_SIGNAL)


def run_cmd(cmd):
    import subprocess

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            encoding="ascii",
            check=True,
        )
        return result.returncode == 0

    except:
        logging.error("Could not run command %s", cmd, exc_info=True)
        return False



# def run_medium():
#     from config.thermalconfig import ThermalConfig

#     thermal_config = ThermalConfig.load_from_file()

#     thermal_config.recorder.rec_window.set_location(
#         *thermal_config.location.get_lat_long(use_default=True),
#         thermal_config.location.altitude,
#     )
#     main(thermal_config)

def main():
    logging.info("Main started")
    global connected
    frame_queue = Queue()
    config = None
    processor = get_processor(frame_queue)
    processor.start()

    if TEST:
        print("Parsing test.cptv")
        parse_cptv("test.cptv", frame_queue)
        processor.join()
        return


    logging.info("Making sock")
    try:
        os.unlink(SOCKET_NAME)
    except OSError:
        if os.path.exists(SOCKET_NAME):
            raise
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    sock.bind(SOCKET_NAME)
    sock.settimeout(3 * 60)  # 3 minutes
    sock.listen(1)
    global start

    while True:
        logging.info("waiting for a connection %s", time.time() - start)
        try:
            connection, client_address = sock.accept()
            connected = True
            logging.info("connection from %s", client_address)
            # log_event("camera-connected", {"type": "thermal"})
            medium_power(connection, frame_queue, processor, config)
        except KeyboardInterrupt:
            logging.info("\nCtrl+C pressed. Exiting gracefully.")
            break
        except Exception as ex:
            logging.error("Error with connection", exc_info=True)

        finally:
            # Clean up the connection
            try:
                connection.close()
            except:
                pass
        connected = False
        start = time.time()
    frame_queue.put(STOP_SIGNAL)
    processor.join()


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
            # logging.info("Headers %s done %s ", headers, done)
            # need the clear message
            left_over = headers[done + 2 :]
            headers = headers[:done]

            # ensure we handle the clear message
            if len(left_over) < 5:
                left_over += connection.recv(5 - len(left_over))

            if left_over[:5] == b"clear":
                left_over = left_over[5:]
            break
    header_s = headers.decode()
    logging.info("header is %s ", header_s)
    return HeaderInfo.parse_header(header_s), left_over


def get_processor(process_queue):
    p_processor = Process(
        target=run_classifier,
        args=(process_queue,),
    )
    return p_processor


def ask_to_stay_on(duration=5):
    import dbus

    try:
        bus = dbus.SystemBus()
        dbus_object = bus.get_object("org.cacophony.ATtiny", "/org/cacophony/ATtiny")
        dbus_object.StayOnForProcess("medium-power", duration)
        logging.info("Asked attiny to stay on for 5 minutes")
        return True
    except:

        logging.error("Error asking to stay on ")
        # error is so verbose and will always happen on startup, exc_info=True)
    return False


def medium_power(connection, frame_queue, processor, config):
    from cptv_rs_python_bindings import CptvStreamReader
    import zlib
    from cptv import Frame
    connection.settimeout(20)
    headers, extra_b = handle_headers(connection)
    stream_i = 0
    connection.settimeout(5)
    logging.info("Medium Power =======")
    asked_to_stay_on = False

    while True:
        # wait for start message
        if extra_b is None or len(extra_b) == 0:
            try:
                extra_b = connection.recv(headers.frame_size)
            except (socket.timeout, TimeoutError):
                extra_b = None
                continue
            except:
                logging.error("Couldnt get start", exc_info=True)
                extra_b = None
                continue
        start_index = extra_b.find(b"start\n\n")
        if start_index > -1:
            extra_b = extra_b[start_index + len("start\n\n") :]
        else:
            # if dbus wasnt on when rec started do it now
            if not asked_to_stay_on:
                asked_to_stay_on = ask_to_stay_on()

            if len(extra_b) == 0:
                logging.info("Disconnected waiting for start")
                # disconnected
                return
            extra_b = None
            continue

      
        reader = CptvStreamReader()
        decompressor = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
        recording = False
        u8_data = None
        frame_i = 0
        read_header = False
        data = b""
        finished = False
        min_value = None
        max_value = None

        while len(extra_b)< 8+4+4:
            logging.info("Missing timestamp info waiting for more data")
            try:
                byte_data = connection.recv(headers.frame_size)
                if len(byte_data) == 0:
                    # disconnected from socket
                    logging.info("Disconnected from socket")
                    return
                extra_b += byte_data
            except:
                time.sleep(1)
                continue  
        timestamp = struct.unpack("<q", extra_b[:8])[0]
        lepton_serial = struct.unpack("<I", extra_b[8:12])[0]
        firmware = struct.unpack("<I", extra_b[12:16])[0]
        headers.firmware =f"DOC-AI-v0.{firmware}"
        headers.serial = str(lepton_serial)
        logging.info("Timestamp received is %s serial %s firmware %s",timestamp,lepton_serial,firmware)
        formatted_time = datetime.fromtimestamp(timestamp/1e+6).strftime("%Y%m%d-%H%M%S.%f")

        extra_b = extra_b[8+4+4:]
        if WRITE_CPTV:
            # write header and cptv file seperately and then concat later
            # this way can write header with total frames and min max value
            f = open(f"/var/spool/cptv/temp/{formatted_time}.cptv", "wb")
            logging.info(f"Writing cptv file %s", f.name)
            # from cptvwriter import write_header
            # write_header(f"/var/spool/cptv/temp/raw{stream_i}-{time.time()}-header.gz",headers, config,timestamp)
        byte_data = b""
        if extra_b is not None:
            data = extra_b
            if WRITE_CPTV:
                f.write(extra_b)
        stream_i += 1
        while not finished:
            try:
                byte_data = connection.recv(headers.frame_size)
                if len(byte_data) == 0:
                    # disconnected from socket
                    logging.info("Disconnected from socket")
                    if recording:
                        logging.error("Mid recording failed to receive more data")
                        frame_queue.put(CLEAR_SIGNAL)
                        f.close()
                        remove_file(f.name)

                    return
            except:
                if recording:
                    logging.error("Mid recording failed to receive more data")
                    frame_queue.put(CLEAR_SIGNAL)
                    f.close()
                    remove_file(f.name)
                    break
                time.sleep(1)
                continue



            clear_index = byte_data.find(b"clear")
            if clear_index > -1:
                byte_data = byte_data[:clear_index]

                logging.info("Received clear finished file")
                finished = True
                frame_queue.put(CLEAR_SIGNAL)
                if WRITE_CPTV:
                    f.write(byte_data)
                    f.close()
                    from cptvwriter import write_header
                    file_path = Path(f.name)

                    write_header(f"/var/spool/cptv/temp/{formatted_time}-header.gz",headers,config,timestamp, min_value,max_value,frame_i)
                    combine_file(f"/var/spool/cptv/temp/{formatted_time}-header.gz", f.name,file_path.parent.parent / file_path.name)
                    # move from temp to actual folder
                    # shutil.move(file_path, file_path.parent.parent / file_path.name)

                # might have another start
                extra_b = byte_data[clear_index + len("clear") :]
            elif byte_data.find(b"abort") > -1:
                logging.info("Received abort signal")
                finished = True
                frame_queue.put(CLEAR_SIGNAL)
                if WRITE_CPTV:
                    f.close()
                    os.remove(f.name)
                break
            else:
                if WRITE_CPTV:
                    f.write(byte_data)
                logging.debug(
                    "Adding new data %s to old data %s", len(byte_data), len(data)
                )
            data = data + byte_data

            if len(data) == 0:
                time.sleep(1)
                continue

            try:
                logging.debug("Decompressing %s", len(data))
                data, decompressed_chunk, read_header = decompress(
                    decompressor, data, read_header
                )
            except:
                # if this happens log it and then get the file from rp2040
                logging.error("Error decompressing ", exc_info=True)
                return
                # time.sleep(1)
                # continue

            if len(decompressed_chunk) == 0:
                continue
            recording = True
            if u8_data is None:
                u8_data = np.frombuffer(decompressed_chunk, dtype=np.uint8)
            else:
                # logging.info("Adding more u8 %s to existing %s", len(decompressed_chunk), len(u8_data))
                u8_data = np.concatenate(
                    (u8_data, np.frombuffer(decompressed_chunk, dtype=np.uint8)), axis=0
                )

            # logging.info("Loading frames wtih %s", len(u8_data))
            while True:
                # need to figure out whats happening with the endiness
                result = reader.next_frame_from_data(u8_data, False)
                if result is not None:
                    frame, used = result
                    u8_data = u8_data[used:]
                    py_frame = Frame(
                        frame.pix,
                        frame.time_on,
                        frame.last_ffc_time,
                        frame.temp_c,
                        frame.last_ffc_temp_c,
                    )
                    frame_min = np.amin(py_frame.pix)
                    frame_max = np.amax(py_frame.pix)

                    if min_value is None or frame_min < min_value:
                        min_value = frame_min
                    if max_value is None or max_value > frame_max:
                        max_value = frame_max
                    

                    frame_queue.put((py_frame, time.time()))
                    frame_i += 1

                else:
                    logging.debug(
                        "Have %s bytes but need more to decompress a frame",
                        len(u8_data),
                    )
                    break
        logging.info(
            "Finished processing left over bytes are %s num frames %s",
            "None" if u8_data is None else len(u8_data),
            frame_i,
        )
        u8_data = None
        data = b""
        reader = None
        asked_to_stay_on = ask_to_stay_on()



def combine_file(header_file, frame_file,output_file):
    import subprocess
    import os
    try:
    # Simple cat command to display a file's content
        command = f"cat {header_file} {frame_file} >> {str(output_file)}"
        result = subprocess.run(    ["sudo", "bash", "-c", command],capture_output=True,check=True)
        # logging.info("Combine output %s",result)
    except:
        logging.error("Failed to combine %s %s",header_file,frame_file,exc_info=True)
   
    remove_file(header_file)
    remove_file(frame_file)

def remove_file(file_name):
    import os
    try:
        os.remove(file_name)
    except:
        logging.error("Failed to remove %s",file_name,exc_info=True)

import zlib
import io
import struct
import gzip


def decompress(decompressor, data, read_header=False):

    fp = io.BytesIO(data)
    if not read_header:
        result = gzip._read_gzip_header(fp)
        if result is None:
            raise Exception("No gzip header found")
            # logging.info("Couldn't read header")
            # return data, b"", read_header
        data = data[fp.tell() :]
        read_header = True
    try:
        decompressed = decompressor.decompress(data)
    except:
        logging.error("Error decompressing ", exc_info=True)
        return data, b"", read_header
    unused_data = decompressor.unused_data[8:].lstrip(b"\x00")

    # print("Tell is no0w ", fp.tell()," Unused data is " , len(decompressor.unused_data), " decompressed is ",len(decompressed))
    if not decompressor.eof or len(decompressor.unused_data) < 8:
        # print("Reach eof")
        # 1/0
        return unused_data, decompressed, read_header
        raise EOFError(
            "Compressed file ended before the end-of-stream " "marker was reached"
        )
    crc, length = struct.unpack("<II", decompressor.unused_data[:8])

    if crc != zlib.crc32(decompressed):
        # not check this proparly so will always error
        # logging.error("CRC error")
        return unused_data, decompressed, read_header

        raise Exception("CRC check failed")
    if length != (len(decompressed) & 0xFFFFFFFF):
        raise Exception("Incorrect length of data produced")
    return unused_data, decompressed, read_header




CLEAR_SIGNAL = "clear"

STOP_SIGNAL = "stop"
SKIP_SIGNAL = "skip"


def get_active_tracks(clip):
    """
    Gets current clips active_tracks and returns the top NUM_CONCURRENT_TRACKS order by priority
    """
    active_tracks = clip.active_tracks
    active_tracks = [track for track in active_tracks if len(track) >= 8]
    return active_tracks


last_frame_predicted = None
classify_executor = None
predicting_track_id = None


# classify first animal track
# otherwise longest unclassified track
# otherwise least false positive confidence track
def best_track_to_classify(clip, monitored_tracks):
    active_tracks = get_active_tracks(clip)
    least_fp_track = None
    unclassified_longest = None
    for track in active_tracks:
        track_pred = monitored_tracks.get(track._id)
        if track_pred is None or track_pred.num_frames_classified == 0:
            if unclassified_longest is None:
                unclassified_longest = track
            elif len(track) > len(unclassified_longest):
                unclassified_longest = track
            continue
        tag = classifier.labels[track_pred.best_label_index]
        if tag == "false-positive":
            conf = track_pred.normalized_best_score()
            if least_fp_track is None:
                least_fp_track = (conf, track)
            elif least_fp_track[0] > conf:
                least_fp_track = (conf, track)
        else:
            # for now just classify first track that isn't fp
            logging.info("Continuing to classify %s with tag %s", track, tag)
            return track
    if unclassified_longest:
        logging.info("Classifying longest unclassified track")
        return unclassified_longest
    elif least_fp_track is not None:
        logging.info("Classifying most unlikely fp track")
        return least_fp_track[1]
    return None

    # return longest track?


def submit_prediction(clip, monitored_tracks, tracking_events):
    """Preprocesses the best track (needs live clip/track state, so runs on
    the caller's thread) then hands the actual inference off to a single
    background worker so the frame/tracking loop doesn't block on it.

    The TrackPrediction entry is created here, on the caller's thread,
    before the job is submitted - so the worker only ever updates an
    existing value in monitored_tracks and never inserts or removes a key.
    That means it can safely run while the main thread is iterating
    monitored_tracks.items() elsewhere, no locking needed."""
    track = best_track_to_classify(clip, monitored_tracks)
    if track is None:
        logging.info("No active tracks %s", len(clip.active_tracks))
        return
    if classifier is None:
        logging.info("Not classifying as couldn't load model")
        return
    from classify.trackprediction import TrackPrediction

    track_pred = monitored_tracks.setdefault(
        track._id,
        TrackPrediction(
            track._id,
            classifier.labels,
            keep_all=False,
            parent_mappings=classifier.parent_mappings,
            scale_thresholds=classifier.scale_thresholds,
            thresholds_per_label=classifier.thresholds_per_label,
            multi_label=classifier.params.multi_label,
        ),
    )
    if track_pred.previous_prediction_was_short():
        logging.info("Resetting as was short")
        track_pred.reset()

    preprocessed_result = classifier.preprocess_track(
        clip,
        track,
        num_predictions=1,
        predict_from_last=100,
    )
    if preprocessed_result is None:
        logging.error("Pred is none for %s", track)
        return
    frames, preprocessed, mass = preprocessed_result

    global predicting_track_id
    predicting_track_id = track._id
    classify_executor.submit(
        _predict_and_apply, track, track_pred, preprocessed, frames, mass, start, tracking_events
    )


def _predict_and_apply(track, track_pred, preprocessed, frames, mass, start, tracking_events):
    """Runs entirely on the single worker thread: infers then writes the
    result straight into monitored_tracks. Sets predicting_track_id itself,
    at the moment it actually starts running rather than when it was queued,
    so the flag always names whichever track is truly mid-update right now -
    correct even if more than one job is queued, since max_workers=1
    guarantees only one ever executes at a time. Safe without a lock because
    the dict entry already exists (created on the caller's thread before
    this was submitted) and predicting_track_id is a plain attribute
    assignment, which the GIL makes atomic regardless of which thread does it."""
    global predicting_track_id
    try:
        prediction = classifier.predict(preprocessed)
        track_pred.classified_frames(frames, prediction, mass)
        logging.info(
            "Track %s is predicted as %s conf %s took %s track frames %s",
            track,
            classifier.labels[track_pred.best_label_index],
            track_pred.description(),
            time.time() - start,
            len(track),
        )
        predicted_as = classifier.labels[track_pred.best_label_index]
        conf = track_pred.normalized_best_score()
        now = datetime.now()
        tracking_events.append(
            (
                track_pred.track_id,
                predicted_as,
                conf,
                track.bounds_history[-1],
                track_pred.last_frame_classified,
                now.strftime("%B %d, %Y %I:%M:%S %p"),
            )
        )
    except Exception:
        logging.error("Could not predict", exc_info=True)
    finally:
        predicting_track_id = None


classifier = None


def load_model(over_network=False):
    global classifier
    from ml_tools.interpreter import LiteInterpreter

    logging.info("Loading tflite model")
    load_start = time.time()
    try:
        print("Loading ", MODEL_PATH)
        classifier = LiteInterpreter(Path(MODEL_PATH), over_network, True)
        # this way metadata is loaded
        # classifier.load_model()
        logging.info("Loaded tflite model in %.2fs", time.time() - load_start)
    except:
        logging.error("Could not load model", exc_info=True)


def run_classifier(frame_queue):


    global predicting_track_id
    run_classifier_start = time.time()
    if PROCESS_LOAD:
        # this needs ot be killed
        classifier_process = run_classifier_process()
        load_model(over_network=True)
    else:
        load_model(over_network=False)
    logging.info(
        "Model load kicked off after %.2fs (THREAD_LOAD=%s)",
        time.time() - run_classifier_start,
        PROCESS_LOAD,
    )

    from piclassifier.motiondetector import RunningMean
    from concurrent.futures import ThreadPoolExecutor
    global classify_executor
    classify_executor = ThreadPoolExecutor(max_workers=1)

    headers = {}
    frame_i = 0
    predict_every = 10
    dbus_service = None
    tracking_events = []
    # only need for over network
    imported_requests = not PROCESS_LOAD
    try:
        while True:
            running_mean = None

            if len(tracking_events) > 0:
                from piclassifier.eventreporter import log_event

                logging.info("Logging tracking events")
                for tracking_event in tracking_events:
                    (
                        track_id,
                        tag,
                        conf,
                        region,
                        last_frame_classified,
                        classified_at,
                    ) = tracking_event
                    region_list = [int(x) for x in region.to_ltrb()]
                    log_event(
                        "tracking",
                        {
                            "track_id": track_id,
                            "tag": tag,
                            "confidence": int(round(100 * conf)),
                            "region": region_list,
                            "last_frame_classified": last_frame_classified,
                            "time": classified_at,
                        },
                    )
                tracking_events = []
            logging.info("Making a new clip")
            monitored_tracks = {}
            stale_track_ids = set()
            # note: predicting_track_id is deliberately left alone here. If a
            # prediction from the previous clip is still running, the worker
            # thread will write its result into an orphaned track_pred/dict
            # (harmless) and clear the flag itself when done; the guard at
            # the predict_every check below already waits for that before
            # submitting the new clip's first prediction.

            track_extractor, clip = new_clip()
            logging.info(
                "Waiting for frames %.2fs after run_classifier started",
                time.time() - run_classifier_start,
            )
            rec_sent = False
            while True:
                try:
                    if not imported_requests:
                        frame = frame_queue.get(timeout=0)
                    else:
                        frame = frame_queue.get()
                except:
                    # means when first predictions comes through this wont need to be imported which takes a second
                    import requests
                    imported_requests = True
                    continue

                if isinstance(frame, str):
                    if frame == CLEAR_SIGNAL:
                        rec_sent= False
                        logging.info(
                            "PiClassifier received clear signal will start a new clip"
                        )
                        if dbus_service:

                            for track_id, track_pred in monitored_tracks.items():
                                # predicted_as = classifier.labels[
                                #     track_pred.best_label_index
                                # ]
                                track = [
                                    track
                                    for track in clip.tracks
                                    if track._id == track_id
                                ]
                                if len(track)==0:
                                    # guessing its finished now probably can be deleted need to check
                                    continue
                                track = track[0]
                                dbus_service.tracking(
                                    clip.id,
                                    track,
                                    track_pred.get_normalized_score(),
                                    track.bounds_history[-1],
                                    False,
                                    track_pred.last_frame_classified,
                                    classifier.labels,
                                    classifier.id,
                                )
                            dbus_service.recording(False,time.time())
                        else:
                            logging.error(
                                "Dbus service never got started and recording is now finished"
                            )
                        frame_i = 0
                        break
                    elif frame == STOP_SIGNAL:
                        logging.info("PiClassifier received stop signal")
                        if PROCESS_LOAD:
                            from piclassifier.utils import kill_process_with_timeout
                            kill_process_with_timeout(classifier_process)
                        return
                else:
                    frame, time_sent = frame
                    if running_mean is None:
                        running_mean = RunningMean([frame.pix], 45)
                    else:
                        oldest_thermal = clip.current_frame

                        # first frame is at 0
                        oldest = clip.frame_buffer.get_frame(oldest_thermal - (running_mean.window_size-1))
                        if oldest is not None:
                            oldest = oldest.thermal

                        running_mean.add(frame.pix, oldest)
                    mean_frame = running_mean.mean()
                    # if this is the first frame background needs to be initialized
                    track_extractor.background_alg.process_frame(mean_frame)
                    

                    
                    new_tracks,stale_tracks = track_extractor.process_frame(clip, frame)
                    for t in new_tracks:
                            t.received_at = time.time()

                    if dbus_service is None and classifier is not None:
                        try:
                            from piclassifier.service import DbusService

                            dbus_service = DbusService(headers,None, classifier.labels,None,None,None,True,True)
                            dbus_service.recording(time.time(),True)
                            rec_sent = True
                        except:
                            logging.error(
                                "Couldnt load dbus will try again ", exc_info=True
                            )
                    elif not rec_sent:
                        try:
                            dbus_service.recording(time.time(),True)
                            rec_sent = True
                        except:
                            pass
                    frame_i += 1
                    if frame_i == 1:
                        logging.info("Recording started")

                    # mark stale tracks for removal - actually finalized in
                    # the reporting loop below, which defers whichever one is
                    # currently being predicted so its last report is fresh
                    if len(monitored_tracks) > 0:
                        for track in stale_tracks:
                            if track._id in monitored_tracks:
                                stale_track_ids.add(track._id)
                    logging.info(
                        "%s Predicting behind by %s ",
                        frame_i,
                        time.time() - time_sent,
                    )

                    
                    if dbus_service:
                        # list(...) snapshots the items so popping a stale
                        # entry below doesn't disturb this iteration
                        for track_id, track_pred in list(monitored_tracks.items()):
                            if track_id in stale_track_ids:
                                if track_id == predicting_track_id:
                                    # worker is still mid-update for this one -
                                    # wait so the final report has its result,
                                    # not whatever was there before
                                    continue
                                stale_track_ids.discard(track_id)
                                monitored_tracks.pop(track_id, None)
                                track = [t for t in clip.tracks if t._id == track_id]
                                if len(track) == 0:
                                    continue
                                track = track[0]
                                tracking = False
                            else:
                                track = [
                                    track
                                    for track in clip.active_tracks
                                    if track._id == track_id
                                ][0]
                                tracking = True
                            dbus_service.tracking(
                                clip.id,
                                track,
                                track_pred.get_normalized_score(),
                                track.bounds_history[-1],
                                tracking,
                                track_pred.last_frame_classified,
                                classifier.labels,
                                classifier.id,
                                track.received_at,
                            )
                    if predicting_track_id is None:
                        logging.info(
                            "%s Predicting behind by %s ",
                            frame_i,
                            time.time() - time_sent,
                        )
                        submit_prediction(clip, monitored_tracks, tracking_events)
    except:
        logging.error("Error running classifier restarting ..", exc_info=True)
        if PROCESS_LOAD:
            from piclassifier.utils import kill_process_with_timeout
            kill_process_with_timeout(classifier_process)
        return


# class DefaultTracking:
#     def 


def init_trackers(tracking_config):
    from track.cliptrackextractor import ClipTrackExtractor
    from piclassifier.motiondetector import WeightedBackground

    track_extractor = ClipTrackExtractor(tracking_config)
    track_extractor.background_alg = WeightedBackground()

    return track_extractor


def new_clip():
    from track.clip import Clip
    from config.trackingconfig import TrackingConfig

    default_tracking = TrackingConfig.get_defaults()
    default_tracking.denoise = False
    track_extractor = init_trackers(default_tracking)

    clip = Clip(default_tracking,"stream",calc_stats= False,model = "lepton3.5")
    clip.crop_rectangle = track_extractor.background_alg.crop_rectangle
    clip.res_x = 160
    clip.res_y = 120
    clip.set_frame_buffer(
            False,
            keep_frames=True,
            max_frames=100,
            lock = False
        )

    return track_extractor, clip


def _classifier_main():
    from piclassifier.servemodel import main

    main(warmup = False,model_file = MODEL_PATH)


def run_classifier_process():
    p_processor = Process(
        target=_classifier_main,
        args=(),
    )
    p_processor.start()
    return p_processor
    


if __name__ == "__main__":
    main()
