#!/usr/bin/python3
from datetime import datetime, timedelta
import numpy as np
import os
import logging
import socket
import time
import absl.logging
import json
import psutil

from cptv import Frame

from classify.trackprediction import Predictions
from load.clip import Clip
from load.cliptrackextractor import ClipTrackExtractor
from .telemetry import Telemetry
from .locationconfig import LocationConfig
from .thermalconfig import ThermalConfig
from .motiondetector import MotionDetector
from .cptvrecorder import CPTVRecorder

from ml_tools.logs import init_logging
from ml_tools import tools
from ml_tools.model import Model
from ml_tools.dataset import Preprocessor
from ml_tools.config import Config
from ml_tools.previewer import Previewer

SOCKET_NAME = "/var/run/lepton-frames"
VOSPI_DATA_SIZE = 160
TELEMETRY_PACKET_COUNT = 4


def get_classifier(config):

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
    try:
        os.unlink(SOCKET_NAME)
    except OSError:
        if os.path.exists(SOCKET_NAME):
            raise
    if not os.path.exists("metadata"):
        os.makedirs("metadata")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sock.bind(SOCKET_NAME)
    sock.listen(1)
    config = Config.load_from_file()
    thermal_config = ThermalConfig.load_from_file()
    location_config = LocationConfig.load_from_file()
    classifier = get_classifier(config)

    clip_classifier = PiClassifier(config, thermal_config, location_config, classifier)
    while True:
        logging.info("waiting for a connection")
        connection, client_address = sock.accept()
        logging.info("connection from %s", client_address)
        try:
            handle_connection(connection, clip_classifier)
        finally:
            # Clean up the connection
            connection.close()


def handle_connection(connection, clip_classifier):
    img_dtype = np.dtype("uint16")
    # big endian > little endian <
    # lepton3 is big endian while python is little endian

    thermal_frame = np.empty(
        (clip_classifier.res_y, clip_classifier.res_x), dtype=img_dtype
    )
    while True:
        data = connection.recv(400 * 400 * 2 + TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE)

        if not data:
            logging.info("disconnected from camera")
            clip_classifier.disconnected()
            return

        if len(data) > clip_classifier.res_y * clip_classifier.res_x * 2:
            telemetry = Telemetry.parse_telemetry(
                data[: TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE]
            )

            thermal_frame = np.frombuffer(
                data, dtype=img_dtype, offset=TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE
            ).reshape(clip_classifier.res_y, clip_classifier.res_x)
        else:
            telemetry = Telemetry()
            telemetry.last_ffc_time = timedelta(milliseconds=time.time())
            telemetry.time_on = timedelta(
                milliseconds=time.time(), seconds=MotionDetector.FFC_PERIOD.seconds + 1
            )
            thermal_frame = np.frombuffer(data, dtype=img_dtype, offset=0).reshape(
                clip_classifier.res_y, clip_classifier.res_x
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
            clip_classifier.skip_frame()
            continue
        clip_classifier.process_frame(lepton_frame)


class PiClassifier:
    """ Classifies frames from leptond """

    PROCESS_FRAME = 3
    NUM_CONCURRENT_TRACKS = 1
    DEBUG_EVERY = 100
    MAX_CONSEC = 3

    def __init__(self, config, thermal_config, location_config, classifier):
        self.frame_num = 0
        self.clip = None
        self.tracking = False
        self.enable_per_track_information = False
        self.rolling_track_classify = {}
        self.skip_classifying = 0
        self.classified_consec = 0
        self.config = config
        self.classifier = classifier
        self.num_labels = len(classifier.labels)
        self.res_x = self.config.classify.res_x
        self.res_y = self.config.classify.res_y
        self.predictions = Predictions(classifier.labels)
        self.preview_frames = (
            thermal_config.recorder.preview_secs * thermal_config.recorder.frame_rate
        )
        edge = self.config.tracking.edge_pixels
        self.crop_rectangle = tools.Rectangle(
            edge, edge, self.res_x - 2 * edge, self.res_y - 2 * edge
        )

        try:
            self.fp_index = self.classifier.labels.index("false-positive")
        except ValueError:
            self.fp_index = None

        self.track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow,
            self.config.classify.cache_to_disk,
            keep_frames=False,
            calc_stats=False,
        )
        self.motion_config = thermal_config.motion
        self.min_frames = (
            thermal_config.recorder.min_secs * thermal_config.recorder.frame_rate
        )
        self.max_frames = (
            thermal_config.recorder.max_secs * thermal_config.recorder.frame_rate
        )
        self.motion_detector = MotionDetector(
            self.res_x,
            self.res_y,
            thermal_config.motion,
            location_config,
            thermal_config.recorder,
            self.config.tracking.dynamic_thresh,
            CPTVRecorder(location_config, thermal_config),
        )
        self.startup_classifier()

    def new_clip(self):
        self.clip = Clip(self.config.tracking, "stream")
        self.clip.video_start_time = datetime.now()
        self.clip.num_preview_frames = self.preview_frames

        self.clip.set_res(self.res_x, self.res_y)
        self.clip.set_frame_buffer(
            self.config.classify_tracking.high_quality_optical_flow,
            self.config.classify.cache_to_disk,
            self.config.use_opt_flow,
            True,
        )

        # process preview_frames
        frames = self.motion_detector.thermal_window.get_frames()
        for frame in frames:
            self.track_extractor.process_frame(self.clip, frame.copy())

    def startup_classifier(self):
        # classifies an empty frame to force loading of the model into memory

        p_frame = np.zeros((5, 48, 48), np.float32)
        self.classifier.classify_frame_with_novelty(p_frame, None)

    def get_active_tracks(self):
        """
        Gets current clips active_tracks and returns the top NUM_CONCURRENT_TRACKS order by priority
        """
        active_tracks = self.clip.active_tracks
        if len(active_tracks) <= PiClassifier.NUM_CONCURRENT_TRACKS:
            return active_tracks
        active_predictions = []
        for track in active_tracks:
            prediction = self.predictions.get_or_create_prediction(track, keep_all=False)
            active_predictions.append(prediction)

        top_priority = sorted(
            active_predictions,
            key=lambda i: i.get_priority(self.clip.frame_on),
            reverse=True,
        )

        top_priority = [
            track.track_id
            for track in top_priority[: PiClassifier.NUM_CONCURRENT_TRACKS]
        ]
        classify_tracks = [
            track for track in active_tracks if track.get_id() in top_priority
        ]
        return classify_tracks

    def identify_last_frame(self):
        """
        Runs through track identifying segments, and then returns it's prediction of what kind of animal this is.
        One prediction will be made for every active_track of the last frame.
        :return: TrackPrediction object
        """

        prediction_smooth = 0.1

        smooth_prediction = None
        smooth_novelty = None

        prediction = 0.0
        novelty = 0.0
        active_tracks = self.get_active_tracks()
        frame = self.clip.frame_buffer.get_last_frame()
        if frame is None:
            return
        thermal_reference = np.median(frame.thermal)

        for i, track in enumerate(active_tracks):
            track_prediction = self.predictions.get_or_create_prediction(track, keep_all=False)
            region = track.bounds_history[-1]
            if region.frame_number != frame.frame_number:
                logging.warning("frame doesn't match last frame")
            else:
                track_data = track.crop_by_region(frame, region)
                # we use a tighter cropping here so we disable the default 2 pixel inset
                frames = Preprocessor.apply(
                    [track_data], [thermal_reference], default_inset=0
                )
                if frames is None:
                    logging.warning(
                        "Frame {} of track could not be classified.".format(
                            region.frame_number
                        )
                    )
                    continue
                p_frame = frames[0]
                prediction, novelty, state = self.classifier.classify_frame_with_novelty(
                    p_frame, track_prediction.state
                )
                track_prediction.state = state

                if self.fp_index is not None:
                    prediction[self.fp_index] *= 0.8
                state *= 0.98
                mass = region.mass
                mass_weight = np.clip(mass / 20, 0.02, 1.0) ** 0.5
                cropped_weight = 0.7 if region.was_cropped else 1.0

                prediction *= mass_weight * cropped_weight

                if len(track_prediction.predictions) == 0:
                    if track_prediction.uniform_prior:
                        smooth_prediction = np.ones([self.num_labels]) * (
                            1 / self.num_labels
                        )
                    else:
                        smooth_prediction = prediction
                    smooth_novelty = 0.5
                else:
                    smooth_prediction = track_prediction.predictions[-1]
                    smooth_novelty = track_prediction.novelties[-1]
                    smooth_prediction = (
                        1 - prediction_smooth
                    ) * smooth_prediction + prediction_smooth * prediction
                    smooth_novelty = (
                        1 - prediction_smooth
                    ) * smooth_novelty + prediction_smooth * novelty
                track_prediction.classified_frame(
                    self.clip.frame_on, smooth_prediction, smooth_novelty
                )
                # track_prediction.print_prediction(self.predictions.labels)

    def disconnected(self):
        self.end_clip()
        self.motion_detector.force_stop()

    def skip_frame(self):
        self.skip_classifying -= 1

        if self.clip:
            self.clip.frame_on += 1

    def process_frame(self, lepton_frame):
        start = time.time()
        self.motion_detector.process_frame(lepton_frame)
        if self.motion_detector.recorder.recording:
            if self.clip is None:
                self.new_clip()
            self.track_extractor.process_frame(
                self.clip, lepton_frame.pix, self.motion_detector.ffc_affected
            )
            if (
                self.motion_detector.ffc_affected is False
                and self.clip.active_tracks
                and self.skip_classifying <= 0
                # and (
                #     self.clip.frame_on % PiClassifier.PROCESS_FRAME == 0
                #     or self.clip.frame_on == self.preview_frames
                # )
            ):
                self.identify_last_frame()
                self.classified_consec += 1
                if self.classified_consec == PiClassifier.MAX_CONSEC:
                    self.skip_classifying = 7
                    self.classified_consec = 0

        elif self.clip is not None:
            self.end_clip()

        self.skip_classifying -= 1
        self.frame_num += 1
        end = time.time()
        timetaken = end - start
        if (
            self.motion_detector.can_record()
            and self.frame_num % PiClassifier.DEBUG_EVERY == 0
        ):
            logging.info(
                "fps {}/sec time to process {}ms cpu % {} memory % {}".format(
                    round(1 / timetaken, 2),
                    round(timetaken * 1000, 2),
                    psutil.cpu_percent(),
                    psutil.virtual_memory()[2],
                )
            )

    def create_mp4(self):
        previewer = Previewer(self.config, "classified")
        previewer.export_clip_preview(
            self.clip.get_id() + ".mp4", self.clip, self.predictions
        )

    def end_clip(self):
        if self.clip:
            for _, prediction in self.predictions.prediction_per_track.items():
                if prediction.max_score:
                    logging.info(prediction.description(self.predictions.labels))
            self.save_metadata()
            self.predictions.clear_predictions()
            self.clip = None
            self.tracking = False

    def save_metadata(self):
        filename = datetime.now().strftime("%Y%m%d.%H%M%S.%f.meta")

        # record results in text file.
        save_file = {}
        start, end = self.clip.start_and_end_time_absolute()
        save_file["start_time"] = start.isoformat()
        save_file["end_time"] = end.isoformat()
        save_file["temp_thresh"] = self.clip.temp_thresh
        save_file["algorithm"] = {}
        save_file["algorithm"]["model"] = self.config.classify.model
        save_file["algorithm"]["tracker_version"] = self.clip.VERSION
        save_file["tracks"] = []
        for track in self.clip.tracks:
            track_info = {}
            prediction = self.predictions.prediction_for(track.get_id())
            start_s, end_s = self.clip.start_and_end_in_secs(track)
            save_file["tracks"].append(track_info)
            track_info["start_s"] = round(start_s, 2)
            track_info["end_s"] = round(end_s, 2)
            track_info["num_frames"] = track.frames
            track_info["frame_start"] = track.start_frame
            track_info["frame_end"] = track.end_frame
            if prediction and prediction.best_label_index is not None:
                track_info["label"] = self.classifier.labels[
                    prediction.best_label_index
                ]
                track_info["confidence"] = round(prediction.score(), 2)
                track_info["clarity"] = round(prediction.clarity, 3)
                track_info["average_novelty"] = round(prediction.average_novelty, 2)
                track_info["max_novelty"] = round(prediction.max_novelty, 2)
                track_info["all_class_confidences"] = {}
                for i, value in enumerate(prediction.class_best_score):
                    label = self.classifier.labels[i]
                    track_info["all_class_confidences"][label] = round(float(value), 3)

            positions = []
            for region in track.bounds_history:
                track_time = round(region.frame_number / self.clip.frames_per_second, 2)
                positions.append([track_time, region])
            track_info["positions"] = positions

        with open("metadata/" + filename, "w") as f:
            json.dump(save_file, f, indent=4, cls=tools.CustomJSONEncoder)
