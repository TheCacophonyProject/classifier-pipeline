from datetime import datetime
import json
import logging
import os
import time
import psutil
import numpy as np
import logging
from pathlib import Path

from classify.trackprediction import Predictions
from load.clip import Clip
from load.irtrackextractor import IRTrackExtractor

from load.cliptrackextractor import ClipTrackExtractor

from ml_tools.previewer import Previewer, add_last_frame_tracking
from ml_tools import tools
from .cptvrecorder import CPTVRecorder
from .throttledrecorder import ThrottledRecorder
from .dummyrecorder import DummyRecorder
from .irrecorder import IRRecorder
from .irmotiondetector import IRMotionDetector
from .cptvmotiondetector import CPTVMotionDetector

from .motiondetector import SlidingWindow
from .processor import Processor

from ml_tools.interpreter import Interpreter
from ml_tools.logs import init_logging
from ml_tools.hyperparams import HyperParams
from ml_tools.tools import CustomJSONEncoder
from ml_tools.forestmodel import ForestModel
from track.region import Region
from . import beacon

from piclassifier.eventreporter import trapped_event

SNAPSHOT_SIGNAL = "snap"
STOP_SIGNAL = "stop"
SKIP_SIGNAL = "skip"
track_extractor = None
clip = None

import cv2


class NeuralInterpreter(Interpreter):
    TYPE = "Neural"

    def __init__(self, model_name, data_type):
        from openvino.inference_engine import IENetwork, IECore

        super().__init__(model_name, data_type)
        model_name = Path(model_name)
        # can use to test on PC
        # device = "CPU"
        device = "MYRIAD"
        model_xml = model_name.with_suffix(".xml")
        model_bin = model_name.with_suffix(".bin")
        ie = IECore()
        ie.set_config({}, device)
        net = ie.read_network(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))
        self.input_shape = net.input_info[self.input_blob].input_data.shape

        net.batch_size = 1
        self.exec_net = ie.load_network(network=net, device_name=device)
        self.preprocess_fn = inc3_preprocess

    def predict(self, input_x):
        if input_x is None:
            return None
        input_x = np.float32(input_x)
        channels_last = input_x.shape[-1] == 3
        if channels_last:
            input_x = np.moveaxis(input_x, 3, 1)
        # input_x = np.transpose(input_x, axes=[3, 1, 2])
        # input_x = np.array([[rearranged_arr]])
        res = self.exec_net.infer(inputs={self.input_blob: input_x})
        res = res[self.out_blob]
        return res

    def shape(self):
        return self.input_shape


class LiteInterpreter(Interpreter):
    TYPE = "TFLite"

    def __init__(self, model_name, data_type):
        super().__init__(model_name, data_type)

        import tflite_runtime.interpreter as tflite

        model_name = Path(model_name)
        model_name = model_name.with_suffix(".tflite")
        self.interpreter = tflite.Interpreter(str(model_name))

        self.interpreter.allocate_tensors()  # Needed before execution!

        self.output = self.interpreter.get_output_details()[
            0
        ]  # Model has single output.
        self.input = self.interpreter.get_input_details()[0]  # Model has single input.
        self.preprocess_fn = inc3_preprocess

    def predict(self, input_x):
        start = time.time()
        input_x = np.float32(input_x)
        # input_x = input_x[np.newaxis, :]

        self.interpreter.set_tensor(self.input["index"], input_x)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output["index"])
        logging.info("taken %s to predict", time.time() - start)
        return pred

    def shape(self):
        return self.input["shape"]


def inc3_preprocess(x):
    x /= 127.5
    x -= 1.0
    return x


def get_full_classifier(model, data_type):
    from ml_tools.kerasmodel import KerasModel

    """
    Returns a classifier object, which is created on demand.
    This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
    """
    t0 = datetime.now()
    logging.info("classifier loading")
    classifier = KerasModel()
    classifier.load_model(model.model_file, weights=model.model_weights)
    classifier.type = data_type
    logging.info("classifier loaded ({})".format(datetime.now() - t0))

    return classifier


def get_classifier(model, data_type):
    # model_name, type = os.path.splitext(model.model_file)

    logging.info(
        "Loading %s of type %s with datatype %s",
        model.model_file,
        model.type,
        data_type,
    )
    if model.type == LiteInterpreter.TYPE:
        classifier = LiteInterpreter(model.model_file, data_type)
    elif model.type == NeuralInterpreter.TYPE:
        classifier = NeuralInterpreter(model.model_file, data_type)
    elif model.type == ForestModel.TYPE:
        classifier = ForestModel(model.model_file, data_type)
    else:
        classifier = get_full_classifier(model, data_type)
    return classifier


def run_classifier(
    frame_queue, config, thermal_config, headers, classify=True, detect_after=None
):
    init_logging()
    try:
        pi_classifier = PiClassifier(
            config, thermal_config, headers, classify, detect_after
        )
        while True:
            frame = frame_queue.get()
            if isinstance(frame, str):
                if frame == STOP_SIGNAL:
                    logging.info("PiClassifier received stop signal")
                    pi_classifier.disconnected()
                    return
                if frame == "skip":
                    pi_classifier.skip_frame()
                elif frame == SNAPSHOT_SIGNAL:
                    pi_classifier.take_snapshot()
            else:
                pi_classifier.process_frame(frame[0], frame[1])
    except:
        logging.error("Error running classifier restarting ..", exc_info=True)


predictions = None


class PiClassifier(Processor):
    """Classifies frames from leptond"""

    NUM_CONCURRENT_TRACKS = 1
    DEBUG_EVERY = 20
    MAX_CONSEC = 1
    # after every MAX_CONSEC frames skip this many frames
    # this gives the cpu a break
    SKIP_FRAMES = 30
    PREDICT_EVERY = 40

    def __init__(
        self,
        config,
        thermal_config,
        headers,
        classify,
        detect_after=None,
        preview_type=None,
        constant_recorder=True,
    ):
        self.constant_recorder = None
        self._output_dir = thermal_config.recorder.output_dir
        self.headers = headers
        super().__init__()
        self.frame_num = 0
        self.clip = None
        self.enable_per_track_information = False
        self.rolling_track_classify = {}
        self.skip_classifying = 0
        self.classified_consec = 0
        self.classify = classify
        self.config = config
        self.predictions = None
        self.process_time = 0
        self.tracking_time = 0
        self.identify_time = 0
        self.total_time = 0
        self.rec_time = 0
        self.tracking = None
        self.tracking_events = thermal_config.motion.tracking_events
        self.bluetooth_beacons = thermal_config.motion.bluetooth_beacons
        self.preview_frames = thermal_config.recorder.preview_secs * headers.fps

        self.fps_timer = SlidingWindow((headers.fps), np.float32)
        self.preview_type = preview_type
        self.max_keep_frames = None if preview_type else 0
        if thermal_config.recorder.disable_recordings:
            self.recorder = DummyRecorder(
                thermal_config, headers, on_recording_stopping
            )

        if headers.model == "IR":
            logging.info("Running on IR")
            PiClassifier.SKIP_FRAMES = 3
            self.track_extractor = IRTrackExtractor(
                self.config.tracking,
                config.use_opt_flow,
                self.config.classify.cache_to_disk,
                verbose=config.verbose,
                calc_stats=False,
                scale=0.25,
                on_trapped=on_track_trapped,
                update_background=False,
            )
            self.tracking_config = self.track_extractor.config

            self.type = "IR"
            if not thermal_config.recorder.disable_recordings:
                self.recorder = IRRecorder(
                    thermal_config, headers, on_recording_stopping
                )
            self.snapshot_recorder = IRRecorder(
                thermal_config, headers, on_recording_stopping, name="IR Snapshot"
            )
            self.motion_detector = IRMotionDetector(
                thermal_config,
                headers,
            )
            if constant_recorder:
                self.constant_recorder = IRRecorder(
                    thermal_config,
                    headers,
                    on_recording_stopping,
                    name="IR Constant",
                    constant_recorder=True,
                )
        else:
            logging.info("Running on Thermal")
            self.track_extractor = ClipTrackExtractor(
                self.config.tracking,
                self.config.use_opt_flow,
                self.config.classify.cache_to_disk,
                calc_stats=False,
            )
            self.tracking_config = self.track_extractor.config

            self.type = "thermal"
            if not thermal_config.recorder.disable_recordings:
                self.recorder = CPTVRecorder(
                    thermal_config, headers, on_recording_stopping
                )
            self.snapshot_recorder = CPTVRecorder(
                thermal_config, headers, on_recording_stopping, name="CPTV Snapshot"
            )
            self.motion_detector = CPTVMotionDetector(
                thermal_config,
                self.tracking_config.motion.dynamic_thresh,
                headers,
                detect_after=detect_after,
            )
            if constant_recorder:
                self.constant_recorder = CPTVRecorder(
                    thermal_config, headers, on_recording_stopping, name="CPTV Constant"
                )
        edge = self.tracking_config.edge_pixels
        self.crop_rectangle = tools.Rectangle(
            edge, edge, headers.res_x - 2 * edge, headers.res_y - 2 * edge
        )
        global track_extractor
        track_extractor = self.track_extractor
        self.motion = thermal_config.motion
        self.min_frames = thermal_config.recorder.min_secs * headers.fps
        self.max_frames = thermal_config.recorder.max_secs * headers.fps
        if thermal_config.recorder.disable_recordings:
            self.recorder = DummyRecorder(
                thermal_config, headers, on_recording_stopping
            )
        if thermal_config.throttler.activate:
            self.recorder = ThrottledRecorder(
                self.recorder, thermal_config, headers, on_recording_stopping
            )
        self.meta_dir = os.path.join(thermal_config.recorder.output_dir)
        if not os.path.exists(self.meta_dir):
            os.makedirs(self.meta_dir)

        if self.classify:
            model = config.classify.models[0]
            self.classifier = get_classifier(model, self.type)

            if self.classifier.TYPE == ForestModel.TYPE:
                self.last_x_frames = 5 * headers.fps
                self.frames_per_classify = self.last_x_frames
                PiClassifier.SKIP_FRAMES = 30
                # probably could be even more

            else:
                # self.preprocess_fn = self.get_preprocess_fn()

                self.last_x_frames = 1
                self.frames_per_classify = (
                    self.classifier.params.square_width
                    * self.classifier.params.square_width
                )
                if self.frames_per_classify > 1:
                    self.last_x_frames = self.frames_per_classify * 2

            self.max_keep_frames = (
                self.frames_per_classify * 2 if not preview_type else None
            )
            self.predictions = Predictions(self.classifier.labels, model)
            self.num_labels = len(self.classifier.labels)
            logging.info("Labels are %s ", self.classifier.labels)
            global predictions
            predictions = self.predictions
            try:
                self.fp_index = self.classifier.labels.index("false-positive")
            except ValueError:
                self.fp_index = None
            self.startup_classifier()

    #
    # def get_preprocess_fn(self):
    #     import tensorflow as tf
    #
    #     pretrained_model = self.classifier.params.model_name
    #     if pretrained_model == "resnet":
    #         return tf.keras.applications.resnet.preprocess_input
    #
    #     elif pretrained_model == "resnetv2":
    #         return tf.keras.applications.resnet_v2.preprocess_input
    #
    #     elif pretrained_model == "resnet152":
    #         return tf.keras.applications.resnet.preprocess_input
    #
    #     elif pretrained_model == "vgg16":
    #         return tf.keras.applications.vgg16.preprocess_input
    #
    #     elif pretrained_model == "vgg19":
    #         return tf.keras.applications.vgg19.preprocess_input
    #
    #     elif pretrained_model == "mobilenet":
    #         return tf.keras.applications.mobilenet_v2.preprocess_input
    #
    #     elif pretrained_model == "densenet121":
    #         return tf.keras.applications.densenet.preprocess_input
    #
    #     elif pretrained_model == "inceptionresnetv2":
    #         return tf.keras.applications.inception_resnet_v2.preprocess_input
    #     elif pretrained_model == "inceptionv3":
    #         return tf.keras.applications.inception_v3.preprocess_input
    #     logging.warn(
    #         "pretrained model %s has no preprocessing function", pretrained_model
    #     )
    #     return None

    def new_clip(self, preview_frames):
        self.clip = Clip(
            self.tracking_config,
            "stream",
            model=self.headers.model,
            type=self.type,
            calc_stats=False,
            fps=self.headers.fps,
        )
        global clip
        clip = self.clip
        self.clip.video_start_time = datetime.now()
        self.clip.num_preview_frames = self.preview_frames
        self.clip.set_res(self.res_x, self.res_y)
        self.clip.set_frame_buffer(
            self.tracking_config.high_quality_optical_flow,
            self.config.classify.cache_to_disk,
            self.config.use_opt_flow,
            True,
            self.max_keep_frames,
        )
        edge_pixels = self.tracking_config.edge_pixels

        self.clip.update_background(self.motion_detector.background.copy())
        self.clip._background_calculated()
        # no need to retrack all of preview
        background_frames = None
        track_frames = self.type != "IR"
        self.track_extractor.start_tracking(
            self.clip,
            preview_frames,
            track_frames=track_frames,
            background_alg=self.motion_detector._background,
            # background_frame=clip.background,
            # background_frames=background_frames,
        )

    def startup_classifier(self):
        # classifies an empty frame to force loading of the model into memory
        in_shape = self.classifier.shape()[1:]
        p_frame = np.zeros((1, *in_shape), np.float32)
        self.classifier.predict(p_frame)

    def get_active_tracks(self):
        """
        Gets current clips active_tracks and returns the top NUM_CONCURRENT_TRACKS order by priority
        """
        active_tracks = self.clip.active_tracks
        active_tracks = [track for track in active_tracks if len(track) >= 8]
        filtered = []
        for track in active_tracks:
            pred = None
            if self.predictions is not None:
                pred = self.predictions.prediction_for(track.get_id())
            if pred is not None:
                if (
                    pred.last_frame_classified is not None
                    and self.clip.current_frame - pred.last_frame_classified
                    < PiClassifier.PREDICT_EVERY
                ):
                    logging.debug(
                        "Skipping %s as predicted %s and now at %s",
                        track,
                        pred.last_frame_classified,
                        self.clip.current_frame,
                    )
                    continue

            filtered.append(track)
        active_tracks = filtered
        if (
            len(active_tracks) <= PiClassifier.NUM_CONCURRENT_TRACKS
            or not self.classify
        ):
            return active_tracks
        active_predictions = []
        for track in active_tracks:
            prediction = self.predictions.get_or_create_prediction(track, keep_all=True)
            active_predictions.append(prediction)

        top_priority = sorted(
            active_predictions,
            key=lambda i: i.get_priority(self.clip.current_frame),
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
        new_prediction = False
        if len(active_tracks) == 0:
            return False

        logging.info("Identifying %s", active_tracks)
        for i, track in enumerate(active_tracks):
            regions = []
            track_prediction = self.predictions.get_or_create_prediction(
                track, keep_all=True
            )
            frames, prediction, mass = self.classifier.predict_track(
                clip,
                track,
                last_x_frames=self.last_x_frames,
                scale=self.track_extractor.scale,
                frames_per_classify=self.frames_per_classify,
            )
            if prediction is None:
                track_prediction.last_frame_classified = self.clip.current_frame
                continue
            for p, m in zip(prediction, mass):
                track_prediction.classified_frames(frames, p, m)
            logging.info(
                "Track %s is predicted as %s", track, track_prediction.get_prediction()
            )

            if self.tracking_events:
                if track_prediction.predicted_tag() != "false-positive":
                    track_prediction.tracking = True
                    self.tracking = track
                    track_prediction.normalize_score()
                    self.service.tracking(
                        track_prediction.predicted_tag(),
                        track_prediction.max_score,
                        track.bounds_history[-1].to_ltrb(),
                        True,
                    )
                elif track_prediction.tracking:
                    track_prediction.tracking = False
                    self.tracking = None
                    track_prediction.normalize_score()
                    self.service.tracking(
                        track_prediction.predicted_tag(),
                        track_prediction.max_score,
                        track.bounds_history[-1].to_ltrb(),
                        False,
                    )

            new_prediction = True
        if self.bluetooth_beacons:
            if new_prediction:
                active_predictions = []
                for track in self.clip.active_tracks:
                    track_prediction = self.predictions.prediction_for(track.get_id())
                    if track_prediction:
                        active_predictions.append(track_prediction)
                beacon.classification(active_predictions)
        return True

    def get_recent_frame(self, last_frame=None):
        # save us having to lock if we dont have a different frame
        if last_frame is not None and self.motion_detector.num_frames == last_frame:
            return None, None, last_frame
        last_frame = self.motion_detector.get_recent_frame()
        if self.clip:
            if last_frame is None:
                return None
            track_meta = []
            tracks = clip.active_tracks
            for track in tracks:
                pred = None
                if self.predictions:
                    pred = {self.predictions.model.id: self.predictions}
                meta = track.get_metadata(pred)
                last_pos = meta["positions"][-1].copy()
                # if self.track_extractor.scale is not None:
                # last_pos.rescale(1 / self.track_extractor.scale)
                meta["positions"] = [last_pos]
                track_meta.append(meta)

            return last_frame, track_meta, self.motion_detector.num_frames
        else:
            return (
                last_frame,
                {},
                self.motion_detector.num_frames,
            )

    def disconnected(self):
        self.motion_detector.disconnected()
        self.recorder.force_stop()
        self.snapshot_recorder.force_stop()
        self.constant_recorder.force_stop()

        self.end_clip()
        self.service.quit()

    def skip_frame(self):
        self.skip_classifying -= 1

        if self.clip:
            self.clip.current_frame += 1

    def take_snapshot(self):
        started = self.snapshot_recorder.start_recording(
            None, [], self.motion_detector.temp_thresh, time.time()
        )
        if not started:
            logging.info("Already taking snapshot recording")
            return False
        logging.info("Taking new snapshot recorder")
        self.snapshot_recorder.write_until = 2 * self.headers.fps
        return True

    def process_frame(self, lepton_frame, received_at):
        import time

        start = time.time()
        self.motion_detector.process_frame(lepton_frame)
        self.process_time += time.time() - start
        if self.snapshot_recorder.recording:
            self.snapshot_recorder.process_frame(False, lepton_frame, received_at)
        if self.constant_recorder is not None and self.motion_detector.can_record():
            if self.constant_recorder.recording:
                self.constant_recorder.process_frame(True, lepton_frame, received_at)
            else:
                logging.info("Starting new constant recorder")
                self.constant_recorder.start_recording(
                    self.motion_detector.background,
                    [],
                    self.motion_detector.temp_thresh,
                    time.time(),
                )
        if not self.recorder.recording and self.motion_detector.movement_detected:
            s_r = time.time()
            preview_frames = self.motion_detector.preview_frames()
            bak = self.motion_detector.background
            recording = self.recorder.start_recording(
                self.motion_detector.background,
                preview_frames,
                self.motion_detector.temp_thresh,
                received_at,
            )
            self.rec_time += time.time() - s_r
            if recording:
                if self.bluetooth_beacons:
                    beacon.recording()
                t_start = time.time()
                self.new_clip(preview_frames)
                self.tracking_time += time.time() - t_start

        if self.recorder.recording:
            t_start = time.time()
            self.track_extractor.process_frame(self.clip, lepton_frame)
            self.tracking_time += time.time() - t_start
            s_r = time.time()
            if self.tracking is not None:
                tracking = self.tracking in self.clip.active_tracks
                score = 0
                prediction = ""
                if self.classify:
                    track_prediction = self.predictions.prediction_for(
                        self.tracking.get_id()
                    )
                    prediction = track_prediction.predicted_tag()
                    score = track_prediction.max_score
                self.service.tracking(
                    prediction,
                    score,
                    self.tracking.bounds_history[-1].to_ltrb(),
                    tracking,
                )

                if not tracking:
                    if self.classify:
                        track_prediction.tracking = False
                    self.tracking = None

            self.recorder.process_frame(
                self.motion_detector.movement_detected, lepton_frame, received_at
            )
            self.rec_time += time.time() - s_r
            if self.classify:
                if self.motion_detector.calibrating or self.clip.on_preview():
                    self.skip_classifying = PiClassifier.SKIP_FRAMES
                    self.classified_consec = 0
                elif (
                    self.classify
                    and self.motion_detector.calibrating is False
                    and self.clip.active_tracks
                    and self.skip_classifying <= 0
                    and not self.clip.on_preview()
                ):
                    id_start = time.time()
                    identified = self.identify_last_frame()
                    if identified:
                        self.identify_time += time.time() - id_start
                        self.classified_consec += 1
                        if self.classified_consec == PiClassifier.MAX_CONSEC:
                            self.skip_classifying = PiClassifier.SKIP_FRAMES
                            self.classified_consec = 0
                else:
                    self.classified_consec = 0
            elif self.tracking is None and self.tracking_events:
                active_tracks = self.get_active_tracks()

                active_tracks = [
                    track
                    for track in active_tracks
                    if len(track) > 10 and track.last_bound.mass > 16
                ]
                logging.debug(
                    "got active tracks bigger than 16 and longer than 10 frames %s",
                    active_tracks,
                )

                active_tracks = sorted(
                    active_tracks,
                    key=lambda track: track.last_mass,
                    reverse=True,
                )

                if len(active_tracks) > 0:
                    logging.debug(
                        "tracking by biggest mass %s",
                        active_tracks[0],
                    )
                    self.tracking = active_tracks[0]

        elif self.clip is not None:
            self.end_clip()

        self.skip_classifying -= 1
        self.frame_num += 1
        self.total_time += time.time() - start
        if (
            self.motion_detector.can_record()
            and self.frame_num % PiClassifier.DEBUG_EVERY == 0
        ):
            average = np.mean(self.fps_timer.get_frames())

            logging.info(
                "tracking %s %% process %s %%  identify %s %% rec %s %% fps %s/sec  cpu %s memory %s behind by %s seconds",
                round(100 * self.tracking_time / self.total_time, 3),
                round(100 * self.process_time / self.total_time, 3),
                round(100 * self.identify_time / self.total_time, 3),
                round(100 * self.rec_time / self.total_time, 3),
                round(1 / average),
                psutil.cpu_percent(),
                psutil.virtual_memory()[2],
                time.time() - received_at,
            )
            self.tracking_time = 0
            self.process_time = 0
            self.identify_time = 0
            self.total_time = 0
            self.rec_time = 0
        self.fps_timer.add(time.time() - start)

    def create_mp4(self):
        previewer = Previewer(self.config, self.preview_type)
        previewer.export_clip_preview(
            os.path.join(self.output_dir, self.clip.get_id() + ".mp4"),
            self.clip,
            self.predictions if self.classify else None,
        )

    def end_clip(self):
        if self.clip:
            if self.preview_type:
                self.create_mp4()
            logging.debug(
                "Ending clip with %s tracks post filtering", len(self.clip.tracks)
            )
            if self.classify:
                for t_id, prediction in self.predictions.prediction_per_track.items():
                    if prediction.max_score:
                        logging.info(
                            "Clip {} {} {}".format(
                                self.clip.get_id(),
                                t_id,
                                prediction.description(),
                            )
                        )
                self.predictions.clear_predictions()
            self.clip = None
            self.tracking = None
        global clip
        clip = None

    @property
    def res_x(self):
        return self.headers.res_x

    @property
    def res_y(self):
        return self.headers.res_y

    @property
    def output_dir(self):
        return self._output_dir


def on_track_trapped(track):
    track.trap_reported = True
    # GP could make a prediction here

    global predictions

    tag = None
    if predictions is not None:
        pred = predictions.prediction_for(track.get_id())
        if pred is not None:
            tag = pred.predicted_tag()
            track.trap_tag = tag
    logging.warn("Trapped track %s with tag %s", track, tag)
    trapped_event(tag)


def on_recording_stopping(filename):
    global clip, track_extractor, predictions

    if predictions is not None:
        for track_prediction in predictions.prediction_per_track.values():
            track_prediction.normalize_score()

    if clip and track_extractor:
        track_extractor.apply_track_filtering(clip)
        # filter criteria has been scaled so resize after
        # if track_extractor.scale is not None:
        #     for track in clip.tracks:
        #         for r in track.bounds_history:
        #             # bring back to orignal size
        #             r.rescale(1 / track_extractor.scale)

        meta_name = os.path.splitext(filename)[0] + ".txt"
        logging.debug("saving meta to %s", meta_name)
        predictions_per_model = None

        if predictions is not None:
            predictions_per_model = {predictions.model.id: predictions}
        meta_data = clip.get_metadata(predictions_per_model)
        meta_data["algorithm"] = {}
        meta_data["algorithm"]["tracker_version"] = track_extractor.VERSION

        if predictions is not None:
            meta_data["models"] = [predictions.model.as_dict()]
            meta_data["algorithm"]["model_name"] = predictions.model.name

        with open(meta_name, "w") as f:
            json.dump(meta_data, f, indent=4, cls=CustomJSONEncoder)
