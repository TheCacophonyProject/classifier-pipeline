from datetime import datetime
import json
import logging
import os
import time
import psutil
import numpy as np
import logging
from pathlib import Path
from PIL import ImageDraw

from classify.trackprediction import Predictions
from load.clip import Clip
from load.cliptrackextractor import ClipTrackExtractor
from ml_tools.preprocess import preprocess_segment
from ml_tools.previewer import Previewer, add_last_frame_tracking
from ml_tools import tools
from .cptvrecorder import CPTVRecorder
from .throttledrecorder import ThrottledRecorder
from .dummyrecorder import DummyRecorder

from .motiondetector import MotionDetector
from .processor import Processor
from ml_tools.preprocess import (
    preprocess_frame,
    preprocess_movement,
)

from ml_tools.interpreter import Interpreter
from ml_tools.logs import init_logging
from ml_tools.hyperparams import HyperParams
from ml_tools.tools import CustomJSONEncoder
from track.region import Region
from . import beacon

STOP_SIGNAL = "stop"
SKIP_SIGNAL = "skip"
track_extractor = None
clip = None


class NeuralInterpreter(Interpreter):
    def __init__(self, model_name):
        from openvino.inference_engine import IENetwork, IECore

        super().__init__(model_name)

        # can use to test on PC
        # device = "CPU"
        device = "MYRIAD"
        model_xml = model_name + ".xml"
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        ie = IECore()
        net = IENetwork(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        net.batch_size = 1
        self.exec_net = ie.load_network(network=net, device_name=device)

    def predict(self, input_x):
        if input_x is None:
            return None
        rearranged_arr = np.transpose(input_x, axes=[2, 0, 1])
        input_x = np.array([[rearranged_arr]])
        res = self.exec_net.infer(inputs={self.input_blob: input_x})
        res = res[self.out_blob]
        return res[0]


class LiteInterpreter(Interpreter):
    def __init__(self, model_name):
        super().__init__(model_name)

        import tensorflow as tf

        model_name = Path(model_name)
        model_name = model_name.with_suffix(".tflite")
        self.interpreter = tf.lite.Interpreter(str(model_name))

        self.interpreter.allocate_tensors()  # Needed before execution!

        self.output = self.interpreter.get_output_details()[
            0
        ]  # Model has single output.
        self.input = self.interpreter.get_input_details()[0]  # Model has single input.

    def predict(self, input_x):
        start = time.time()
        input_x = np.float32(input_x)
        input_x = input_x[np.newaxis, :]

        self.interpreter.set_tensor(self.input["index"], input_x)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output["index"])[0]
        logging.info("taken %s to predict", time.time() - start)

        return pred

    def shape(self):
        return self.input["shape"]


def get_full_classifier(model):
    from ml_tools.kerasmodel import KerasModel

    """
    Returns a classifier object, which is created on demand.
    This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
    """
    t0 = datetime.now()
    logging.info("classifier loading")
    classifier = KerasModel()
    classifier.load_model(model.model_file, weights=model.model_weights)
    logging.info("classifier loaded ({})".format(datetime.now() - t0))

    return classifier


def get_classifier(model):
    model_name, model_type = os.path.splitext(model.model_file)
    logging.info("Loading %s", model_name)
    if model_type == ".tflite":
        classifier = LiteInterpreter(model_name)
    elif model_type == ".xml":
        classifier = NeuralInterpreter(model_name)
    else:
        classifier = get_full_classifier(
            model,
        )
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
            else:
                pi_classifier.process_frame(frame)
    except:
        logging.error("Error running classifier restarting ..", exc_info=True)


predictions = None


class PiClassifier(Processor):
    """Classifies frames from leptond"""

    NUM_CONCURRENT_TRACKS = 1
    DEBUG_EVERY = 100
    MAX_CONSEC = 1
    # after every MAX_CONSEC frames skip this many frames
    # this gives the cpu a break
    SKIP_FRAMES = 10

    def __init__(
        self,
        config,
        thermal_config,
        headers,
        classify,
        detect_after=None,
        preview_type=None,
    ):
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
        edge = self.config.tracking.edge_pixels
        self.crop_rectangle = tools.Rectangle(
            edge, edge, headers.res_x - 2 * edge, headers.res_y - 2 * edge
        )
        self.preview_type = preview_type
        self.max_keep_frames = None if preview_type else 0
        self.track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow,
            self.config.classify.cache_to_disk,
            calc_stats=False,
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
        else:
            self.recorder = CPTVRecorder(thermal_config, headers, on_recording_stopping)
        if thermal_config.throttler.activate:
            self.recorder = ThrottledRecorder(
                self.recorder, thermal_config, headers, on_recording_stopping
            )

        self.motion_detector = MotionDetector(
            thermal_config,
            self.config.tracking.motion.dynamic_thresh,
            headers,
            detect_after=detect_after,
        )
        self.meta_dir = os.path.join(thermal_config.recorder.output_dir)
        if not os.path.exists(self.meta_dir):
            os.makedirs(self.meta_dir)

        if self.classify:
            model = config.classify.models[0]
            self.classifier = get_classifier(model)
            self.frames_per_classify = (
                self.classifier.params.square_width
                * self.classifier.params.square_width
            )
            self.max_keep_frames = (
                self.frames_per_classify * 2 if not preview_type else None
            )
            self.predictions = Predictions(self.classifier.labels, model)
            self.num_labels = len(self.classifier.labels)
            global predictions
            predictions = self.predictions
            try:
                self.fp_index = self.classifier.labels.index("false-positive")
            except ValueError:
                self.fp_index = None
            self.preprocess_fn = self.get_preprocess_fn()
            self.startup_classifier()

    def get_preprocess_fn(self):
        import tensorflow as tf

        pretrained_model = self.classifier.params.model_name
        if pretrained_model == "resnet":
            return tf.keras.applications.resnet.preprocess_input

        elif pretrained_model == "resnetv2":
            return tf.keras.applications.resnet_v2.preprocess_input

        elif pretrained_model == "resnet152":
            return tf.keras.applications.resnet.preprocess_input

        elif pretrained_model == "vgg16":
            return tf.keras.applications.vgg16.preprocess_input

        elif pretrained_model == "vgg19":
            return tf.keras.applications.vgg19.preprocess_input

        elif pretrained_model == "mobilenet":
            return tf.keras.applications.mobilenet_v2.preprocess_input

        elif pretrained_model == "densenet121":
            return tf.keras.applications.densenet.preprocess_input

        elif pretrained_model == "inceptionresnetv2":
            return tf.keras.applications.inception_resnet_v2.preprocess_input
        elif pretrained_model == "inceptionv3":
            return tf.keras.applications.inception_v3.preprocess_input
        logging.warn(
            "pretrained model %s has no preprocessing function", pretrained_model
        )
        return None

    def new_clip(self, preview_frames):
        self.clip = Clip(self.config.tracking, "stream", model=self.headers.model)
        global clip
        clip = self.clip
        self.clip.video_start_time = datetime.now()
        self.clip.num_preview_frames = self.preview_frames
        self.clip.set_res(self.res_x, self.res_y)
        self.clip.set_frame_buffer(
            self.config.tracking.high_quality_optical_flow,
            self.config.classify.cache_to_disk,
            self.config.use_opt_flow,
            True,
            self.max_keep_frames,
        )
        edge_pixels = self.config.tracking.edge_pixels

        self.clip.update_background(self.motion_detector.background.copy())
        self.clip._background_calculated()
        # no need to retrack all of preview
        for frame in preview_frames[-9:]:
            self.track_extractor.process_frame(self.clip, frame.pix.copy())

    def startup_classifier(self):
        # classifies an empty frame to force loading of the model into memory

        p_frame = np.zeros((160, 160, 3), np.float32)
        self.classifier.predict(p_frame)

    def get_active_tracks(self):
        """
        Gets current clips active_tracks and returns the top NUM_CONCURRENT_TRACKS order by priority
        """
        active_tracks = self.clip.active_tracks
        active_tracks = [track for track in active_tracks if len(track) > 10]
        if (
            len(active_tracks) <= PiClassifier.NUM_CONCURRENT_TRACKS
            or not self.classify
        ):
            return active_tracks
        active_predictions = []
        for track in active_tracks:
            prediction = self.predictions.get_or_create_prediction(
                track, keep_all=False
            )
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
        for i, track in enumerate(active_tracks):
            regions = []
            track_prediction = self.predictions.get_or_create_prediction(
                track, keep_all=False
            )
            regions = track.bounds_history[-self.frames_per_classify * 2 :]
            frames = self.clip.frame_buffer.get_last_x(len(regions))
            if frames is None:
                return
            indices = np.random.choice(
                len(regions),
                min(self.frames_per_classify, len(regions)),
                replace=False,
            )
            indices.sort()
            frames = np.array(frames)[indices]
            regions = np.array(regions)[indices]

            refs = []
            segment_data = []
            mass = 0
            for frame, region in zip(frames, regions):
                refs.append(np.median(frame.thermal))
                thermal_reference = np.median(frame.thermal)
                segment_data.append(frame.crop_by_region(region))
                mass += region.mass

            params = self.classifier.params
            preprocessed = preprocess_movement(
                segment_data,
                params.square_width,
                params.frame_size,
                red_type=params.red_type,
                green_type=params.green_type,
                blue_type=params.blue_type,
                preprocess_fn=self.preprocess_fn,
                reference_level=refs,
                keep_edge=params.keep_edge,
            )
            if preprocessed is None:
                track_prediction.last_frame_classified = self.clip.current_frame
                # probably a false positive or bad frame data
                continue
            prediction = self.classifier.predict(preprocessed)
            track_prediction.classified_frame(self.clip.current_frame, prediction, mass)
            logging.debug(
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

    def get_recent_frame(self):
        if self.clip:
            last_frame = self.motion_detector.get_recent_frame()
            if last_frame is None:
                return None
            track_meta = []
            tracks = clip.active_tracks
            for track in tracks:
                pred = None
                if self.predictions:
                    pred = {self.predictions.model.id: self.predictions}
                meta = track.get_metadata(pred)
                meta["positions"] = meta["positions"][-1:]
                track_meta.append(meta)

            return last_frame, track_meta, self.motion_detector.num_frames
        else:
            return (
                self.motion_detector.get_recent_frame(),
                {},
                self.motion_detector.num_frames,
            )

    def disconnected(self):
        self.motion_detector.disconnected()
        self.recorder.force_stop()
        self.end_clip()
        self.service.quit()

    def skip_frame(self):
        self.skip_classifying -= 1

        if self.clip:
            self.clip.current_frame += 1

    def process_frame(self, lepton_frame):
        start = time.time()
        self.motion_detector.process_frame(lepton_frame)
        self.process_time += time.time() - start

        if not self.recorder.recording and self.motion_detector.movement_detected:
            s_r = time.time()
            preview_frames = self.motion_detector.thermal_window.get_frames()[:-1]

            recording = self.recorder.start_recording(
                self.motion_detector.background,
                preview_frames,
                self.motion_detector.temp_thresh,
                lepton_frame.received_at,
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
            self.track_extractor.process_frame(
                self.clip, lepton_frame.pix, self.motion_detector.ffc_affected
            )
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
                self.motion_detector.movement_detected, lepton_frame
            )
            self.rec_time += time.time() - s_r
            if self.classify:
                if self.motion_detector.ffc_affected or self.clip.on_preview():
                    self.skip_classifying = PiClassifier.SKIP_FRAMES
                    self.classified_consec = 0
                elif (
                    self.classify
                    and self.motion_detector.ffc_affected is False
                    and self.clip.active_tracks
                    and self.skip_classifying <= 0
                    and not self.clip.on_preview()
                ):
                    id_start = time.time()
                    self.identify_last_frame()
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
            logging.info(
                "tracking {}% process {}%  identify {}% rec{}%s fps {}/sec  cpu % {} memory % {} behind by {} seconds".format(
                    round(100 * self.tracking_time / self.total_time, 3),
                    round(100 * self.process_time / self.total_time, 3),
                    round(100 * self.identify_time / self.total_time, 3),
                    round(100 * self.rec_time / self.total_time, 3),
                    round(self.total_time / PiClassifier.DEBUG_EVERY, 2),
                    psutil.cpu_percent(),
                    psutil.virtual_memory()[2],
                    time.time() - lepton_frame.received_at,
                )
            )
            self.tracking_time = 0
            self.process_time = 0
            self.identify_time = 0
            self.motion_detector.rec_time = 0
            self.total_time = 0
            self.rec_time = 0

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
                "Ending clip with %s tracks pre filtering", len(self.clip.tracks)
            )
            if self.classify:
                for _, prediction in self.predictions.prediction_per_track.items():
                    if prediction.max_score:
                        logging.info(
                            "Clip {} {}".format(
                                self.clip.get_id(),
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


def on_recording_stopping(filename):
    global clip, track_extractor, predictions
    if predictions is not None:
        for track_prediction in predictions.prediction_per_track.values():
            track_prediction.normalize_score()

    if clip and track_extractor:
        track_extractor.apply_track_filtering(clip)
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
