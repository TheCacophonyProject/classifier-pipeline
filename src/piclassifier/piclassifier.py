from datetime import datetime, timedelta
import json
import logging
import os
import time
import psutil
import numpy as np
import logging
from track.clip import Clip
from track.track import ThumbInfo

from .motiondetector import SlidingWindow
from .processor import Processor

from ml_tools.logs import init_logging
from ml_tools.rectangle import Rectangle
from . import beacon

from piclassifier.trapcontroller import trigger_trap
from piclassifier.attiny import set_recording_state
from pathlib import Path
from ml_tools.imageprocessing import normalize
from functools import partial
from piclassifier import utils

SNAPSHOT_SIGNAL = "snap"
STOP_SIGNAL = "stop"
SKIP_SIGNAL = "skip"
track_extractor = None
clip = None
fp_model = None
classifier = None


def run_classifier(
    frame_queue, config, thermal_config, headers, classify=True, detect_after=None
):
    init_logging()
    pi_classifier = None
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
        if pi_classifier is not None:
            pi_classifier.disconnected()
        return


predictions = None
use_low_power_mode = False


class PiClassifier(Processor):
    """Classifies frames from leptond"""

    NUM_CONCURRENT_TRACKS = 1
    DEBUG_EVERY = 20
    MAX_CONSEC = 1
    # after every MAX_CONSEC frames skip this many frames
    # this gives the cpu a break

    # try classify a non fp track every X frames
    SKIP_FRAMES = 25
    # only do another full classification on the same track after this many frames
    PREDICT_EVERY = 40

    # run fp model predictions every X frames
    FP_MODEL_SKIP_FRAMES = 10
    # only do another fp classification on the same track after this many frames
    FP_PREDICT_EVERY = 30

    def __init__(
        self,
        config,
        thermal_config,
        headers,
        classify,
        detect_after=None,
        preview_type=None,
    ):
        self.constant_recorder = None
        global output_dir
        output_dir = thermal_config.recorder.output_dir
        # ensure thumb dir is made
        thumbnail_dir = Path(output_dir) / "thumbnails"
        thumbnail_dir.mkdir(exist_ok=True)

        self._output_dir = thermal_config.recorder.output_dir
        self.headers = headers
        self.classifier = None
        self.classifier_initialised = False
        self.fp_model = None
        self.frame_num = 0
        self.clip = None
        self.prev_clip = None
        self.enable_per_track_information = False
        self.rolling_track_classify = {}
        self.next_classify_frame = 0
        self.next_fp_classification_frame = 0
        self.classified_consec = 0
        self.classify = classify
        self.config = config
        self.predictions = {}
        self.process_time = 0
        self.tracking_time = 0
        self.identify_time = 0
        self.fp_identify_time = 0
        self.total_time = 0
        self.rec_time = 0
        self.fp_time = 0
        self.monitored_tracks = {}
        self.recording = False
        self.tracking_events = thermal_config.motion.tracking_events
        self.bluetooth_beacons = thermal_config.motion.bluetooth_beacons
        self.preview_frames = thermal_config.recorder.preview_secs * headers.fps
        self.do_tracking = thermal_config.motion.do_tracking
        self.fps_timer = SlidingWindow((headers.fps * 3), np.float32)
        self.preview_type = preview_type
        self.max_keep_frames = None if preview_type else 0
        self.track_extractor = None
        self.use_low_power_mode = thermal_config.recorder.use_low_power_mode
        global use_lower_power_mode
        use_low_power_mode = self.use_low_power_mode
        if not use_low_power_mode:
            # clear state
            set_recording_state(False)

        self.max_keep_frames = 25
        self.max_pred_frames = None

        if self.classify and self.do_tracking:
            self.init_classifiers(config.classify.models, preview_type)
        # call after model is setup
        super().__init__(thumbnail_dir)

        if self.headers.model == "IR":
            self.init_ir(thermal_config)
        else:
            self.init_thermal(thermal_config, detect_after)

        edge = self.tracking_config.edge_pixels
        self.crop_rectangle = Rectangle(
            edge, edge, headers.res_x - 2 * edge, headers.res_y - 2 * edge
        )
        global track_extractor
        track_extractor = self.track_extractor

        self.motion = thermal_config.motion
        self.min_frames = thermal_config.recorder.min_secs * headers.fps
        self.max_frames = thermal_config.recorder.max_secs * headers.fps

        self.meta_dir = os.path.join(thermal_config.recorder.output_dir)
        if not os.path.exists(self.meta_dir):
            os.makedirs(self.meta_dir)

    def init_ir_recorders(self, thermal_config):
        recording_stopping_callback = partial(
            on_recording_stopping,
            output_dir=self._output_dir,
            service=self.service,
            tracking_events=self.tracking_events,
        )

        from .irrecorder import IRRecorder

        if not thermal_config.recorder.disable_recordings:
            self.recorder = IRRecorder(
                thermal_config,
                self.headers,
                on_recording_stopping=recording_stopping_callback,
            )

        # dont want snaps getting postprocess
        postprocess = thermal_config.motion.postprocess
        thermal_config.motion.postprocess = False
        self.snapshot_recorder = IRRecorder(
            thermal_config,
            self.headers,
            on_recording_stopping=recording_stopping_callback,
            name="IR Snapshot",
        )
        thermal_config.motion.postprocess = postprocess

        if thermal_config.recorder.constant_recorder:
            self.constant_recorder = IRRecorder(
                thermal_config,
                self.headers,
                on_recording_stopping=recording_stopping_callback,
                name="IR Constant",
                constant_recorder=True,
            )

    def init_ir_tracking(self, thermal_config):
        from track.irtrackextractor import IRTrackExtractor

        logging.info("Running on IR")
        PiClassifier.SKIP_FRAMES = 3
        self.track_extractor = IRTrackExtractor(
            self.config.tracking,
            cache_to_disk=self.config.classify.cache_to_disk,
            keep_frames=False,
            verbose=self.config.verbose,
            calc_stats=False,
            scale=0.25,
            on_trapped=on_track_trapped,
            update_background=False,
            trap_size=thermal_config.device_setup.trap_size,
            from_pi=True,
        )
        self.tracking_config = self.config.tracking.get(IRTrackExtractor.TYPE)

        self.type = IRTrackExtractor.TYPE

    def init_ir(self, thermal_config):
        from .irmotiondetector import IRMotionDetector

        logging.info("Running on IR")
        if self.do_tracking:
            self.init_ir_tracking(thermal_config)
        self.init_ir_recorders(thermal_config)
        self.type = "IR"
        self.motion_detector = IRMotionDetector(
            thermal_config,
            self.headers,
        )

    def init_thermal(self, thermal_config, detect_after):
        from .cptvmotiondetector import CPTVMotionDetector

        logging.info("Running on Thermal")
        if self.do_tracking:
            self.init_tracking(thermal_config)
        self.init_recorders(thermal_config)
        self.type = "thermal"
        self.motion_detector = CPTVMotionDetector(
            thermal_config,
            self.tracking_config.motion.dynamic_thresh,
            self.headers,
            detect_after=detect_after,
        )

    def init_recorders(self, thermal_config):
        recording_stopping_callback = partial(
            on_recording_stopping,
            output_dir=self._output_dir,
            service=self.service,
            tracking_events=self.tracking_events,
        )

        if thermal_config.recorder.disable_recordings:
            from .dummyrecorder import DummyRecorder

            self.recorder = DummyRecorder(
                thermal_config,
                self.headers,
                on_recording_stopping=recording_stopping_callback,
            )
        else:
            from .cptvrecorder import CPTVRecorder

            self.recorder = CPTVRecorder(
                thermal_config,
                self.headers,
                on_recording_stopping=recording_stopping_callback,
            )

        if thermal_config.throttler.activate:
            from .throttledrecorder import ThrottledRecorder

            self.recorder = ThrottledRecorder(
                self.recorder,
                thermal_config,
                self.headers,
                on_recording_stopping=recording_stopping_callback,
            )

        # dont want snaps getting reprocessed
        postprocess = thermal_config.motion.postprocess
        thermal_config.motion.postprocess = False
        self.snapshot_recorder = CPTVRecorder(
            thermal_config,
            self.headers,
            on_recording_stopping=recording_stopping_callback,
            constant_recorder=False,
            name="CPTV Snapshot",
            file_suffix="-snap",
        )
        thermal_config.motion.postprocess = postprocess

        if thermal_config.recorder.constant_recorder:
            self.constant_recorder = CPTVRecorder(
                thermal_config,
                self.headers,
                on_recording_stopping=recording_stopping_callback,
                name="CPTV Constant",
                constant_recorder=True,
            )

    def init_tracking(self, thermal_config, detect_after=None):
        from track.cliptrackextractor import ClipTrackExtractor

        self.track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow,
            self.config.classify.cache_to_disk,
            keep_frames=False,
            calc_stats=False,
            update_background=False,
            from_pi=True,
        )
        self.tracking_config = self.config.tracking.get("thermal")

    def init_classifiers(self, models_config, preview_type):

        from ml_tools.interpreter import get_interpreter
        from classify.trackprediction import Predictions

        model = None
        fp_config = None
        for model_config in models_config:
            if model_config.type != "RandomForest":
                model = model_config
            else:
                fp_config = model_config

        if model is not None:
            self.classifier = get_interpreter(
                model, run_over_network=model.run_over_network
            )
            global classifier
            classifier = self.classifier
            self.frames_per_classify = (
                self.classifier.params.square_width
                * self.classifier.params.square_width
            )
            if self.frames_per_classify > 1:
                self.predict_from_last = self.frames_per_classify * 2

            self.max_keep_frames = (
                self.frames_per_classify * 2 if not preview_type else None
            )
            self.predictions[model.id] = Predictions(self.classifier.labels, model)
            self.num_labels = len(self.classifier.labels)
            logging.info("Labels are %s ", self.classifier.labels)
            global predictions
            predictions = self.predictions
            try:
                self.fp_index = self.classifier.labels.index("false-positive")
            except ValueError:
                self.fp_index = None
        if fp_config is not None:
            self.fp_model = get_interpreter(fp_config)
            global fp_model
            fp_model = self.fp_model
            self.predictions[self.fp_model.id] = Predictions(
                self.fp_model.labels, fp_config
            )

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
        self.clip.video_start_time = datetime.now() - timedelta(
            seconds=len(preview_frames) / self.headers.fps
        )

        self.clip.num_preview_frames = len(preview_frames)
        self.next_classify_frame = 0
        self.next_fp_classification_frame = 0
        self.clip.set_res(self.res_x, self.res_y)
        self.clip.set_frame_buffer(
            self.tracking_config.high_quality_optical_flow,
            self.config.classify.cache_to_disk,
            self.config.use_opt_flow,
            keep_frames=(
                self.max_keep_frames > 0 if self.max_keep_frames is not None else True
            ),
            max_frames=self.max_keep_frames,
        )

        self.clip.update_background(self.motion_detector.background.copy())
        self.clip._background_calculated()
        if self.do_tracking == False:
            return
        # no need to retrack all of preview
        background_frames = None
        track_frames = -1
        retrack_back = True
        if self.type == "IR":
            track_frames = 5
            retrack_back = True
            # background is calculated in motion, so already 5 frames ahead
        self.track_extractor.start_tracking(
            self.clip,
            preview_frames,
            track_frames=track_frames,
            background_alg=self.motion_detector._background,
            retrack_back=retrack_back,
            # background_frame=clip.background,
            # background_frames=background_frames,
        )

    def startup_classifier(self):
        self.classifier_initialised = True
        if self.classifier.run_over_network:
            if not utils.is_service_running("thermal-classifier"):
                success = utils.startup_network_classifier(True)
                if not success:
                    raise Exception("COuild not start network classifier")
            return
        # classifies an empty frame to force loading of the model into memory
        num_inputs, in_shape = self.classifier.shape()
        if num_inputs > 1:
            zero_input = []
            for shape in in_shape:
                zero_input.append(np.zeros((1, *shape[1:]), np.float32))
        else:
            zero_input = np.zeros((1, *in_shape[1:]), np.float32)
        self.classifier.predict(zero_input)

    def get_active_tracks(self):
        """
        Gets current clips active_tracks and returns the top NUM_CONCURRENT_TRACKS order by priority
        """
        active_tracks = self.clip.active_tracks
        active_tracks = [track for track in active_tracks if len(track) >= 8]
        return active_tracks

    def identify_last_frame(self):
        """
        Runs through track identifying segments, and then returns it's prediction of what kind of animal this is.
        One prediction will be made for every active_track of the last frame.
        :return: TrackPrediction object
        """
        if (
            self.next_fp_classification_frame >= self.clip.current_frame
            and self.next_classify_frame >= self.clip.current_frame
        ):
            return

        self.next_fp_classification_frame += PiClassifier.FP_MODEL_SKIP_FRAMES
        active_tracks = self.get_active_tracks()
        new_prediction = False
        if len(active_tracks) == 0:
            return False

        if self.fp_model is not None:
            fp_time = time.time()
            for track in active_tracks:
                start = time.time()
                if self.classifier is not None:
                    full_model = self.predictions[self.classifier.id].prediction_for(
                        track.get_id()
                    )
                    if full_model is not None and full_model.num_frames_classified > 0:
                        logging.debug(
                            "Skipping fp model for %s as has full model prediction",
                            track,
                        )
                        continue
                track_prediction = self.predictions[
                    self.fp_model.id
                ].get_or_create_prediction(
                    track,
                    keep_all=True,
                    smooth_preds=self.fp_model.params.smooth_predictions,
                )
                if (
                    track_prediction.last_frame_classified is not None
                    and self.clip.current_frame - track_prediction.last_frame_classified
                    < PiClassifier.FP_PREDICT_EVERY
                ):
                    logging.debug(
                        "Skipping %s #%s last %s",
                        track,
                        self.clip.current_frame,
                        track_prediction.last_frame_classified,
                    )
                    continue
                result = self.fp_model.predict_track(
                    clip,
                    track,
                    predict_from_last=45,
                    max_frames=PiClassifier.FP_PREDICT_EVERY // 5,
                    num_predictions=1,
                    last_frame_predicted=track_prediction.last_frame_classified,
                )
                if result is None:
                    track_prediction.last_frame_classified = self.clip.current_frame
                    continue
                frames, prediction, mass = result
                if prediction is None:
                    track_prediction.last_frame_classified = self.clip.current_frame
                    continue
                track_prediction.classified_frames(frames, prediction, mass)

                logging.debug(
                    "Track %s is predicted as %s took %s track frames %s",
                    track,
                    track_prediction.get_prediction(),
                    time.time() - start,
                    len(track),
                )
                new_prediction = True
            self.fp_identify_time += time.time() - fp_time
        if (
            self.classifier is not None
            and self.next_classify_frame <= self.clip.current_frame
        ):
            id_start = time.time()

            self.next_classify_frame += PiClassifier.SKIP_FRAMES
            animal_tracks = self.get_active_animal_tracks_for_predicting()
            # filter based of fp model
            for i, track in enumerate(animal_tracks):
                logging.debug("Running full classifier on %s", track)
                track_prediction = self.predictions[
                    self.classifier.id
                ].get_or_create_prediction(track, keep_all=True)
                start = time.time()
                pred_result = self.classifier.predict_recent_frames(
                    clip,
                    track,
                    predict_from_last=self.predict_from_last,
                    scale=self.track_extractor.scale,
                    frames_per_classify=self.frames_per_classify,
                    max_frames=self.max_pred_frames,
                    num_predictions=1,
                    calculate_filtered=True,
                    last_frame_predicted=track_prediction.last_frame_classified,
                )
                if pred_result is None:
                    track_prediction.last_frame_classified = self.clip.current_frame
                    continue
                prediction, frames, mass = pred_result

                if prediction is None:
                    track_prediction.last_frame_classified = self.clip.current_frame
                    continue
                track_prediction.classified_frames(frames, prediction, mass)

                logging.info(
                    "Track %s is predicted as %s took %s track frames %s",
                    track,
                    track_prediction.get_prediction(),
                    time.time() - start,
                    len(track),
                )
                new_prediction = True

            self.identify_time += time.time() - id_start

        for i, track in enumerate(active_tracks):
            if self.tracking_events:
                track_prediction, model_id = self.get_best_prediction(track.get_id())
                if track_prediction is None:
                    continue
                if track_prediction.predicted_tag() != "false-positive":
                    track_prediction.tracking = True
                    self.monitored_tracks[track.get_id()] = track
                elif track_prediction.tracking:
                    # tracking ended as is false-positive
                    track_prediction.tracking = False
                    track_prediction.normalize_score()
                    self.service.tracking(
                        self.clip._id,
                        track,
                        track_prediction.class_best_score,
                        track.bounds_history[-1],
                        False,
                        track_prediction.last_frame_classified,
                        self.predictions[model_id].labels,
                        model_id,
                    )
                    if track.get_id() in self.monitored_tracks:
                        del self.monitored_tracks[track.get_id()]

        if self.bluetooth_beacons:
            if new_prediction:
                active_predictions = []
                for track in self.clip.active_tracks:
                    track_prediction, model_id = self.get_best_prediction(
                        track.get_id()
                    )
                    if track_prediction:
                        active_predictions.append(track_prediction)
                beacon.classification(active_predictions)
        return new_prediction

    def get_active_animal_tracks_for_predicting(self):
        active_tracks = self.get_active_tracks()
        filtered = []
        least_fp_track = None
        for track in active_tracks:
            if self.fp_model is not None:
                pred, model_id = self.get_best_prediction(track.get_id())
                logging.debug(
                    "track %s -%s - %s",
                    track.get_id(),
                    pred.predicted_tag(),
                    pred.normalized_best_score(),
                )
                if pred is not None:
                    if pred.predicted_tag() == "false-positive":
                        confidence = pred.normalized_best_score()

                        if confidence >= 0.7:
                            if least_fp_track is None or least_fp_track[0] > confidence:
                                least_fp_track = (confidence, track)
                            logging.debug(
                                "Skipping track %s as is FP confidence %s",
                                track,
                                confidence,
                            )
                            continue

            pred = None
            if self.predictions is not None:
                pred = self.predictions[self.classifier.id].prediction_for(
                    track.get_id()
                )
            if pred is not None:
                if len(pred.predictions) < 2:
                    classify_every = PiClassifier.PREDICT_EVERY // 2
                else:
                    classify_every = PiClassifier.PREDICT_EVERY
                if (
                    pred.last_frame_classified is not None
                    and self.clip.current_frame - pred.last_frame_classified
                    < classify_every
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
        if len(active_tracks) == 0:
            if least_fp_track is None:
                return []
            logging.debug("Using least fp track %s", least_fp_track[1])
            return [least_fp_track[1]]

        # choose most likely animals first
        active_tracks = sorted(
            active_tracks,
            key=lambda track: self.animal_ranking(track),
            reverse=True,
        )
        if len(active_tracks) > PiClassifier.NUM_CONCURRENT_TRACKS:
            active_tracks = active_tracks[: PiClassifier.NUM_CONCURRENT_TRACKS]
        return active_tracks

    def animal_ranking(self, track):
        track_pred, model_id = self.get_best_prediction(track.get_id())

        if track_pred is None or track_pred.class_best_score is None:
            return 0
        fp_confidence = track_pred.class_best_score[track_pred.fp_index] / np.sum(
            track_pred.class_best_score
        )
        return 1 - fp_confidence

    def update_thumbnail(self, clip, tracks):
        best_contour = None
        from ml_tools.imageprocessing import resize_and_pad
        import cv2

        for track in tracks:
            confidence = None
            tag = None
            if predictions is not None:
                pred, model_id = self.get_best_prediction(track.get_id())
                if pred is not None and pred.max_score is not None:
                    confidence = round(100 * pred.max_score)
                    tag = pred.predicted_tag()
            regions = track.bounds_history
            if track.thumb_info is None:
                track.thumb_info = ThumbInfo(track.get_id())
            track.thumb_info.predicted_confidence = confidence
            track.thumb_info.predicted_tag = tag

            i = len(regions) - 1
            first_loop = True
            # go reverse and break when reach already checked frame
            while i >= 0:
                region = regions[i]
                if (
                    track.thumb_info.last_frame_check is not None
                    and track.thumb_info.last_frame_check >= region.frame_number
                ):
                    break
                frame = clip.frame_buffer.get_frame(region.frame_number)

                if frame is None:
                    break

                if first_loop:
                    track.thumb_info.last_frame_check = region.frame_number
                first_loop = False
                assert frame.frame_number == region.frame_number
                contour_image = frame.filtered if frame.mask is None else frame.mask
                contours, _ = cv2.findContours(
                    np.uint8(region.subimage(contour_image)),
                    cv2.RETR_EXTERNAL,
                    # cv2.CHAIN_APPROX_SIMPLE,
                    cv2.CHAIN_APPROX_TC89_L1,
                )

                contour_points = 0
                if len(contours) > 0:
                    contour_points = len(contours[0])
                if track.thumb_info.points < contour_points:
                    track.thumb_info.points = contour_points
                    track.thumb_info.region = region
                    track.thumb_info.thumb = None
                i -= 1
            if (
                best_contour is None
                or track.thumb_info.score() > best_contour.score()
                or (
                    track.thumb_info.predicted_tag != "false-positive"
                    and best_contour.predicted_tag == "false-positive"
                )
            ):
                best_contour = track.thumb_info

        for track in tracks:
            if track.thumb_info.region is None:
                continue
            thumb_frame = track.thumb_info.region.frame_number
            if (
                track.thumb_info.thumb is None
                or thumb_frame > track.thumb_info.thumb_frame
            ):
                frame = clip.frame_buffer.get_frame(thumb_frame)
                thumb_thermal = track.thumb_info.region.subimage(frame.thermal)
                if thumb_thermal.shape[0] > 32 or thumb_thermal.shape[1] > 32:
                    thumb_thermal = resize_and_pad(thumb_thermal, (32, 32), None, None)
                track.thumb_info.thumb = np.uint16(thumb_thermal)
                track.thumb_info.thumb_frame = thumb_frame
        return best_contour

    def get_and_update_thumbnail(self, clip_id=None, track_id=None):
        # gets best thumbnail for prevodied clip and track id
        # if no track id is provided it will choose the best over all thumbnail
        # if the thumbnail info for tracks is old it will calculate thumbnail data for missing frames
        # and update the tracks thumb info to be the up to date best thumbnail frame
        if clip_id is not None:
            if self.prev_clip is not None and self.prev_clip._id == clip_id:
                logging.info("Finding thumbnail in previous clip %s", clip_id)
                if track_id is not None:
                    best_track = next(
                        iter(
                            track
                            for track in self.prev_clip.tracks
                            if track.get_id() == track_id
                        ),
                        None,
                    )
                    if best_track is None:
                        logging.info("Couldn't find track %s", track_id)
                        return None
                else:
                    return self.prev_clip.thumb_info
                return best_track.thumb_info
            elif self.clip is None or self.clip._id != clip_id:
                logging.info("Cant find requested clip id %s", clip_id)
                return None
        elif self.clip is None:
            logging.info("Have no clip")
            return None
        if (
            self.clip.frame_buffer is None
            or self.clip.frame_buffer.frames is None
            or len(self.clip.frame_buffer.frames) == 0
        ):
            logging.info("Have no frames")
            return None

        # calculate on current clip
        with self.clip.frame_buffer.frame_lock:
            if track_id is not None:
                tracks = clip.tracks
                tracks = [track for track in tracks if track.get_id() == track_id]
                if len(tracks) == 0:
                    logging.info("Couldn't find track %s", track_id)
                    return None
            else:
                tracks = self.clip.active_tracks
            best_contour = self.update_thumbnail(self.clip, tracks)
            if track_id is not None:
                track = tracks[0]

            if best_contour is None:
                return None
            return best_contour

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
                    pred = {
                        self.predictions[self.classifier.id].model.id: self.predictions[
                            self.classifier.id
                        ]
                    }
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
        if self.recorder.recording and self.tracking_events:
            self.recording = False
            self.service.recording(False)
        self.recorder.force_stop()
        self.snapshot_recorder.force_stop()
        if self.constant_recorder is not None:
            self.constant_recorder.force_stop()

        self.end_clip()
        self.service.quit()

    def skip_frame(self):
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
        if (
            self.motion_detector.can_record()
            and not self.classifier_initialised
            and self.classify
        ):
            self.startup_classifier()
        self.motion_detector.process_frame(lepton_frame)
        self.process_time += time.time() - start
        if self.snapshot_recorder.recording:
            self.snapshot_recorder.process_frame(False, lepton_frame, received_at)
        if self.constant_recorder is not None and self.motion_detector.can_record():
            if self.constant_recorder.recording:
                self.constant_recorder.process_frame(True, lepton_frame, received_at)
            else:
                logging.info("Starting new constant recorder")
                self.recording = self.constant_recorder.start_recording(
                    self.motion_detector.background,
                    [],
                    self.motion_detector.temp_thresh,
                    time.time(),
                )
                if self.recording and not self.use_low_power_mode:
                    set_recording_state(True)
        if (
            not self.recorder.recording
            and self.motion_detector.movement_detected
            and not lepton_frame.ffc_imminent
            and not lepton_frame.ffc_status in [1, 2]
        ):
            s_r = time.time()
            preview_frames = self.motion_detector.preview_frames()
            bak = self.motion_detector.background
            self.recording = self.recorder.start_recording(
                self.motion_detector.background,
                preview_frames,
                self.motion_detector.temp_thresh,
                received_at,
            )
            self.rec_time += time.time() - s_r
            if self.recording:
                if self.tracking_events:
                    self.service.recording(True)
                if not self.use_low_power_mode:
                    set_recording_state(True)

                if self.bluetooth_beacons:
                    beacon.recording()
                t_start = time.time()
                self.new_clip(preview_frames)
                self.tracking_time += time.time() - t_start

        if self.recorder.recording:
            t_start = time.time()
            if self.do_tracking:
                self.track_extractor.process_frame(self.clip, lepton_frame)
                active_best = self.get_and_update_thumbnail()
                if self.clip.thumb_info is None or (
                    active_best is not None
                    and active_best.score() > self.clip.thumb_info.score()
                ):
                    self.clip.thumb_info = active_best
            self.tracking_time += time.time() - t_start
            s_r = time.time()

            self.recorder.process_frame(
                self.motion_detector.movement_detected, lepton_frame, received_at
            )
            self.rec_time += time.time() - s_r
            if self.classify:
                if self.motion_detector.calibrating:
                    # dont think we will get ffcs if we are recording
                    self.classified_consec = 0
                else:
                    identified = self.identify_last_frame()
                    if not identified:
                        self.classified_consec = 0
            elif len(self.monitored_tracks) == 0 and self.tracking_events:
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
                    self.monitored_tracks[active_tracks[0].get_id()] = active_tracks[0]

            if len(self.monitored_tracks) > 0:
                monitored_tracks = list(self.monitored_tracks.values())
                for monitored_track in monitored_tracks:
                    tracking = monitored_track in self.clip.active_tracks
                    score = 0
                    prediction = ""
                    all_scores = None
                    model_id = None
                    track_prediction = None
                    last_prediction = 0
                    if self.classify:
                        track_prediction, model_id = self.get_best_prediction(
                            monitored_track.get_id()
                        )
                        all_scores = track_prediction.get_normalized_score()
                        last_prediction = track_prediction.last_frame_classified
                    self.service.tracking(
                        self.clip._id,
                        monitored_track,
                        all_scores,
                        monitored_track.bounds_history[-1],
                        tracking,
                        last_prediction,
                        [] if model_id is None else self.predictions[model_id].labels,
                        model_id,
                    )

                    if not tracking:
                        del self.monitored_tracks[monitored_track.get_id()]
                        if self.classify:
                            track_prediction.tracking = False
        elif self.clip is not None:
            self.end_clip()

        if not self.recorder.recording and self.recording and self.tracking_events:
            self.recording = False
            self.service.recording(False)

        self.frame_num += 1
        self.total_time += time.time() - start
        if (
            self.motion_detector.can_record()
            and self.frame_num % PiClassifier.DEBUG_EVERY == 0
        ):
            if self.clip is not None:
                average = np.mean(self.fps_timer.get_frames())
                mem = process_mem()
                logging.debug(
                    "tracking %s %% process %s %%  identify %s %% FP Id %s %%  rec %s %% fps %s/sec process  system cpu %s process memory %s%% system memory %s behind by %s seconds",
                    round(100 * self.tracking_time / self.total_time, 3),
                    round(100 * self.process_time / self.total_time, 3),
                    round(100 * self.identify_time / self.total_time, 3),
                    round(100 * self.fp_identify_time / self.total_time, 3),
                    round(100 * self.rec_time / self.total_time, 3),
                    round(1 / average),
                    psutil.cpu_percent(),
                    mem,
                    psutil.virtual_memory()[2],
                    time.time() - received_at,
                )
            self.tracking_time = 0
            self.process_time = 0
            self.identify_time = 0
            self.fp_identify_time = 0
            self.total_time = 0
            self.rec_time = 0
        self.fps_timer.add(time.time() - start)

    def create_mp4(self):
        from ml_tools.previewer import Previewer

        previewer = Previewer(self.config, self.preview_type)
        previewer.export_clip_preview(
            os.path.join(self.output_dir, self.clip.get_id() + ".mp4"),
            self.clip,
            self.predictions if self.classify else None,
        )

    def get_best_prediction(self, track_id):
        if self.classifier is not None:
            main_classifier = self.predictions[self.classifier.id].prediction_for(
                track_id
            )
            if (
                main_classifier is not None
                and main_classifier.num_frames_classified > 0
            ):
                return main_classifier, self.classifier.id
        if self.fp_model is not None:
            return (
                self.predictions[self.fp_model.id].prediction_for(track_id),
                self.fp_model.id,
            )
        return None, None

    def end_clip(self):
        if self.clip:
            global clip
            if self.preview_type:
                self.create_mp4()
            logging.debug(
                "Ending clip with %s tracks post filtering", len(self.clip.tracks)
            )
            if self.classify:
                for pred in self.predictions.values():
                    logging.info("Pred for model %s", pred.model)
                    for t_id, prediction in pred.prediction_per_track.items():
                        if prediction.max_score:
                            logging.info(
                                "Clip {} {} {}".format(
                                    self.clip.get_id(),
                                    t_id,
                                    prediction.description(),
                                )
                            )
                    pred.clear_predictions()
            # set so we can get thumbnail info
            self.prev_clip = clip
            self.prev_clip.frame_buffer = None
            self.clip = None
            self.monitored_tracks = {}
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
    trigger_trap(tag)


def on_recording_stopping(
    filename, output_dir=None, service=None, tracking_events=False
):
    from ml_tools.tools import CustomJSONEncoder

    global use_low_power_mode
    if not use_low_power_mode:
        set_recording_state(False)
    if "-snap" in filename.stem:
        return
    global clip, track_extractor, predictions
    if clip and track_extractor:
        # save thumbs
        filtered_tracks = track_extractor.apply_track_filtering(clip)
        if tracking_events:
            for track in filtered_tracks:
                service.track_filtered(clip._id, track.get_id())
        for track in clip.tracks:
            if track.thumb_info is not None:
                np.save(
                    f"{str(output_dir)}/thumbnails/{clip.get_id()}-{track.get_id()}.npy",
                    track.thumb_info.thumb,
                )
        if predictions is not None:
            valid_preds = {}

            # remove track prediction
            for track in clip.tracks:
                for model_pred in predictions.values():
                    pred = model_pred.prediction_for(track.get_id())
                    if pred is not None:
                        pred.normalize_score()
            #             valid_preds[model_pred.model.id] = pred
            # for model_id in predictions.keys():
            #     if model_id in valid_preds:
            #         predictions[model_id].prediction_per_track = valid_preds[model_id].prediction_per_track

        # filter criteria has been scaled so resize after
        # if track_extractor.scale is not None:
        #     for track in clip.tracks:
        #         for r in track.bounds_history:
        #             # bring back to orignal size
        #             r.rescale(1 / track_extractor.scale)

        meta_name = filename.with_suffix(".txt")
        logging.debug("saving meta to %s", meta_name)
        predictions_per_model = None

        if predictions is not None:
            predictions_per_model = predictions
        meta_data = clip.get_metadata(predictions_per_model)
        meta_data["algorithm"] = {}
        meta_data["algorithm"]["tracker_version"] = f"PI-{track_extractor.VERSION}"
        meta_data["metadata_source"] = "PI"
        if clip.thumb_info is not None:
            meta_data["thumbnail"] = clip.thumb_info.to_metadata()
        if predictions is not None:
            models = []
            model_name = ""
            joiner = ""
            for model_preds in predictions.values():
                models.append(model_preds.model.as_dict())
                model_name = f"{model_name}{joiner}{model_preds.model.name}"
                joiner = " and "
            meta_data["algorithm"]["model_name"] = model_name
            meta_data["models"] = models

        with open(meta_name, "w") as f:
            json.dump(meta_data, f, indent=4, cls=CustomJSONEncoder)


def process_mem():
    # return the memory usage in percentage like top
    process = psutil.Process(os.getppid())
    return process.memory_percent()
