from datetime import datetime, timedelta
import json
import logging
import os
import time
import psutil
import numpy as np
import logging
from track.clip import Clip

from .motiondetector import SlidingWindow
from .processor import Processor

from ml_tools.logs import init_logging
from ml_tools.rectangle import Rectangle
from . import beacon

from piclassifier.trapcontroller import trigger_trap
from piclassifier.attiny import set_recording_state

SNAPSHOT_SIGNAL = "snap"
STOP_SIGNAL = "stop"
SKIP_SIGNAL = "skip"
track_extractor = None
clip = None


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
    ):
        self.constant_recorder = None
        self._output_dir = thermal_config.recorder.output_dir
        self.headers = headers
        self.classifier = None
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
        if thermal_config.recorder.disable_recordings:
            from .dummyrecorder import DummyRecorder

            self.recorder = DummyRecorder(
                thermal_config, headers, on_recording_stopping=on_recording_stopping
            )

        if headers.model == "IR":
            from track.irtrackextractor import IRTrackExtractor
            from .irmotiondetector import IRMotionDetector

            logging.info("Running on IR")
            PiClassifier.SKIP_FRAMES = 3

            if self.do_tracking:

                self.track_extractor = IRTrackExtractor(
                    self.config.tracking,
                    cache_to_disk=self.config.classify.cache_to_disk,
                    keep_frames=False,
                    verbose=config.verbose,
                    calc_stats=False,
                    scale=0.25,
                    on_trapped=on_track_trapped,
                    update_background=False,
                    trap_size=thermal_config.device_setup.trap_size,
                    from_pi=True,
                )
            self.tracking_config = self.config.tracking.get(IRTrackExtractor.TYPE)

            self.type = IRTrackExtractor.TYPE
            from .irrecorder import IRRecorder

            if not thermal_config.recorder.disable_recordings:
                self.recorder = IRRecorder(
                    thermal_config, headers, on_recording_stopping=on_recording_stopping
                )
            self.snapshot_recorder = IRRecorder(
                thermal_config,
                headers,
                on_recording_stopping=on_recording_stopping,
                name="IR Snapshot",
            )
            self.motion_detector = IRMotionDetector(
                thermal_config,
                headers,
            )
            if thermal_config.recorder.constant_recorder:
                self.constant_recorder = IRRecorder(
                    thermal_config,
                    headers,
                    on_recording_stopping=on_recording_stopping,
                    name="IR Constant",
                    constant_recorder=True,
                )
        else:
            logging.info("Running on Thermal")
            from .cptvmotiondetector import CPTVMotionDetector

            if self.do_tracking:
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

            self.type = "thermal"
            if not thermal_config.recorder.disable_recordings:
                from .cptvrecorder import CPTVRecorder

                self.recorder = CPTVRecorder(
                    thermal_config, headers, on_recording_stopping=on_recording_stopping
                )
            self.snapshot_recorder = CPTVRecorder(
                thermal_config,
                headers,
                on_recording_stopping=on_recording_stopping,
                constant_recorder=False,
                name="CPTV Snapshot",
                file_suffix="-snap",
            )
            self.motion_detector = CPTVMotionDetector(
                thermal_config,
                self.tracking_config.motion.dynamic_thresh,
                headers,
                detect_after=detect_after,
            )
            if thermal_config.recorder.constant_recorder:
                self.constant_recorder = CPTVRecorder(
                    thermal_config,
                    headers,
                    on_recording_stopping=on_recording_stopping,
                    name="CPTV Constant",
                    constant_recorder=True,
                )
        edge = self.tracking_config.edge_pixels
        self.crop_rectangle = Rectangle(
            edge, edge, headers.res_x - 2 * edge, headers.res_y - 2 * edge
        )
        global track_extractor
        track_extractor = self.track_extractor
        self.motion = thermal_config.motion
        self.min_frames = thermal_config.recorder.min_secs * headers.fps
        self.max_frames = thermal_config.recorder.max_secs * headers.fps
        if thermal_config.recorder.disable_recordings:
            from .dummyrecorder import DummyRecorder

            self.recorder = DummyRecorder(
                thermal_config, headers, on_recording_stopping=on_recording_stopping
            )
        if thermal_config.throttler.activate:
            from .throttledrecorder import ThrottledRecorder

            self.recorder = ThrottledRecorder(
                self.recorder,
                thermal_config,
                headers,
                on_recording_stopping=on_recording_stopping,
            )
        self.meta_dir = os.path.join(thermal_config.recorder.output_dir)
        if not os.path.exists(self.meta_dir):
            os.makedirs(self.meta_dir)

        if self.classify and self.do_tracking:
            from ml_tools.interpreter import get_interpreter
            from classify.trackprediction import Predictions

            model = config.classify.models[0]
            self.classifier = get_interpreter(model)

            if self.classifier.TYPE == "RandomForest":
                self.predict_from_last = 5 * headers.fps
                self.frames_per_classify = self.predict_from_last
                PiClassifier.SKIP_FRAMES = 30
                # probably could be even more

            else:
                # self.preprocess_fn = self.get_preprocess_fn()

                self.predict_from_last = 1
                self.frames_per_classify = (
                    self.classifier.params.square_width
                    * self.classifier.params.square_width
                )
                if self.frames_per_classify > 1:
                    self.predict_from_last = self.frames_per_classify * 2

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
        super().__init__()

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
                predict_from_last=self.predict_from_last,
                scale=self.track_extractor.scale,
                frames_per_classify=self.frames_per_classify,
                num_predictions=1,
                calculate_filtered=True,
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
                        track_prediction.class_best_score,
                        track.bounds_history[-1],
                        True,
                        track_prediction.last_frame_classified,
                    )
                elif track_prediction.tracking:
                    track_prediction.tracking = False
                    self.tracking = None
                    track_prediction.normalize_score()
                    self.service.tracking(
                        track_prediction.class_best_score,
                        track.bounds_history[-1],
                        False,
                        track_prediction.last_frame_classified,
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
            self.tracking_time += time.time() - t_start
            s_r = time.time()
            if self.tracking is not None:
                tracking = self.tracking in self.clip.active_tracks
                score = 0
                prediction = ""
                all_scores = None
                last_prediction = 0
                if self.classify:
                    track_prediction = self.predictions.prediction_for(
                        self.tracking.get_id()
                    )
                    all_scores = track_prediction.class_best_score
                    last_prediction = track_prediction.last_frame_classified
                self.service.tracking(
                    all_scores,
                    self.tracking.bounds_history[-1],
                    tracking,
                    last_prediction,
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

        if not self.recorder.recording and self.recording and self.tracking_events:
            self.recording = False
            self.service.recording(False)

        self.skip_classifying -= 1
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
                    "tracking %s %% process %s %%  identify %s %% rec %s %% fps %s/sec process  system cpu %s process memory %s%% system memory %s behind by %s seconds",
                    round(100 * self.tracking_time / self.total_time, 3),
                    round(100 * self.process_time / self.total_time, 3),
                    round(100 * self.identify_time / self.total_time, 3),
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
    trigger_trap(tag)


def on_recording_stopping(filename):
    from ml_tools.tools import CustomJSONEncoder

    global use_low_power_mode
    if not use_low_power_mode:
        set_recording_state(False)

    if "-snap" in filename.stem:
        return
    global clip, track_extractor, predictions

    if clip and track_extractor:
        track_extractor.apply_track_filtering(clip)
        if predictions is not None:
            valid_preds = {}
            for track in clip.tracks:
                if track.get_id() in predictions.prediction_per_track:
                    valid_preds[track.get_id()] = predictions.prediction_per_track[
                        track.get_id()
                    ]
                    valid_preds[track.get_id()].normalize_score()
            predictions.prediction_per_track = valid_preds

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
        meta_data["algorithm"]["tracker_version"] = f"PI-{track_extractor.VERSION}"
        meta_data["metadata_source"] = "PI"
        if predictions is not None:
            meta_data["models"] = [predictions.model.as_dict()]
            meta_data["algorithm"]["model_name"] = predictions.model.name

        with open(meta_name, "w") as f:
            json.dump(meta_data, f, indent=4, cls=CustomJSONEncoder)


def process_mem():
    # return the memory usage in percentage like top
    process = psutil.Process(os.getppid())
    return process.memory_percent()
