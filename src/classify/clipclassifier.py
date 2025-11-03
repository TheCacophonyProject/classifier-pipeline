import gc
import json
import logging
import os.path
import time
import math
import numpy as np

import cv2
from classify.trackprediction import Predictions
from track.clip import Clip
from track.cliptrackextractor import ClipTrackExtractor, is_affected_by_ffc
from ml_tools import tools
from track.irtrackextractor import IRTrackExtractor
from ml_tools.previewer import Previewer
from ml_tools.interpreter import get_interpreter
from track.trackextractor import extract_file
from classify.thumbnail import get_thumbnail_info, best_trackless_thumb
from pathlib import Path


class ClipClassifier:
    """Classifies tracks within CPTV files."""

    # skips every nth frame.  Speeds things up a little, but reduces prediction quality.
    FRAME_SKIP = 1

    def __init__(
        self, config, model=None, keep_original_predictions=False, tracking_events=False
    ):
        """Create an instance of a clip classifier"""
        self.keep_original_predictions = keep_original_predictions
        self.config = config
        # super(ClipClassifier, self).__init__(config, tracking_config)
        self.model = model
        if self.keep_original_predictions:
            self.model.id = f"post-{self.model.id}"
            self.model.name = f"post-{self.model.name}"

        # prediction record for each track

        self.previewer = Previewer.create_if_required(config, config.classify.preview)

        self.models = {}
        self.tracking_events = tracking_events
        self._is_recording = False

    def set_is_recording(self, is_recording):
        self._is_recording = is_recording

    def load_models(self):
        for model in self.config.classify.models:
            logging.info("Loading %s", model)
            classifier = self.get_classifier(model)

    def get_classifier(self, model):
        """
        Returns a classifier object, which is created on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        if model.id in self.models:
            return self.models[model.id]
        load_start = time.time()
        logging.info("classifier loading %s", model.model_file)
        classifier = get_interpreter(model, model.run_over_network)
        logging.info("classifier loaded (%s)", time.time() - load_start)
        self.models[model.id] = classifier
        return classifier

    def get_meta_data(self, filename):
        """Reads meta-data for a given cptv file."""
        source_meta_filename = os.path.splitext(filename)[0] + ".txt"
        if os.path.exists(source_meta_filename):
            meta_data = tools.load_clip_metadata(source_meta_filename)

            tags = set()
            for record in meta_data["Tags"]:
                # skip automatic tags
                if record.get("automatic", False):
                    continue
                else:
                    tags.add(record["animal"])

            tags = list(tags)

            if len(tags) == 0:
                tag = "no tag"
            elif len(tags) == 1:
                tag = tags[0] if tags[0] else "none"
            else:
                tag = "multi"
            meta_data["primary_tag"] = tag
            return meta_data
        else:
            return None

    def process(
        self,
        source,
        cache=None,
        reuse_frames=None,
        track=False,
        calculate_thumbnails=False,
    ):
        # IF passed a dir extract all cptv files, if a cptv just extract this cptv file
        if not os.path.exists(source):
            logging.error("Could not find file or directory %s", source)
            return
        if os.path.isfile(source):
            self.process_file(
                source,
                cache=cache,
                reuse_frames=reuse_frames,
                track=track,
                calculate_thumbnails=calculate_thumbnails,
            )
            return
        for folder_path, _, files in os.walk(source):
            for name in files:
                if os.path.splitext(name)[1] in [".mp4", ".cptv", ".avi"]:
                    full_path = os.path.join(folder_path, name)
                    self.process_file(
                        full_path,
                        cache=cache,
                        reuse_frames=reuse_frames,
                        track=track,
                        calculate_thumbnails=calculate_thumbnails,
                    )

    def process_file(
        self,
        filename,
        cache=None,
        reuse_frames=None,
        track=False,
        calculate_thumbnails=False,
    ):
        """
        Process a file extracting tracks and identifying them.
        :param filename: filename to process
        :param enable_preview: if true an MPEG preview file is created.
        """
        _, ext = os.path.splitext(filename)
        cache_to_disk = (
            cache if cache is not None else self.config.classify.cache_to_disk
        )

        if track:
            logging.info("Doing tracking")
            clip, track_extractor = extract_file(
                filename, self.config, cache_to_disk, to_stdout=False
            )
        elif ext == ".cptv":
            track_extractor = ClipTrackExtractor(
                self.config.tracking,
                self.config.use_opt_flow,
                cache_to_disk,
                do_tracking=track,
                calculate_filtered=True,
                verbose=self.config.verbose,
                calculate_thumbnail_info=calculate_thumbnails,
            )
            logging.info("Using clip extractor")

        elif ext in [".avi", ".mp4"]:
            track_extractor = IRTrackExtractor(self.config.tracking, cache_to_disk)
            logging.info("Using ir extractor")
        else:
            logging.error("Unknown extention %s", ext)
            return False

        base_filename = os.path.splitext(os.path.basename(filename))[0]
        meta_file = os.path.join(os.path.dirname(filename), base_filename + ".txt")
        if not os.path.exists(filename):
            logging.error("File %s not found.", filename)
            return False
        if not os.path.exists(meta_file):
            logging.error("File %s not found.", meta_file)
            return False
        meta_data = tools.load_clip_metadata(meta_file)

        logging.info("Processing file '{}'".format(filename))

        if not track:
            clip = Clip(track_extractor.config, filename)
            clip.load_metadata(
                meta_data,
                self.config.build.tag_precedence,
            )
            track_extractor.parse_clip(clip)

        predictions_per_model = {}
        if self.model:
            prediction = self.classify_clip(
                clip,
                self.model,
                meta_data,
                reuse_frames=reuse_frames,
            )
            predictions_per_model[self.model.id] = prediction
        else:
            for model in self.config.classify.models:
                prediction = self.classify_clip(
                    clip,
                    model,
                    meta_data,
                    reuse_frames=reuse_frames,
                )
                predictions_per_model[model.id] = prediction
        destination_folder = os.path.dirname(filename)
        dirname = destination_folder

        if self.previewer:
            mpeg_filename = os.path.join(dirname, base_filename + "-classify.mp4")

            logging.info("Exporting preview to '{}'".format(mpeg_filename))

            self.previewer.export_clip_preview(
                mpeg_filename, clip, predictions_per_model
            )
        models = [self.model] if self.model else self.config.classify.models
        meta_data = self.save_metadata(
            meta_data,
            meta_file,
            clip,
            predictions_per_model,
            models,
            calculate_thumbnails=calculate_thumbnails,
        )
        if cache_to_disk:
            clip.frame_buffer.remove_cache()
        return meta_data

    def classify_clip(self, clip, model, meta_data, reuse_frames=None):
        start = time.time()
        classifier = self.get_classifier(model)
        predictions = Predictions(classifier.labels, model)
        predictions.model_load_time = time.time() - start

        for i, track in enumerate(clip.tracks):
            segment_frames = None
            if reuse_frames:
                tracks = meta_data.get("tracks")
                meta_track = next(
                    (x for x in tracks if x["id"] == track.get_id()), None
                )
                if meta_track is not None:
                    prediction_tag = next(
                        (
                            x
                            for x in meta_track["tags"]
                            if x.get("data", {}).get("name") == model.name
                        ),
                        None,
                    )
                    if prediction_tag is not None:
                        if "prediction_frames" in prediction_tag["data"]:
                            logging.info("Reusing previous prediction frames %s", model)
                            segment_frames = prediction_tag["data"]["prediction_frames"]
                            segment_frames = np.uint16(segment_frames)

            prediction = classifier.classify_track(
                clip, track, segment_frames=segment_frames, min_segments=1
            )
            if prediction is not None:
                predictions.prediction_per_track[track.get_id()] = prediction
                description = prediction.description()
                logging.info(
                    "{} - [{}/{}] prediction: {}".format(
                        track.get_id(), i + 1, len(clip.tracks), description
                    )
                )
        if self.config.verbose:
            ms_per_frame = (
                (time.time() - start) * 1000 / max(1, len(clip.frame_buffer.frames))
            )
            logging.info("Took {:.1f}ms per frame".format(ms_per_frame))
        if classifier.TYPE == "Keras":
            tools.clear_session()
        del classifier
        gc.collect()

        return predictions

    def save_metadata(
        self,
        meta_data,
        meta_filename,
        clip,
        predictions_per_model,
        models,
        calculate_thumbnails=False,
    ):
        tracks = meta_data.get("tracks")
        for track in clip.tracks:
            meta_track = next((x for x in tracks if x["id"] == track.get_id()), None)
            if meta_track is None:
                logging.error(
                    "Got prediction for track which doesn't exist in metadata"
                )
                continue
            prediction_info = []
            for model_id, predictions in predictions_per_model.items():
                prediction = predictions.prediction_for(track.get_id())
                if prediction is None:
                    continue
                prediction_meta = prediction.get_metadata()
                prediction_meta["model_id"] = model_id
                if self.keep_original_predictions:
                    prediction_meta["reprocessed"] = True
                prediction_info.append(prediction_meta)
            if self.keep_original_predictions:
                existing_predictions = meta_track.get("predictions", [])
                if existing_predictions is None:
                    existing_predictions = []
                prediction_info.extend(existing_predictions)
            meta_track["predictions"] = prediction_info

            if calculate_thumbnails:
                best_thumb, best_score = get_thumbnail_info(clip, track)
                if best_thumb is None:
                    meta_track["thumbnail"] = None
                else:
                    thumbnail_info = {
                        "region": best_thumb.region,
                        "contours": best_thumb.contours,
                        "median_diff": best_thumb.median_diff,
                        "score": round(best_score),
                    }
                    meta_track["thumbnail"] = thumbnail_info
        if calculate_thumbnails and len(clip.tracks) == 0:
            # if no tracks choose a clip thumb
            region = best_trackless_thumb(clip)
            meta_data["thumbnail_region"] = region

        model_dictionaries = {}

        for existing_model in meta_data.get("models", []):
            model_dictionaries[existing_model["id"]] = existing_model

        for model in models:
            if model.id in model_dictionaries:
                model_dic = model_dictionaries[model.id]
            else:
                model_dic = model.as_dict()
            model_predictions = predictions_per_model[model.id]
            model_dic["classify_time"] = float(
                round(
                    model_predictions.classify_time + model_predictions.model_load_time,
                    1,
                )
            )
            model_dictionaries[model.id] = model_dic
        meta_data["models"] = list(model_dictionaries.values())
        if self.config.classify.meta_to_stdout:
            logging.info("Printing json meta data")

            print(json.dumps(meta_data, cls=tools.CustomJSONEncoder))
        else:
            logging.info("saving meta data %s", meta_filename)
            with open(meta_filename, "w") as f:
                json.dump(meta_data, f, indent=4, cls=tools.CustomJSONEncoder)
        return meta_data

    def post_process_file(self, filename, service):
        from cptv_rs_python_bindings import CptvReader
        from piclassifier.motiondetector import RunningMean, SlidingWindow
        from piclassifier.cptvmotiondetector import CPTVMotionDetector
        from ml_tools.frame import Frame
        from ml_tools.preprocess import preprocess_frame, preprocess_movement
        from datetime import datetime
        import dbus

        filename = Path(filename)
        meta_file = filename.with_suffix(".txt")
        if not filename.exists():
            logging.error("File %s not found.", filename)
            return False
        if not meta_file.exists():
            logging.error("File %s not found.", meta_file)
            return False
        meta_data = tools.load_clip_metadata(meta_file)
        filename = Path(filename)
        meta_file = filename.with_suffix(".txt")
        if not filename.exists():
            logging.error("File %s not found.", filename)
            return False
        if not meta_file.exists():
            logging.error("File %s not found.", meta_file)
            return False
        meta_data = tools.load_clip_metadata(meta_file)

        rec_end = datetime.fromisoformat(meta_data["end_time"])

        # get segments here, or frames
        # only extra data for segments
        track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow,
            calculate_filtered=True,
            verbose=self.config.verbose,
        )

        clip = Clip(track_extractor.config, filename)
        clip.load_metadata(
            meta_data,
            self.config.build.tag_precedence,
        )
        track_extractor.init_clip(clip)

        logging.info("Just running on first model")
        start = time.time()
        model = self.config.classify.models[0]
        classifier = self.get_classifier(model)
        predictions = Predictions(classifier.labels, model)
        predictions.model_load_time = time.time() - start

        track_samples = {}
        track_data = {}

        for track in clip.tracks:
            pred_frames = classifier.frames_for_prediction(clip, track)

            track_data[track.get_id()] = {
                "pred_frames": pred_frames,
                "limits": None,
                "frames": {},
                "track": track,
            }

            for seg in pred_frames:

                for r in seg.regions:
                    frame_data = track_samples.setdefault(r.frame_number, {})
                    frame_data[track.get_id()] = r
                    # frame_samples.append(r)
        reader = CptvReader(str(clip.source_file))
        current_frame_num = 0
        running_mean = None
        thermal_window = SlidingWindow(CPTVMotionDetector.MEAN_FRAMES, "O")

        if classifier.params.thermal_diff_norm:
            logging.error("Thermal min diff is not implemented so will not be used")

        while True:
            frame = reader.next_frame()

            if frame is None:
                break
            if frame.background_frame:
                continue

            if current_frame_num in track_samples:
                thermal_median = np.median(frame.pix)
                for track_id, region in track_samples[current_frame_num].items():
                    # region = track_samples[current_frame_num]
                    thermal = region.subimage(frame.pix).astype(np.float32)
                    background = region.subimage(
                        track_extractor.background_alg.background
                    )
                    filtered = thermal - background
                    thermal -= thermal_median
                    f = Frame(thermal, filtered, current_frame_num, region=region)
                    track_data[track_id]["frames"][region.frame_number] = f
                    if classifier.params.diff_norm:
                        f_min = np.min(filtered)
                        f_max = np.max(filtered)
                        existing_limits = track_data[track_id]["limits"]

                        if existing_limits is None:
                            track_data[track_id]["limits"] = [f_min, f_max]
                        else:
                            if f_min < existing_limits[0]:
                                existing_limits[0] = f_min
                            if f_max > existing_limits[1]:
                                existing_limits[1] = f_max
                            track_data[track_id]["limits"] = existing_limits
            # track_extractor.process_frame(clip, frame)
            is_ffc = is_affected_by_ffc(frame)
            oldest_thermal = thermal_window.oldest
            thermal_window.add(frame, is_ffc)

            if running_mean is None:
                running_mean = RunningMean([frame.pix], CPTVMotionDetector.MEAN_FRAMES)
            else:
                running_mean.add(frame.pix, oldest_thermal.pix)
            if not is_ffc:
                track_extractor.background_alg.process_frame(running_mean.mean())
            current_frame_num += 1
        i = 0
        for track_id, data in track_data.items():
            i += 1
            pred_frames = data["pred_frames"]
            pred_frame_numbers = []
            preprocessed = []
            masses = []
            for segment in pred_frames:
                segment_frames = []
                for frame_i in segment.frame_indices:
                    f = data["frames"][frame_i]
                    if not f.preprocessed:
                        f = preprocess_frame(
                            f,
                            (
                                classifier.params.frame_size,
                                classifier.params.frame_size,
                            ),
                            region,
                            clip.background,
                            clip.crop_rectangle,
                            calculate_filtered=False,
                            filtered_norm_limits=data["limits"],
                            cropped=True,
                            sub_median=False,
                        )
                        data["frames"][frame_i] = f
                    # probably no need to copy
                    segment_frames.append(f)
                frames = preprocess_movement(
                    segment_frames,
                    classifier.params.square_width,
                    classifier.params.frame_size,
                    classifier.params.channels,
                    classifier.preprocess_fn,
                    sample=f"{clip.get_id()}-{track_id}",
                )
                preprocessed.append(frames)
                masses.append(segment.mass)
                pred_frame_numbers.append(segment.frame_indices)
            if len(preprocessed) == 0:
                logging.info("No prediction made for track %s", track_id)
                continue
                # dont think this should happen
            preprocessed = np.array(preprocessed)

            # what to do if recording
            if self._is_recording:
                while self._is_recording:
                    logging.info("Waiting for current recording to finish")
                    time.sleep(10)
                # sleep here until not recording

            preds = []
            chunk_size = 5
            chunks = int(math.ceil(len(preprocessed) / chunk_size))
            # if is a very long track should break this up into samller chunks
            for chunk in range(chunks):
                preprocessed_chunk = preprocessed[
                    chunk * chunk_size : chunk * chunk_size + chunk_size
                ]
                logging.info(
                    "Predicting chunk %s (%s #) of %s %s:%s total preprocessed %s",
                    chunk,
                    len(preprocessed_chunk),
                    chunks,
                    chunk * chunk_size,
                    chunk * chunk_size + chunk_size,
                    len(preprocessed),
                )
                pred = classifier.predict(preprocessed_chunk)
                preds.extend(pred)
            track_prediction = classifier.track_prediction_from_raw(
                track_id, pred_frame_numbers, preds, masses
            )
            predictions.prediction_per_track[track_id] = track_prediction

            description = track_prediction.description()
            logging.info(
                "{} - [{}/{}] prediction: {}".format(
                    track_id, i, len(clip.tracks), description
                )
            )

            if (
                self.tracking_events
                and len(track_prediction.predictions) > 0
                # incase wasn't false positive during active tracking would want it reported now as FP
                # and track_prediction.predicted_tag() != "false-positive"
            ):
                dbus_preds = track_prediction.class_best_score.copy()
                dbus_preds = np.uint8(np.round(dbus_preds * 100))
                best = np.argmax(predictions)
                dbus_preds = dbus_preds.tolist()

                service.TrackReprocessed(
                    meta_data.get("id", 0),
                    track_id,
                    dbus_preds,
                    track_prediction.predicted_tag(),
                    int(round(100 * track_prediction.max_score)),
                    region.to_ltrb(),
                    region.frame_number,
                    region.mass,
                    region.blank,
                    True,
                    data["track"].bounds_history[-1].frame_number,
                    model.id,
                    rec_end.timestamp(),
                )

        models = [model]
        predictions_per_model = {model.id: predictions}

        meta_data = self.save_metadata(
            meta_data,
            meta_file,
            clip,
            predictions_per_model,
            models,
            calculate_thumbnails=False,
        )
