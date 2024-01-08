import gc
import json
import logging
import os.path
import time

import numpy as np

import cv2
from classify.trackprediction import Predictions
from track.clip import Clip
from track.cliptrackextractor import ClipTrackExtractor, is_affected_by_ffc
from ml_tools import tools
from ml_tools.kerasmodel import KerasModel
from track.irtrackextractor import IRTrackExtractor
from ml_tools.previewer import Previewer
from track.track import Track

from cptv import CPTVReader
from datetime import datetime
from ml_tools.interpreter import get_interpreter


class ClipClassifier:
    """Classifies tracks within CPTV files."""

    # skips every nth frame.  Speeds things up a little, but reduces prediction quality.
    FRAME_SKIP = 1

    def __init__(self, config, model=None):
        """Create an instance of a clip classifier"""

        self.config = config
        # super(ClipClassifier, self).__init__(config, tracking_config)
        self.model = model
        # prediction record for each track

        self.previewer = Previewer.create_if_required(config, config.classify.preview)

        self.models = {}

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
        classifier = get_interpreter(model)
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

    def process(self, source, cache=None, reuse_frames=None):
        # IF passed a dir extract all cptv files, if a cptv just extract this cptv file
        if os.path.isfile(source):
            self.process_file(source, cache=cache, reuse_frames=reuse_frames)
            return
        for folder_path, _, files in os.walk(source):
            for name in files:
                if os.path.splitext(name)[1] in [".mp4", ".cptv", ".avi"]:
                    full_path = os.path.join(folder_path, name)
                    self.process_file(full_path, cache=cache, reuse_frames=reuse_frames)

    def process_file(self, filename, cache=None, reuse_frames=None):
        """
        Process a file extracting tracks and identifying them.
        :param filename: filename to process
        :param enable_preview: if true an MPEG preview file is created.
        """
        _, ext = os.path.splitext(filename)
        cache_to_disk = (
            cache if cache is not None else self.config.classify.cache_to_disk
        )
        if ext == ".cptv":
            track_extractor = ClipTrackExtractor(
                self.config.tracking, self.config.use_opt_flow, cache_to_disk
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

        start = time.time()
        clip = Clip(track_extractor.config, filename)
        clip.load_metadata(
            meta_data,
            self.config.load.tag_precedence,
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
                mpeg_filename, clip, list(predictions_per_model.values())[0]
            )
        logging.info("saving meta data %s", meta_file)
        models = [self.model] if self.model else self.config.classify.models
        meta_data = self.save_metadata(
            meta_data,
            meta_file,
            clip,
            predictions_per_model,
            models,
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
                clip, track, segment_frames=segment_frames
            )
            if prediction is not None:
                predictions.prediction_per_track[track.get_id()] = prediction
                description = prediction.description()
                logging.info(
                    " - [{}/{}] prediction: {}".format(
                        i + 1, len(clip.tracks), description
                    )
                )
        if self.config.verbose:
            ms_per_frame = (
                (time.time() - start) * 1000 / max(1, len(clip.frame_buffer.frames))
            )
            logging.info("Took {:.1f}ms per frame".format(ms_per_frame))
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
                prediction_info.append(prediction_meta)
            meta_track["predictions"] = prediction_info

        model_dictionaries = []
        for model in models:
            model_dic = model.as_dict()
            model_predictions = predictions_per_model[model.id]
            model_dic["classify_time"] = round(
                model_predictions.classify_time + model_predictions.model_load_time, 1
            )
            model_dictionaries.append(model_dic)

        meta_data["models"] = model_dictionaries
        if self.config.classify.meta_to_stdout:
            print(json.dumps(meta_data, cls=tools.CustomJSONEncoder))
        else:
            with open(meta_filename, "w") as f:
                json.dump(meta_data, f, indent=4, cls=tools.CustomJSONEncoder)
        return meta_data
