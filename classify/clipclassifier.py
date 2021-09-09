import gc
import json
import logging
import os.path
import time

import numpy as np

from classify.trackprediction import Predictions
from load.clip import Clip
from load.cliptrackextractor import is_affected_by_ffc
from ml_tools import tools
from ml_tools.kerasmodel import KerasModel

from ml_tools.preprocess import preprocess_segment
from ml_tools.previewer import Previewer
from track.track import Track

from classify.thumbnail import get_thumbnail
from cptv import CPTVReader


class ClipClassifier:
    """Classifies tracks within CPTV files."""

    # skips every nth frame.  Speeds things up a little, but reduces prediction quality.
    FRAME_SKIP = 1

    def __init__(self, config, model=None, cache_to_disk=None, resuse_frames=False):
        """Create an instance of a clip classifier"""

        self.config = config
        # super(ClipClassifier, self).__init__(config, tracking_config)
        self.model = model
        # prediction record for each track

        self.previewer = Previewer.create_if_required(config, config.classify.preview)

        if cache_to_disk is None:
            self.cache_to_disk = self.config.classify.cache_to_disk
        else:
            self.cache_to_disk = cache_to_disk
        self.high_quality_optical_flow = self.config.tracking.high_quality_optical_flow
        self.resuse_frames = resuse_frames

    def get_classifier(self, model):
        """
        Returns a classifier object, which is created on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        logging.info("classifier loading")
        classifier = KerasModel(self.config.train)
        classifier.load_model(model.model_file, weights=model.model_weights)

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

    def process(self, source):
        # IF passed a dir extract all cptv files, if a cptv just extract this cptv file
        if os.path.splitext(source)[1] == ".cptv":
            self.process_file(source)
            return
        for folder_path, _, files in os.walk(source):
            for name in files:
                if os.path.splitext(name)[1] == ".cptv":
                    full_path = os.path.join(folder_path, name)
                    self.process_file(full_path)

    def process_file(self, filename):
        """
        Process a file extracting tracks and identifying them.
        :param filename: filename to process
        :param enable_preview: if true an MPEG preview file is created.
        """
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        meta_file = os.path.join(os.path.dirname(filename), base_filename + ".txt")
        if not os.path.exists(filename):
            raise Exception("File {} not found.".format(filename))
        if not os.path.exists(meta_file):
            raise Exception("File {} not found.".format(meta_file))
        meta_data = tools.load_clip_metadata(meta_file)
        logging.info("Processing file '{}'".format(filename))

        start = time.time()
        clip = Clip(self.config.tracking, filename)
        clip.set_frame_buffer(
            self.high_quality_optical_flow,
            self.cache_to_disk,
            self.config.use_opt_flow,
            True,
        )
        clip.load_metadata(
            meta_data,
            self.config.load.tag_precedence,
        )
        frames = []
        with open(clip.source_file, "rb") as f:
            reader = CPTVReader(f)
            clip.calculate_background(reader)
            f.seek(0)
            for frame in reader:
                if frame.background_frame:
                    continue
                clip.set_res(reader.x_resolution, reader.y_resolution)
                clip.add_frame(
                    frame.pix,
                    frame.pix - clip.background,
                    ffc_affected=is_affected_by_ffc(frame),
                )
        predictions_per_model = {}
        if self.model:
            prediction = self.classify_clip(clip, self.model, meta_data)
            predictions_per_model[self.model.id] = prediction
        else:
            for model in self.config.classify.models:
                prediction = self.classify_clip(clip, model, meta_data)
                predictions_per_model[model.id] = prediction

        if self.previewer:
            mpeg_filename = os.path.join(
                os.path.dirname(filename), base_filename + ".mp4"
            )
            logging.info("Exporting preview to '{}'".format(mpeg_filename))

            self.previewer.export_clip_preview(
                mpeg_filename, clip, list(predictions_per_model.values())[0]
            )
        logging.info("saving meta data %s", meta_file)
        models = [self.model] if self.model else self.config.classify.models
        self.save_metadata(
            meta_data,
            meta_file,
            clip,
            predictions_per_model,
            models,
        )

    #
    # def classify_file(self, filename):
    #     base_filename = os.path.splitext(os.path.basename(filename))[0]
    #     meta_file = os.path.join(os.path.dirname(filename), base_filename + ".txt")
    #     if not os.path.exists(filename):
    #         raise Exception("File {} not found.".format(filename))
    #     if not os.path.exists(meta_file):
    #         raise Exception("File {} not found.".format(meta_file))
    #     meta_data = tools.load_clip_metadata(meta_file)
    #     logging.info("Processing file '{}'".format(filename))
    #
    #     start = time.time()
    #     clip = Clip(self.config.tracking, filename)
    #     clip.set_frame_buffer(
    #         self.high_quality_optical_flow,
    #         self.cache_to_disk,
    #         self.config.use_opt_flow,
    #         True,
    #     )
    #     clip.load_metadata(
    #         meta_data,
    #         self.config.load.tag_precedence,
    #     )
    #     frames = []
    #     with open(clip.source_file, "rb") as f:
    #         reader = CPTVReader(f)
    #         clip.calculate_background(reader)
    #         f.seek(0)
    #         for frame in reader:
    #             if frame.background_frame:
    #                 continue
    #             clip.set_res(reader.x_resolution, reader.y_resolution)
    #             clip.add_frame(
    #                 frame.pix,
    #                 frame.pix - clip.background,
    #                 ffc_affected=is_affected_by_ffc(frame),
    #             )
    #     predictions_per_model = {}
    #     if self.model:
    #         prediction = self.classify_clip(clip, self.model)
    #         predictions_per_model[self.model.id] = prediction
    #     else:
    #         for model in self.config.classify.models:
    #             prediction = self.classify_clip(clip, model)
    #             predictions_per_model[model.id] = prediction
    #     return clip, predictions_per_model, metadata

    def classify_clip(self, clip, model, meta_data):
        load_start = time.time()
        classifier = self.get_classifier(model)
        load_time = time.time() - load_start
        logging.info("classifier loaded (%s)", load_time)
        predictions = Predictions(classifier.labels, model)
        predictions.model_load_time = load_time
        for i, track in enumerate(clip.tracks):
            segment_frames = None
            if self.resuse_frames:
                tracks = meta_data.get("tracks")
                meta_track = next(
                    (x for x in tracks if x["id"] == track.get_id()), None
                )
                previous_predictions = meta_track.get("predictions")
                if previous_predictions is not None:
                    previous_prediction = next(
                        (x for x in previous_predictions if x["model_id"] == model.id),
                        None,
                    )
                    if previous_prediction is not None:
                        segment_frames = previous_prediction.get("prediction_frames")
                        if segment_frames is not None:
                            segment_frames = np.uint16(segment_frames)
            prediction = classifier.classify_track(
                clip, track, segment_frames=segment_frames
            )

            predictions.prediction_per_track[track.get_id()] = prediction
            description = prediction.description()
            logging.info(
                " - [{}/{}] prediction: {}".format(i + 1, len(clip.tracks), description)
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
                prediciont_meta = prediction.get_metadata()
                prediciont_meta["model_id"] = model_id
                prediction_info.append(prediciont_meta)
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
        thumbnail_region = get_thumbnail(clip, predictions_per_model)
        meta_data["thumbnail_region"] = thumbnail_region
        if self.config.classify.meta_to_stdout:
            print(json.dumps(meta_data, cls=tools.CustomJSONEncoder))
        else:
            with open(meta_filename, "w") as f:
                json.dump(meta_data, f, indent=4, cls=tools.CustomJSONEncoder)
        if self.cache_to_disk:
            clip.frame_buffer.remove_cache()
