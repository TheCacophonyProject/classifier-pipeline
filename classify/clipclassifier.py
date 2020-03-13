import json
import logging
import os.path
import time
from typing import Dict

from datetime import datetime
import numpy as np

from classify.trackprediction import Predictions
from load.clip import Clip
from load.cliptrackextractor import ClipTrackExtractor
from ml_tools import tools
from ml_tools.cptvfileprocessor import CPTVFileProcessor
import ml_tools.globals as globs
from ml_tools.model import Model
from ml_tools.dataset import Preprocessor
from ml_tools.previewer import Previewer
from track.track import Track


class ClipClassifier(CPTVFileProcessor):
    """ Classifies tracks within CPTV files. """

    # skips every nth frame.  Speeds things up a little, but reduces prediction quality.
    FRAME_SKIP = 1

    def __init__(self, config, tracking_config, model_file):
        """ Create an instance of a clip classifier"""

        super(ClipClassifier, self).__init__(config, tracking_config)
        self.model_file = model_file

        # prediction record for each track
        self.predictions = Predictions(self.classifier.labels)

        self.previewer = Previewer.create_if_required(config, config.classify.preview)

        self.start_date = None
        self.end_date = None
        self.cache_to_disk = self.config.classify.cache_to_disk
        # enables exports detailed information for each track.  If preview mode is enabled also enables track previews.
        self.enable_per_track_information = False
        self.track_extractor = ClipTrackExtractor(
            self.config.tracking,
            self.config.use_opt_flow
            or config.classify.preview == Previewer.PREVIEW_TRACKING,
            self.config.classify.cache_to_disk,
        )

    def preprocess(self, frame, thermal_reference):
        """
        Applies preprocessing to frame required by the model.
        :param frame: numpy array of shape [C, H, W]
        :return: preprocessed numpy array
        """

        # note, would be much better if the model did this, as only the model knows how preprocessing occurred during
        # training
        frame = np.float32(frame)
        frame[2 : 3 + 1] *= 1 / 256
        frame[0] -= thermal_reference

        return frame

    def identify_track(self, clip: Clip, track: Track):
        """
        Runs through track identifying segments, and then returns it's prediction of what kind of animal this is.
        One prediction will be made for every frame.
        :param track: the track to identify.
        :return: TrackPrediction object
        """

        # uniform prior stats start with uniform distribution.  This is the safest bet, but means that
        # it takes a while to make predictions.  When off the first prediction is used instead causing
        # faster, but potentially more unstable predictions.
        UNIFORM_PRIOR = False

        num_labels = len(self.classifier.labels)
        prediction_smooth = 0.1

        smooth_prediction = None
        smooth_novelty = None

        prediction = 0.0
        novelty = 0.0
        try:
            fp_index = self.classifier.labels.index("false-positive")
        except ValueError:
            fp_index = None

        # go through making classifications at each frame
        # note: we should probably be doing this every 9 frames or so.
        state = None
        track_prediction = self.predictions.get_or_create_prediction(track)
        for i, region in enumerate(track.bounds_history):
            frame = clip.frame_buffer.get_frame(region.frame_number)
            track_data = track.crop_by_region(frame, region)

            # note: would be much better for the tracker to store the thermal references as it goes.
            # frame = clip.frame_buffer.get_frame(frame_number)
            thermal_reference = np.median(frame.thermal)
            # track_data = track.crop_by_region_at_trackframe(frame, i)
            if i % self.FRAME_SKIP == 0:
                # we use a tighter cropping here so we disable the default 2 pixel inset
                frames = Preprocessor.apply(
                    [track_data], [thermal_reference], default_inset=0
                )

                if frames is None:
                    logging.info(
                        "Frame {} of track could not be classified.".format(
                            region.frame_number
                        )
                    )
                    return
                frame = frames[0]
                (
                    prediction,
                    novelty,
                    state,
                ) = self.classifier.classify_frame_with_novelty(frame, state)
                # make false-positive prediction less strong so if track has dead footage it won't dominate a strong
                # score
                if fp_index is not None:
                    prediction[fp_index] *= 0.8

                # a little weight decay helps the model not lock into an initial impression.
                # 0.98 represents a half life of around 3 seconds.
                state *= 0.98

                # precondition on weight,  segments with small mass are weighted less as we can assume the error is
                # higher here.
                mass = region.mass

                # we use the square-root here as the mass is in units squared.
                # this effectively means we are giving weight based on the diameter
                # of the object rather than the mass.
                mass_weight = np.clip(mass / 20, 0.02, 1.0) ** 0.5

                # cropped frames don't do so well so restrict their score
                cropped_weight = 0.7 if region.was_cropped else 1.0

                prediction *= mass_weight * cropped_weight

            if smooth_prediction is None:
                if UNIFORM_PRIOR:
                    smooth_prediction = np.ones([num_labels]) * (1 / num_labels)
                else:
                    smooth_prediction = prediction
                smooth_novelty = 0.5
            else:
                smooth_prediction = (
                    1 - prediction_smooth
                ) * smooth_prediction + prediction_smooth * prediction
                smooth_novelty = (
                    1 - prediction_smooth
                ) * smooth_novelty + prediction_smooth * novelty
            track_prediction.classified_frame(
                region.frame_number, smooth_prediction, smooth_novelty
            )
        return track_prediction

    @property
    def classifier(self):
        """
        Returns a classifier object, which is created on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        if globs._classifier is None:
            t0 = datetime.now()
            logging.info("classifier loading")
            globs._classifier = Model(
                train_config=self.config.train,
                session=tools.get_session(disable_gpu=not self.config.use_gpu),
            )
            globs._classifier.load(self.model_file)
            logging.info("classifier loaded ({})".format(datetime.now() - t0))
        return globs._classifier

    def get_meta_data(self, filename):
        """ Reads meta-data for a given cptv file. """
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

    def get_classify_filename(self, input_filename):
        return os.path.splitext(
            os.path.join(
                self.config.classify.classify_folder, os.path.basename(input_filename)
            )
        )[0]

    def process_file(self, filename, **kwargs):
        """
        Process a file extracting tracks and identifying them.
        :param filename: filename to process
        :param enable_preview: if true an MPEG preview file is created.
        """

        if not os.path.exists(filename):
            raise Exception("File {} not found.".format(filename))

        logging.info("Processing file '{}'".format(filename))

        start = time.time()
        clip = Clip(self.tracker_config, filename)
        self.track_extractor.parse_clip(clip)

        classify_name = self.get_classify_filename(filename)
        destination_folder = os.path.dirname(classify_name)

        if not os.path.exists(destination_folder):
            logging.info("Creating folder {}".format(destination_folder))
            os.makedirs(destination_folder)

        mpeg_filename = classify_name + ".mp4"

        meta_filename = classify_name + ".txt"

        logging.info(os.path.basename(filename) + ":")

        for i, track in enumerate(clip.tracks):
            prediction = self.identify_track(clip, track)
            description = prediction.description(self.classifier.labels)
            logging.info(
                " - [{}/{}] prediction: {}".format(i + 1, len(clip.tracks), description)
            )

        if self.previewer:
            logging.info("Exporting preview to '{}'".format(mpeg_filename))
            self.previewer.export_clip_preview(mpeg_filename, clip, self.predictions)
        logging.info("saving meta data")
        self.save_metadata(filename, meta_filename, clip)
        self.predictions.clear_predictions()

        if self.tracker_config.verbose:
            ms_per_frame = (
                (time.time() - start) * 1000 / max(1, len(clip.frame_buffer.frames))
            )
            logging.info("Took {:.1f}ms per frame".format(ms_per_frame))

    def save_metadata(self, filename, meta_filename, clip):
        if self.cache_to_disk:
            clip.frame_buffer.remove_cache()

        # read in original metadata
        meta_data = self.get_meta_data(filename)

        # record results in text file.
        save_file = {}
        save_file["source"] = filename
        start, end = clip.start_and_end_time_absolute()
        save_file["start_time"] = start.isoformat()
        save_file["end_time"] = end.isoformat()
        save_file["algorithm"] = {}
        save_file["algorithm"]["model"] = self.model_file
        save_file["algorithm"]["tracker_version"] = clip.VERSION
        save_file["algorithm"]["tracker_config"] = self.tracker_config.as_dict()
        if meta_data:
            save_file["camera"] = meta_data["Device"]["devicename"]
            save_file["cptv_meta"] = meta_data
            save_file["original_tag"] = meta_data["primary_tag"]
        save_file["tracks"] = []
        for track in clip.tracks:
            track_info = {}
            prediction = self.predictions.prediction_for(track.get_id())
            start_s, end_s = clip.start_and_end_in_secs(track)
            save_file["tracks"].append(track_info)
            track_info["start_s"] = round(start_s, 2)
            track_info["end_s"] = round(end_s, 2)
            track_info["num_frames"] = prediction.num_frames
            track_info["frame_start"] = track.start_frame
            track_info["frame_end"] = track.end_frame
            track_info["label"] = self.classifier.labels[prediction.best_label_index]
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
                track_time = round(region.frame_number / clip.frames_per_second, 2)
                positions.append([track_time, region])
            track_info["positions"] = positions

        if self.config.classify.meta_to_stdout:
            print(json.dumps(save_file, cls=tools.CustomJSONEncoder))
        else:
            with open(meta_filename, "w") as f:
                json.dump(save_file, f, indent=4, cls=tools.CustomJSONEncoder)
