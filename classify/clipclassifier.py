import json
import logging
import os.path
import time
from typing import Dict

from datetime import datetime, timedelta
import numpy as np

from classify.trackprediction import TrackPrediction
from ml_tools import tools
from ml_tools.cptvfileprocessor import CPTVFileProcessor
from ml_tools.dataset import Preprocessor
import ml_tools.globals as globs
from ml_tools.previewer import Previewer
from ml_tools.model import Model
from track.track import Track
from track.trackextractor import TrackExtractor

class ClipClassifier(CPTVFileProcessor):
    """ Classifies tracks within CPTV files. """

    # skips every nth frame.  Speeds things up a little, but reduces prediction quality.
    FRAME_SKIP = 1

    def __init__(self, config, tracking_config):
        """ Create an instance of a clip classifier"""

        super(ClipClassifier, self).__init__(config, tracking_config)

        # prediction record for each track
        self.track_prediction: Dict[Track, TrackPrediction] = {}

        self.previewer = Previewer.create_if_required(config, config.classify.preview)

        self.start_date = None
        self.end_date = None

        # enables exports detailed information for each track.  If preview mode is enabled also enables track previews.
        self.enable_per_track_information = False

    def preprocess(self, frame, thermal_reference):
        """
        Applies preprocessing to frame required by the model.
        :param frame: numpy array of shape [C, H, W]
        :return: preprocessed numpy array
        """

        # note, would be much better if the model did this, as only the model knows how preprocessing occured during
        # training
        frame = np.float32(frame)
        frame[2:3+1] *= (1 / 256)
        frame[0] -= thermal_reference

        return frame

    def identify_track(self, tracker:TrackExtractor, track: Track):
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

        predictions = []
        novelties = []

        num_labels = len(self.classifier.labels)
        prediction_smooth = 0.1

        smooth_prediction = None
        smooth_novelty = None

        prediction = 0.0
        novelty = 0.0

        fp_index = self.classifier.labels.index('false-positive')

        # go through making clas sifications at each frame
        # note: we should probably be doing this every 9 frames or so.
        state = None
        for i in range(len(track)):
            # note: would be much better for the tracker to store the thermal references as it goes.
            thermal_reference = np.median(tracker.frame_buffer.thermal[track.start_frame + i])

            frame = tracker.get_track_channels(track, i)

            if i % self.FRAME_SKIP == 0:

                # we use a tigher cropping here so we disable the default 2 pixel inset
                frames = Preprocessor.apply([frame], [thermal_reference], default_inset=0)

                if frames is None:
                    logging.info("Frame {} of track could not be classified.".format(i))
                    return

                frame = frames[0]
                prediction, novelty, state = self.classifier.classify_frame_with_novelty(frame, state)

                # make false-positive prediction less strong so if track has dead footage it won't dominate a strong
                # score
                prediction[fp_index] *= 0.8

                # a little weight decay helps the model not lock into an initial impression.
                # 0.98 represents a half life of around 3 seconds.
                state *= 0.98

                # precondition on weight,  segments with small mass are weighted less as we can assume the error is
                # higher here.
                mass = track.bounds_history[i].mass

                # we use the square-root here as the mass is in units squared.
                # this effectively means we are giving weight based on the diameter
                # of the object rather than the mass.
                mass_weight = np.clip(mass / 20, 0.02, 1.0) ** 0.5

                # cropped frames don't do so well so restrict their score
                cropped_weight = 0.7 if track.bounds_history[i].was_cropped else 1.0

                prediction *= mass_weight * cropped_weight

            if smooth_prediction is None:
                if UNIFORM_PRIOR:
                    smooth_prediction = np.ones([num_labels]) * (1 / num_labels)
                else:
                    smooth_prediction = prediction
                smooth_novelty = 0.5
            else:
                smooth_prediction = (1-prediction_smooth) * smooth_prediction + prediction_smooth * prediction
                smooth_novelty = (1-prediction_smooth) * smooth_novelty + prediction_smooth * novelty

            predictions.append(smooth_prediction)
            novelties.append(smooth_novelty)


        return TrackPrediction(predictions, novelties)

    @property
    def classifier(self):
        """
        Returns a classifier object, which is created on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        if globs._classifier is None:
            t0 = datetime.now()
            logging.info("classifier loading")
            globs._classifier = Model(tools.get_session(disable_gpu=not self.config.use_gpu))
            globs._classifier.load(self.config.classify.model)
            logging.info("classifier loaded ({})".format(datetime.now() - t0))

        return globs._classifier

    def get_clip_prediction(self):
        """ Returns list of class predictions for all tracks in this clip. """

        class_best_score = [0 for _ in range(len(self.classifier.labels))]

        # keep track of our highest confidence over every track for each class
        for _, prediction in self.track_prediction.items():
            for i in range(len(self.classifier.labels)):
                class_best_score[i] = max(class_best_score[i], prediction.class_best_score[i])

        results = []
        for n in range(1, 1+len(self.classifier.labels)):
            nth_label = int(np.argsort(class_best_score)[-n])
            nth_score = float(np.sort(class_best_score)[-n])
            results.append((self.classifier.labels[nth_label], nth_score))

        return results

    def needs_processing(self, filename):
        """
        Returns True if this file needs to be processed, false otherwise.
        :param filename: the full path and filename of the cptv file in question.
        :return: returns true if file should be processed, false otherwise
        """

        # check date filters
        date_part = str(os.path.basename(filename).split("-")[0])
        date = datetime.strptime(date_part, "%Y%m%d")
        if self.start_date and date < self.start_date:
            return False
        if self.end_date and date > self.end_date:
            return False

        # look to see of the destination file already exists.
        base_name = self.get_base_name(filename)
        meta_filename = base_name + '.txt'

        # if no stats file exists we haven't processed file, so reprocess
        if self.config.reprocess:
            return True
        else:
            return not os.path.exists(meta_filename)

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
                    tags.add(record['animal'])

            tags = list(tags)

            if len(tags) == 0:
                tag = 'no tag'
            elif len(tags) == 1:
                tag = tags[0] if tags[0] else "none"
            else:
                print(tags)
                tag = 'multi'
            meta_data["primary_tag"] = tag
            return meta_data
        else:
            return None

    def get_base_name(self, input_filename):
        return os.path.splitext(os.path.join(self.config.classify.classify_folder, os.path.basename(input_filename)))[0]

    def process_all(self, root):
        for root, folders, _ in os.walk(root):
            for folder in folders:
                if folder not in self.config.excluded_folders:
                    self.process_folder(os.path.join(root,folder), tag=folder.lower())

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

        tracker = TrackExtractor(self.tracker_config)
        tracker.load(filename)

        tracker.extract_tracks()

        if len(tracker.tracks) > 0:
            tracker.generate_optical_flow()

        base_name = self.get_base_name(filename)
        destination_folder = os.path.dirname(base_name)

        if not os.path.exists(destination_folder):
            logging.info("Creating folder {}".format(destination_folder))
            os.makedirs(destination_folder)

        mpeg_filename = base_name + '.mp4'

        meta_filename = base_name + '.txt'

        # reset track predictions
        self.track_prediction = {}

        logging.info(os.path.basename(filename)+":")

        # identify each track
        for i, track in enumerate(tracker.tracks):

            prediction = self.identify_track(tracker, track)

            self.track_prediction[track] = prediction

            description = prediction.description(self.classifier.labels)

            logging.info(" - [{}/{}] prediction: {}".format(i + 1, len(tracker.tracks), description))

        if self.previewer:
            logging.info("Exporting preview to '{}'".format(mpeg_filename))
            prediction_string = ""
            for label, score in self.get_clip_prediction():
                if score > 0.5:
                    prediction_string = prediction_string + " {} {:.1f}".format(label, score * 10)
            self.previewer.export_clip_preview(mpeg_filename, tracker, self.track_prediction)


        self.save_metadata(filename, meta_filename, tracker)

        if self.tracker_config.verbose:
            ms_per_frame = (time.time() - start) * 1000 / max(1, len(tracker.frame_buffer.thermal))
            logging.info("Took {:.1f}ms per frame".format(ms_per_frame))

    def save_metadata(self, filename, meta_filename, tracker):
        # read in original metadata
        meta_data = self.get_meta_data(filename)

        # record results in text file.
        save_file = {}
        save_file['source'] = filename
        save_file['start_time'] = tracker.video_start_time.isoformat()
        save_file['end_time'] = (tracker.video_start_time + timedelta(seconds=len(tracker.frame_buffer.thermal) / 9.0)).isoformat()
        save_file['algorithm'] = {}
        save_file['algorithm']['model'] = self.config.classify.model
        save_file['algorithm']['tracker_version'] = tracker.VERSION
        save_file['algorithm']['tracker_config'] = self.tracker_config.as_dict()

        if meta_data:
            save_file['camera'] = meta_data['Device']['devicename']
            save_file['cptv_meta'] = meta_data
            save_file['original_tag'] = meta_data['primary_tag']
        save_file['tracks'] = []
        for track, prediction in self.track_prediction.items():
            track_info = {}
            start_s, end_s = tracker.start_and_end_in_secs(track)
            save_file['tracks'].append(track_info)
            track_info['start_s'] = start_s
            track_info['end_s'] = end_s
            track_info['num_frames'] = prediction.num_frames
            track_info['frame_start'] = track.start_frame
            track_info['label'] = self.classifier.labels[prediction.label()]
            track_info['confidence'] = round(prediction.score(), 2)
            track_info['clarity'] = round(prediction.clarity, 3)
            track_info['average_novelty'] = round(prediction.average_novelty, 2)
            track_info['max_novelty'] = round(prediction.max_novelty, 2)
            track_info['all_class_confidences'] = {}
            for i, value in enumerate(prediction.class_best_score):
                label = self.classifier.labels[i]
                track_info['all_class_confidences'][label] = round(value, 3)

            positions = []
            for index, bounds in enumerate(track.bounds_history):
                track_time = tracker.frame_time_in_secs(track, index)
                positions.append([track_time, bounds])
            track_info['positions'] = positions

        if self.config.classify.meta_to_stdout:
            output = json.dumps(save_file, indent=4, cls=tools.CustomJSONEncoder)
            print(output)
        else:
            f = open(meta_filename, 'w')
            json.dump(save_file, f, indent=4, cls=tools.CustomJSONEncoder)
