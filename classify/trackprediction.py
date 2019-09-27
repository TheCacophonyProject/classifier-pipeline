import attr
import logging
import numpy as np


class Predictions:
    def __init__(self, labels):
        self.labels = labels
        self.prediction_per_track = {}

    def get_or_create_prediction(self, track):
        prediction = self.prediction_per_track.setdefault(
            track.get_id(), TrackPrediction(track.get_id(), track.start_frame)
        )
        return prediction

    def clear_predictions(self):
        self.prediction_per_track = {}

    def prediction_for(self, track_id):
        return self.prediction_per_track.get(track_id)

    def print_prediction(self, track_id):
        self.prediction_for(track_id).print_prediction(self.labels)

class TrackPrediction:
    """
    Class to hold the information about the predicted class of a track.
    Predictions are recorded for every frame, and methods provided for extracting the final predicted class of the
    track.
    """

    def __init__(self, track_id, start_frame):
        self.track_prediction = None
        self.state = None
        self.track_id = track_id
        self.predictions = []
        self.novelties = []
        self.uniform_prior = False
        self.class_best_score = None
        self.track_prediction = None
        self.last_frame_classified = start_frame
        self.num_frames_classified = 0

    def classified_clip(self, predictions, novelties, last_frame):
        self.last_frame_classified = last_frame
        self.num_frames_classified = len(predictions)
        self.predictions = predictions
        self.novelties = novelties
        self.class_best_score = np.max(self.predictions, axis=0)

    def classified_frame(self, frame_number, prediction, novelty):
        self.last_frame_classified = frame_number
        self.num_frames_classified += 1
        self.predictions.append(prediction)
        self.novelties.append(novelty)
        if self.class_best_score is None:
            self.class_best_score = prediction
        else:
            self.class_best_score = np.maximum(self.class_best_score, prediction)

    def get_priority(self, frame_number):
        skipepd_frames = frame_number - self.last_frame_classified
        priority = skipepd_frames / 9
        if self.num_frames_classified == 0:
            priority += 2
        logging.debug(
            "priority {} for track# {} num_frames {} last classified {}".format(
                priority,
                self.track_id,
                self.num_frames_classified,
                self.last_frame_classified,
            )
        )
        return priority

    def get_prediction(self, labels):
        return self.description(labels)

    def get_classified_footer(self, labels):
        # self.track_prediction = TrackPrediction(self.predictions, self.novelties)
        if self.predictions:
            return "({:.1f} {})\nnovelty={:.2f}".format(
                self.max_score * 10, labels[self.best_label_index], self.max_novelty
            )
        else:
            return "no classification"

    def get_result(self, labels):
        if self.predictions:
            return TrackResult(
                labels[self.best_label_index],
                self.average_novelty,
                self.max_novelty,
                self.max_score,
            )
        else:
            return None

    def description(self, labels):
        """
        Returns a summary description of this prediction
        :param classes: Name of class for each label.
        :return:
        """
        score = self.max_score
        if score > 0.5:
            first_guess = "{} {:.1f} (clarity {:.1f})".format(
                labels[self.best_label_index], score * 10, self.clarity * 10
            )
        else:
            first_guess = "[nothing]"

        second_score = self.score(2)

        if second_score > 0.5:
            second_guess = "[second guess - {} {:.1f}]".format(
                labels[self.label_index(2)], second_score * 10
            )
        else:
            second_guess = ""

        return (first_guess + " " + second_guess).strip()

    @property
    def num_frames(self):
        return len(self.predictions)

    @property
    def best_label_index(self):
        if self.class_best_score is None:
            return None
        return np.argmax(self.class_best_score)

    @property
    def max_novelty(self):
        """ maximum novelty for this track """
        return max(self.novelties)

    @property
    def max_score(self):
        if self.class_best_score is None:
            return None

        return float(np.amax(self.class_best_score))

    @property
    def average_novelty(self):
        """ average novelty for this track """
        return sum(self.novelties) / len(self.novelties)

    @property
    def clarity(self):
        """ The distance between our highest scoring class and second highest scoring class. """
        if self.class_best_score is None or len(self.class_best_score) < 2:
            return None
        return self.max_score - self.score(2)

    def label_index(self, n=1):
        """ idnex of label of nth best guess. """
        if self.class_best_score is None:
            return None
        return int(np.argsort(self.class_best_score)[-n])

    def score(self, n=1):
        """ class score of nth best guess. """
        if self.class_best_score is None:
            return None
        return float(sorted(self.class_best_score)[-n])

    def label_at_time(self, frame_number, n=1):
        """ class label of nth best guess at a point in time."""
        if len(self.predictions < n):
            return None
        return int(np.argsort(self.predictions[frame_number])[-n])

    def score_at_time(self, frame_number, n=1):
        """ class label of nth best guess at a point in time."""
        if len(self.predictions < n):
            return None
        return float(sorted(self.predictions[frame_number])[-n])


    def print_prediction(self, labels):
        logging.info(
            "Track {} is {}".format(
                self.track_id, self.get_classified_footer(labels)
            )
        )

@attr.s(slots=True)
class TrackResult:
    what = attr.ib()
    avg_novelty = attr.ib()
    max_novelty = attr.ib()
    confidence = attr.ib()
