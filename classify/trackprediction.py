import attr
import logging
import numpy as np

UNIFORM_PRIOR = False


class Predictions:
    def __init__(self, labels):
        self.labels = labels
        self.prediction_per_track = {}

    def get_or_create_prediction(self, track, keep_all=True):
        prediction = self.prediction_per_track.setdefault(
            track.get_id(), TrackPrediction(track.get_id(), track.start_frame, keep_all)
        )
        return prediction

    def clear_predictions(self):
        self.prediction_per_track = {}

    def prediction_for(self, track_id):
        return self.prediction_per_track.get(track_id)

    def guesses_for(self, track_id):
        prediction = self.prediction_per_track.get(track_id)
        if prediction:
            return prediction.guesses(self.labels)
        return []

    def print_prediction(self, track_id):
        self.prediction_for(track_id).print_prediction(self.labels)

    def prediction_description(self, track_id):
        return self.prediction_per_track.get(track_id).description(self.labels)


class TrackPrediction:
    """
    Class to hold the information about the predicted class of a track.
    Predictions are recorded for every frame, and methods provided for extracting the final predicted class of the
    track.
    """

    def __init__(self, track_id, start_frame, keep_all=True):
        self.track_prediction = None
        self.state = None
        self.track_id = track_id
        self.predictions = []
        self.smoothed_predictions = []
        self.smoothed_novelties = []
        self.uniform_prior = False
        self.class_best_score = None
        self.track_prediction = None

        self.last_frame_classified = start_frame
        self.num_frames_classified = 0
        self.keep_all = keep_all
        self.max_novelty = 0
        self.novelty_sum = 0

    def classified_clip(
        self, predictions, smoothed_predictions, smoothed_novelties, last_frame
    ):
        self.last_frame_classified = last_frame
        self.num_frames_classified = len(predictions)
        self.smoothed_predictions = smoothed_predictions
        self.predictions = predictions
        self.smoothed_novelties = smoothed_novelties
        self.class_best_score = np.max(self.smoothed_predictions, axis=0)
        self.max_novelty = max(self.smoothed_novelties)
        self.novelty_sum = sum(self.smoothed_novelties)

    def classified_frame(self, frame_number, prediction, mass_scale=1, novelty=None):
        self.last_frame_classified = frame_number
        self.num_frames_classified += 1
        if novelty:
            self.max_novelty = max(self.max_novelty, novelty)
            self.novelty_sum += novelty
        smoothed_prediction, smoothed_novelty = self.smooth_prediction(
            prediction, mass_scale=mass_scale, novelty=novelty
        )
        if self.keep_all:
            self.predictions.append(prediction)
            self.smoothed_predictions.append(smoothed_prediction)
            self.smoothed_novelties.append(smoothed_novelty)
        else:
            self.predictions = [prediction]
            self.smoothed_predictions = [smoothed_prediction]
            self.smoothed_novelties = [smoothed_novelty]

        if self.class_best_score is None:
            self.class_best_score = smoothed_prediction
        else:
            self.class_best_score = np.maximum(
                self.class_best_score, smoothed_prediction
            )

    def smooth_prediction(self, prediction, mass_scale=1, novelty=None):
        prediction_smooth = 0.1
        smooth_novelty = None
        prev_prediction = None
        adjusted_prediction = prediction * mass_scale
        if len(self.smoothed_predictions):
            prev_prediction = self.smoothed_predictions[-1]
        num_labels = len(prediction)
        if prev_prediction is None:
            if UNIFORM_PRIOR:
                smooth_prediction = np.ones([num_labels]) * (1 / num_labels)
            else:
                smooth_prediction = adjusted_prediction
            if novelty:
                smooth_novelty = 0.5
        else:
            smooth_prediction = (
                1 - prediction_smooth
            ) * prev_prediction + prediction_smooth * adjusted_prediction
            if novelty:
                smooth_novelty = (
                    1 - prediction_smooth
                ) * smooth_novelty + prediction_smooth * novelty
        return smooth_prediction, smooth_novelty

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

    def get_classified_footer(self, labels, frame_number=None):
        # self.track_prediction = TrackPrediction(self.predictions, self.novelties)
        if frame_number is None or frame_number >= len(self.smoothed_novelties):
            return "({:.1f} {})\nnovelty={:.2f}".format(
                self.max_score * 10, labels[self.best_label_index], self.max_novelty
            )
        if self.smoothed_predictions:
            return "({:.1f} {})\nnovelty={:.2f}".format(
                self.score_at_time(frame_number) * 10,
                labels[self.label_at_time(frame_number)],
                self.novelty_at(frame_number),
            )
        else:
            return "no classification"

    def get_result(self, labels):
        if self.smoothed_predictions:
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
        if score is None:
            return None

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
        return self.num_frames_classified

    def novelty_at(self, n=None):
        if n is None:
            return self.max_novelty

        if self.smoothed_novelties is None:
            return None
        return self.smoothed_novelties[n]

    @property
    def best_label_index(self):
        if self.class_best_score is None:
            return None
        return np.argmax(self.class_best_score)

    @property
    def max_score(self):
        if self.class_best_score is None:
            return None

        return float(np.amax(self.class_best_score))

    @property
    def average_novelty(self):
        """ average novelty for this track """
        return self.novelty_sum / self.num_frames_classified

    @property
    def clarity(self):
        """ The distance between our highest scoring class and second highest scoring class. """
        if self.class_best_score is None or len(self.class_best_score) < 2:
            return None
        return self.max_score - self.score(2)

    def label_index(self, n=None):
        """ index of label of nth best guess. """

        if n is None:
            return self.best_label_index
        if self.class_best_score is None:
            return None
        return int(np.argsort(self.class_best_score)[-n])

    def score(self, n=None):
        """ class score of nth best guess. """
        if n is None:
            return self.max_score
        if self.class_best_score is None:
            return None
        return float(sorted(self.class_best_score)[-n])

    def label_at_time(self, frame_number, n=-1):
        """ class label of nth best guess at a point in time."""

        if n is None:
            return None
        return int(np.argsort(self.smoothed_predictions[frame_number])[-n])

    def score_at_time(self, frame_number, n=-1):
        """ class label of nth best guess at a point in time."""
        if n is None:
            return None
        return float(sorted(self.smoothed_predictions[frame_number])[-n])

    def print_prediction(self, labels):
        logging.info(
            "Track {} is {}".format(self.track_id, self.get_classified_footer(labels))
        )

    def guesses(self, labels):
        guesses = [
            "{} ({:.1f})".format(labels[self.label_index(i)], self.score(i) * 10)
            for i in range(1, min(len(labels), 4))
            if self.score(i) > 0.5
        ]
        return guesses


@attr.s(slots=True)
class TrackResult:
    what = attr.ib()
    avg_novelty = attr.ib()
    max_novelty = attr.ib()
    confidence = attr.ib()
