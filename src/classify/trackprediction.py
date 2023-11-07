import attr
import logging
import numpy as np

# uniform prior stats start with uniform distribution.  This is the safest bet, but means that
# it takes a while to make predictions.  When off the first prediction is used instead causing
# faster, but potentially more unstable predictions.
UNIFORM_PRIOR = False


class Predictions:
    def __init__(self, labels, model):
        self.labels = labels
        self.prediction_per_track = {}
        self.model = model
        self.model_load_time = None

    def get_or_create_prediction(self, track, keep_all=True):
        prediction = self.prediction_per_track.setdefault(
            track.get_id(),
            TrackPrediction(
                track.get_id(),
                self.labels,
                keep_all=keep_all,
                start_frame=track.start_frame,
            ),
        )
        return prediction

    def clear_predictions(self):
        self.prediction_per_track = {}

    def prediction_for(self, track_id):
        return self.prediction_per_track.get(track_id)

    def guesses_for(self, track_id):
        prediction = self.prediction_per_track.get(track_id)
        if prediction:
            return prediction.guesses()
        return []

    def print_predictions(self, track_id):
        self.prediction_for(track_id).print_prediction()

    def prediction_description(self, track_id):
        return self.prediction_per_track.get(track_id).description()

    @property
    def classify_time(self):
        classify_time = [
            prediction.classify_time
            for prediction in self.prediction_per_track.values()
            if prediction.classify_time is not None
        ]
        return np.sum(classify_time)


class TrackPrediction:
    """
    Class to hold the information about the predicted class of a track.
    Predictions are recorded for every frame, and methods provided for extracting the final predicted class of the
    track.
    """

    def __init__(self, track_id, labels, keep_all=True, start_frame=None):
        try:
            fp_index = labels.index("false-positive")
        except ValueError:
            fp_index = None
        self.track_id = track_id
        self.predictions = []
        self.prediction_frames = []
        self.fp_index = fp_index
        self.smoothed_predictions = []
        self.class_best_score = None
        self.start_frame = start_frame

        self.last_frame_classified = None
        self.num_frames_classified = 0
        self.keep_all = keep_all
        self.labels = labels
        self.classify_time = None
        self.tracking = False
        self.masses = []

    def classified_clip(
        self, predictions, smoothed_predictions, prediction_frames, top_score=None
    ):
        self.num_frames_classified = len(predictions)
        if smoothed_predictions is None:
            self.smoothed_predictions = predictions
        else:
            self.smoothed_predictions = smoothed_predictions
        self.predictions = predictions
        self.prediction_frames = prediction_frames

        if self.num_frames_classified > 0:
            self.class_best_score = np.sum(self.smoothed_predictions, axis=0)
            # normalize so it sums to 1
            if top_score is None:
                self.class_best_score = self.class_best_score / np.sum(
                    self.class_best_score
                )
            else:
                # possibility it doesn't sum to 1 i.e multi label model
                self.class_best_score /= top_score

    def normalize_score(self):
        # normalize so it sums to 1
        if self.class_best_score is not None:
            self.class_best_score = self.class_best_score / np.sum(
                self.class_best_score
            )

    def classified_frames(self, frame_numbers, prediction, mass):
        self.num_frames_classified += len(frame_numbers)
        smoothed_prediction = prediction**2 * mass
        if self.keep_all:
            self.prediction_frames.append(frame_numbers)
            self.predictions.append(prediction)
            self.smoothed_predictions.append(smoothed_prediction)

        else:
            self.prediction_frames = [frame_numbers]
            self.predictions = [prediction]
            self.smoothed_predictions = [smoothed_prediction]
        if self.class_best_score is None:
            self.class_best_score = smoothed_prediction.copy()
        else:
            self.class_best_score += smoothed_prediction

    def classified_frame(self, frame_number, prediction, mass):
        self.prediction_frames.append([frame_number])
        self.last_frame_classified = frame_number
        self.num_frames_classified += 1
        self.masses.append(mass)
        smoothed_prediction = prediction * prediction * mass
        if self.keep_all:
            self.predictions.append(prediction)
            self.smoothed_predictions.append(smoothed_prediction)

        else:
            self.predictions = [prediction]
            self.smoothed_predictions = [smoothed_prediction]

        if self.class_best_score is None:
            self.class_best_score = smoothed_prediction
        else:
            self.class_best_score += smoothed_prediction

    def get_priority(self, frame_number):
        if self.tracking:
            return 100
        if self.last_frame_classified:
            skipepd_frames = frame_number - self.last_frame_classified
        else:
            skipepd_frames = frame_number - self.start_frame

        priority = skipepd_frames / 9
        if self.num_frames_classified == 0:
            priority += 2
        if self.fp_index and self.best_label_index == self.fp_index:
            # dont bother with fps unless nothing else to do
            priority -= 100
        logging.debug(
            "priority {} for track# {} num_frames {} last classified {} skipped {}".format(
                priority,
                self.track_id,
                self.num_frames_classified,
                self.last_frame_classified,
                skipepd_frames,
            )
        )
        return priority

    def get_prediction(self):
        return self.description()

    def get_classified_footer(self, frame_number=None):
        if len(self.smoothed_predictions) == 0 or not self.keep_all:
            return "no classification"
        score = self.score_at_time(frame_number) * 10
        label = self.labels[self.label_at_time(frame_number)]

        score_2 = self.score_at_time(frame_number, n=2) * 10
        label_2 = self.labels[self.label_at_time(frame_number, n=2)]
        footer = "({:.1f} {}) second guess ({:.1f} {})".format(
            score, label, score_2, label_2
        )
        return footer

    def get_result(self):
        if self.smoothed_predictions:
            return TrackResult(
                self.labels[self.best_label_index],
                self.max_score,
            )
        else:
            return None

    def description(self):
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
                self.labels[self.best_label_index], score * 10, self.clarity * 10
            )
        else:
            first_guess = "[nothing]"

        second_score = self.score(2)

        if second_score > 0.5:
            second_guess = "[second guess - {} {:.1f}]".format(
                self.labels[self.label_index(2)], second_score * 10
            )
        else:
            second_guess = ""

        return (first_guess + " " + second_guess).strip()

    @property
    def num_frames(self):
        return self.num_frames_classified

    def predicted_tag(self):
        index = self.best_label_index
        if index is None:
            return None
        return self.labels[index]

    def class_confidences(self):
        confidences = {}
        if self.class_best_score is None:
            return confidences
        for i, value in enumerate(self.class_best_score):
            confidences[self.labels[i]] = round(float(value), 3)
        return confidences

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

    def clarity_at(self, frame):
        pred = self.predictions[frame]
        best = np.argsort(pred)
        return pred[best[-1]] - pred[best[-2]]

    @property
    def clarity(self):
        """The distance between our highest scoring class and second highest scoring class."""
        if self.class_best_score is None or len(self.class_best_score) < 2:
            return None
        return self.max_score - self.score(2)

    def label_index(self, n=None):
        """index of label of nth best guess."""

        if n is None:
            return self.best_label_index
        if self.class_best_score is None:
            return None
        return int(np.argsort(self.class_best_score)[-n])

    def score(self, n=None):
        """class prediction of nth best guess."""
        if n is None:
            return self.max_score
        if self.class_best_score is None:
            return None
        return float(sorted(self.class_best_score)[-n])

    def label_at_time(self, frame_number, n=1):
        """class label of nth best guess at a point in time."""
        if n is None:
            return None
        frames_per_prediction = len(self.smoothed_predictions) / self.num_frames
        prediction_index = int(frame_number / frames_per_prediction) + 1
        average = np.mean(self.smoothed_predictions[:prediction_index], axis=0)
        return int(np.argsort(average)[-n])

    def label_at_time(self, frame_number, n=1):
        """class prediction of nth best at a point in time."""
        if n is None:
            return None
        predictions = []
        class_best_score = None
        for i, frames in enumerate(self.prediction_frames):
            a_min = np.amin(frames)
            a_max = np.amax(frames)

            if a_min <= frame_number:
                predictions.append(self.smoothed_predictions[i])

        class_best_score = np.sum(predictions, axis=0)
        if len(predictions) == 0:
            return 0
        class_best_score = class_best_score / np.sum(class_best_score)
        return int(np.argsort(class_best_score)[-n])

    def score_at_time(self, frame_number, n=1):
        """class prediction of nth best at a point in time."""
        if n is None:
            return None
        predictions = []
        class_best_score = None
        for i, frames in enumerate(self.prediction_frames):
            a_min = np.amin(frames)
            a_max = np.amax(frames)

            if a_min <= frame_number:
                predictions.append(self.smoothed_predictions[i])
        class_best_score = np.sum(predictions, axis=0)
        if len(predictions) == 0:
            return 0

        class_best_score = class_best_score / np.sum(class_best_score)
        return float(sorted(class_best_score)[-n])

    def print_prediction(self):
        logging.info(
            "Track {} prediction {}".format(self.track_id, self.get_classified_footer())
        )

    def guesses(self):
        guesses = [
            "{} ({:.1f})".format(self.labels[self.label_index(i)], self.score(i) * 10)
            for i in range(1, min(len(self.labels), 4))
            if self.score(i) and self.score(i) > 0.5
        ]
        return guesses

    def get_metadata(self):
        prediction_meta = {}
        if self.classify_time is not None:
            prediction_meta["classify_time"] = round(self.classify_time, 1)

        prediction_meta["label"] = self.predicted_tag()
        # GP makes api pick up the label this will change when logic is moved to API
        prediction_meta["confident_tag"] = self.predicted_tag()

        prediction_meta["confidence"] = (
            round(self.max_score, 2) if self.max_score else 0
        )
        prediction_meta["clarity"] = round(self.clarity, 3) if self.clarity else 0
        prediction_meta["all_class_confidences"] = {}
        if self.prediction_frames is not None:
            prediction_meta["prediction_frames"] = self.prediction_frames

        # for ease always multiply by 100, depending on smoothing applied values might be large
        prediction_meta["predictions"] = np.uint32(
            np.round(100 * self.smoothed_predictions)
        )
        if self.class_best_score is not None:
            for i, value in enumerate(self.class_best_score):
                label = self.labels[i]
                prediction_meta["all_class_confidences"][label] = round(value, 3)
        return prediction_meta


@attr.s(slots=True)
class TrackResult:
    what = attr.ib()
    confidence = attr.ib()
