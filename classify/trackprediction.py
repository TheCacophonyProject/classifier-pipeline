import numpy as np


class RollingTrackPrediction:
    def __init__(self, track_id, start_frame):
        self.track_prediction = None
        self.state = None
        self.track_id = track_id
        self.predictions = []
        self.novelties = []
        self.uniform_prior = False
        self.class_best_score = []
        self.track_prediction = None
        self.last_frame_classified = start_frame
        self.num_frames_classified = 0

    def get_priority(self, frame_number):
        skipepd_frames = frame_number - self.last_frame_classified
        priority = skipepd_frames / 9
        if self.num_frames_classified == 0:
            priority += 2
        # elif self.num_frames_classified > 30:
        #     priority -= 1
        print(
            "priority {} is {} num_frames {} last classified {} frame {} ".format(
                priority,
                self.track_id,
                self.num_frames_classified,
                self.last_frame_classified,
                frame_number,
            )
        )
        return priority

    def classified(self, frame_number, prediction, novelty):
        self.last_frame_classified = frame_number
        self.num_frames_classified += 1
        self.predictions.append(prediction)
        self.novelties.append(novelty)

    def get_prediction(self, labels):
        self.track_prediction = TrackPrediction(self.predictions, self.novelties)
        return self.track_prediction.description(labels)

    def get_classified_footer(self, labels):
        # self.track_prediction = TrackPrediction(self.predictions, self.novelties)
        if len(self.predictions):
            class_best_score = np.max(self.predictions, axis=0).tolist()
            score = max(class_best_score)
            label = labels[np.argmax(class_best_score)]
            max_novelty = max(self.novelties)
            return "({:.1f} {})\nnovelty={:.2f}".format(score * 10, label, max_novelty)
        else:
            return "no classification"

    def get_result(self, labels):
        if len(self.predictions):
            class_best_score = np.max(self.predictions, axis=0).tolist()
            score = max(class_best_score)
            label = labels[np.argmax(class_best_score)]
            avg_novelty = sum(self.novelties) / len(self.novelties)
            max_novelty = max(self.novelties)
            return TrackResult(label, avg_novelty, max_novelty, score)
        else:
            return None


class TrackResult:
    def __init__(self, label, avg_novelty, max_novelty, score):
        self.what = label
        self.avg_novelty = avg_novelty
        self.max_novelty = max_novelty
        self.confidence = score


class TrackPrediction:
    """
    Class to hold the information about the predicted class of a track.
    Predictions are recorded for every frame, and methods provided for extracting the final predicted class of the
    track.
    """

    def __init__(self, prediction_history, novelty_history):
        """
        Setup track prediction with given prediction history
        :param prediction_history: list of predictions for each frame of this track.
        :param novelty_history: list of novelty scores for each frame of this track.
        """
        self.prediction_history = prediction_history.copy()
        self.novelty_history = novelty_history.copy()
        self.class_best_score = np.max(np.float32(prediction_history), axis=0).tolist()

    def label(self, n=1):
        """ class label of nth best guess. """
        return int(np.argsort(self.class_best_score)[-n])

    @property
    def max_novelty(self):
        """ maximum novelty for this track """
        return max(self.novelty_history)

    @property
    def average_novelty(self):
        """ average novelty for this track """
        return sum(self.novelty_history) / len(self.novelty_history)

    def score(self, n=1):
        """ class score of nth best guess. """
        return float(sorted(self.class_best_score)[-n])

    def label_at_time(self, frame_number, n=1):
        """ class label of nth best guess at a point in time."""
        return int(np.argsort(self.prediction_history[frame_number])[-n])

    def score_at_time(self, frame_number, n=1):
        """ class label of nth best guess at a point in time."""
        return float(sorted(self.prediction_history[frame_number])[-n])

    @property
    def clarity(self):
        """ The distance between our highest scoring class and second highest scoring class. """
        return self.score(1) - self.score(2)

    def description(self, classes):
        """
        Returns a summary description of this prediction
        :param classes: Name of class for each label.
        :return:
        """

        if self.score() > 0.5:
            first_guess = "{} {:.1f} (clarity {:.1f})".format(
                classes[self.label()], self.score() * 10, self.clarity * 10
            )
        else:
            first_guess = "[nothing]"

        if self.score(2) > 0.5:
            second_guess = "[second guess - {} {:.1f}]".format(
                classes[self.label(2)], self.score(2) * 10
            )
        else:
            second_guess = ""

        return (first_guess + " " + second_guess).strip()

    @property
    def num_frames(self):
        return len(self.prediction_history)
