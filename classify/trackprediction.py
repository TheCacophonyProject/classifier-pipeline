
import numpy as np

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

    def label(self, n = 1):
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

    def score(self, n = 1):
        """ class score of nth best guess. """
        return float(sorted(self.class_best_score)[-n])

    def label_at_time(self, frame_number, n = 1):
        """ class label of nth best guess at a point in time."""
        return int(np.argsort(self.prediction_history[frame_number])[-n])

    def score_at_time(self, frame_number, n = 1):
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
                classes[self.label()], self.score() * 10, self.clarity * 10)
        else:
            first_guess = "[nothing]"

        if self.score(2) > 0.5:
            second_guess = "[second guess - {} {:.1f}]".format(
                classes[self.label(2)], self.score(2) * 10)
        else:
            second_guess = ""

        return (first_guess+" "+second_guess).strip()

    @property
    def num_frames(self):
        return len(self.prediction_history)
