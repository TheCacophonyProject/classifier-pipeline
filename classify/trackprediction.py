import attr
import logging
import numpy as np

CPTV_FILE_WIDTH = 160
CPTV_FILE_HEIGHT = 120
FRAMES_PER_SECOND = 9

MASS_DIFF_PERCENT = 0.20
MAX_VELOCITY = 2
MAX_CROP_PERCENT = 0.3
MIN_CLARITY = 0.8
MIN_PERCENT = 0.85

RES_X = 160
RES_Y = 120
EDGE_PIXELS = 1
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

    def set_important_frames(self):
        for track_id, track_prediction in self.prediction_per_track.items():
            print(
                "set important frames for track starting at",
                track_prediction.track_id,
            )
            label_i = None
            fp_i = None
            # if self.label in self.labels:
            #     label_i = list(labels).index(self.label)
            if "false-positive" in self.labels:
                fp_i = list(self.labels).index("false-positive")
            clear_frames = []
            best_preds = []
            incorrect_best = []
            # print("track", track_id)

            for i, pred in enumerate(track_prediction.predictions):
                best = np.argsort(pred)
                if fp_i and best[-1] == fp_i:
                    continue

                clarity = pred[best[-1]] - pred[best[-2]]
                if clarity > MIN_CLARITY:
                    clear_frames.append((i, clarity))
                print(i, np.int16(np.around(100 * np.array(pred))))

                # if label_i:
                pred_percent = pred[best[-1]]
                if pred_percent >= MIN_PERCENT:
                    print(pred_percent, "saving", i)

                    best_preds.append((i, pred_percent))
                    # print(i, pred_percent, self.labels[best[-1]])
                # if not self.correct_prediction:
                #     if pred[best[-1]] > MIN_PERCENT:
                #         incorrect_best.append((i, pred[best[-1]]))

            clear_frames = sorted(
                clear_frames, reverse=True, key=lambda frame: frame[1]
            )
            # print("clearest frames", clear_frames)
            best_preds = sorted(best_preds, reverse=True, key=lambda frame: frame[1])
            # print("highest pred", best_preds)

            sorted(incorrect_best, reverse=True, key=lambda frame: frame[1])
            pred_frames = set()
            track_prediction.best_predictions = [f[0] for f in best_preds]
            track_prediction.clearest_frames = [f[0] for f in clear_frames]
            track_prediction.clearest_frames.sort()
            track_prediction.best_predictions.sort()
            print("track", track_id, track_prediction.best_predictions)
            # print(
            #     "predictions not clear",
            #     set(track_prediction.best_predictions)
            #     - set(track_prediction.clearest_frames),
            # )
            #
            # print(
            #     "clear not predictions",
            #     set(track_prediction.clearest_frames)
            #     - set(track_prediction.best_predictions),
            # )
            # pred_frames.update(f[0] for f in clear_frames)
            pred_frames.update(f[0] for f in best_preds)
            track_prediction.important_frames = list(pred_frames)
            # print(pred_frames)
            print(pred_frames)


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
        self.novelties = []
        self.uniform_prior = False
        self.class_best_score = None
        self.start_frame = start_frame
        self.last_frame_classified = start_frame
        self.num_frames_classified = 0
        self.keep_all = keep_all
        self.max_novelty = 0
        self.novelty_sum = 0
        self.important_frames = []
        self.clearest_frames = []
        self.best_predictions = []
        self.original = []

    def classified_clip(self, predictions, novelties, last_frame):
        self.last_frame_classified = last_frame
        self.num_frames_classified = len(predictions)
        self.predictions = predictions
        self.novelties = novelties
        self.class_best_score = np.max(self.predictions, axis=0)
        self.max_novelty = max(self.novelties)
        self.novelty_sum = sum(self.novelties)

    def classified_frame(self, frame_number, prediction, novelty=0, smooth=True):
        self.last_frame_classified = frame_number
        self.num_frames_classified += 1
        self.original.append(prediction)
        if smooth:
            prediction, novelty = self.smooth_prediction(prediction, novelty)
        if novelty:
            self.max_novelty = max(self.max_novelty, novelty)
            self.novelty_sum += novelty

        if self.keep_all:
            # if len(self.predictions) == 1:
            # print(self.predictions[0])
            # print(len(self.predictions), "adding prediction", prediction)
            self.predictions.append(prediction)
            # print(self.predictions[0])
            self.novelties.append(novelty)
            # if len(self.predictions) == 2:
            #     raise "ERROR"
        else:
            self.predictions = [prediction]
            self.novelties = [novelty]

        if self.class_best_score is None:
            self.class_best_score = prediction
        else:
            self.class_best_score = np.maximum(self.class_best_score, prediction)

    def smooth_prediction(self, prediction, novelty):
        prediction_smooth = 0.1
        smooth_novelty = None
        prev_prediction = None
        if len(self.predictions):
            prev_prediction = self.predictions[-1]
        num_labels = len(prediction)
        if prev_prediction is None:
            if UNIFORM_PRIOR:
                smooth_prediction = np.ones([num_labels]) * (1 / num_labels)
            else:
                smooth_prediction = prediction
            if novelty:
                smooth_novelty = 0.5
        else:
            smooth_prediction = (
                1 - prediction_smooth
            ) * prev_prediction + prediction_smooth * prediction
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
        important = False
        if frame_number in self.important_frames:
            # print("important_frame", frame_number)
            important = True
        if frame_number is None or frame_number >= len(self.novelties):
            return "({:.1f} {}){} {}".format(
                self.max_score * 100,
                labels[self.best_label_index],
                int(round(self.clarity, 2) * 100),
                important,
            )
        if self.predictions:
            return "({:.1f} {}) {} {}".format(
                self.score_at_time(frame_number) * 100,
                labels[self.label_at_time(frame_number)],
                int(round(self.clarity_at(frame_number), 2) * 100),
                important,
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

        if self.novelties is None:
            return None
        return self.novelties[n]

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

    def clarity_at(self, frame):
        pred = self.predictions[frame]
        best = np.argsort(pred)
        return pred[best[-1]] - pred[best[-2]]

    @property
    def clarity(self):
        """ The distance between our highest scoring class and second highest scoring class. """
        if self.class_best_score is None or len(self.class_best_score) < 2:
            return 0
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

    def label_at_time(self, frame_number, n=1):
        """ class label of nth best guess at a point in time."""

        if n is None:
            return None

        return int(np.argsort(self.predictions[frame_number])[-n])

    def score_at_time(self, frame_number, n=1):
        """ class label of nth best guess at a point in time."""
        if n is None:
            return None
        return float(sorted(self.predictions[frame_number])[-n])

    def print_prediction(self, labels):
        logging.info(
            "Track {} is {}".format(self.track_id, self.get_classified_footer(labels))
        )

    def best_gap(self, above=0.7):
        # frame and diff
        best = (0, 0)
        for i, frame in enumerate(self.predictions):
            a = np.argsort(frame)
            if frame[a[-1]] > above:
                gap = frame[a[-1]] - frame[a[-2]]
                if gap > best[1]:
                    best = [i, gap]
        return best

    def best_frame(self, label=None):
        preds = self.predictions
        if label:
            preds = np.array(preds)[:, label]

        max = np.amax(preds)
        result = np.where(preds == max)
        return max, result

    def guesses(self, labels):
        guesses = [
            "{} ({:.1f})".format(labels[self.label_index(i)], self.score(i) * 10)
            for i in range(1, 4)
            if self.score(i) > 0.5
        ]
        return guesses


@attr.s(slots=True)
class TrackResult:
    what = attr.ib()
    avg_novelty = attr.ib()
    max_novelty = attr.ib()
    confidence = attr.ib()
