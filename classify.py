"""
Script to classify animals within a CPTV video file.
"""

import argparse
import os
import trackextractor
import trackclassifier
import numpy as np
import json
import ml_tools.tools
import math
import PIL as pillow
import time

class TrackPrediction():
    """ Class to hold the information about the predicted class of a track. """

    def __init__(self, prediction_history, weights = None):

        # the highest confidence rating for each class
        self.class_best_confidence = [0.0]

        # history of probability for each class at various intervals
        self.prediction_history = prediction_history.copy()

        self.weights = weights

        self._apply_bayesian_update()

    def label(self, n = 1):
        """ class label of nth best guess. """
        return int(np.argsort(self.class_best_confidence)[-n])

    def confidence(self, n = 1):
        """ class label of nth best guess. """
        return float(sorted(self.class_best_confidence)[-n])

    @property
    def clarity(self):
        """ The distance between our highest scoring class and second highest scoring class. """
        return self.confidence(1) - self.confidence(2)

    def save(self, filename):
        """ Saves prediction history to file. """
        save_file = {}
        save_file['label'] = self.label()
        save_file['confidence'] = self.confidence()
        save_file['clarity'] = self.clarity
        save_file['predictions'] = self.class_best_confidence
        save_file['history'] = [x.tolist() for x in self.prediction_history]

        if self.weights is not None: save_file['weights'] = [float(x) for x in self.weights]
        json.dump(save_file, open(filename, 'w'), indent=True)

    def _apply_bayesian_update(self):
        """
        Processes classifier predictions and builds a belief over time about the true class of the tracked object.
        :param predictions: list of predictions for each class at each time-step.
        :param weights: if given changes in confidence will be multiplied by these weights.
        :return: prediction of track class
        """

        if len(self.prediction_history) == 0:
            return

        # start with a uniform prior, could bias this if we want encourage certian classes.
        classes = len(self.prediction_history[0])
        confidence = np.asarray([1/classes for _ in range(classes)])

        # per class best confidence
        best = [0 for _ in range(classes)]
        best_result = None

        # apply a bayesian update for each prediction.
        for i, prediction in enumerate(self.prediction_history):
            # too much confidence can make updates slow, even with very strong evidence.
            prediction = np.clip(prediction, 0.02, 0.98)

            # we scale down the predictions based on the weighting.  This will cause the algorithm to
            # give very low scores when the weight is low.
            if self.weights is not None:
                prediction *= self.weights[i]

            confidence = (confidence * prediction) / (confidence * prediction + (confidence * (1.0 - prediction)))

            best = [max(best[i], confidence[i]) for i in range(classes)]

        self.class_best_confidence = best

def identify_track(classifier: trackclassifier.TrackClassifier, track: trackextractor.Track):
    """
    Runs through track identifying segments, and then returns it's prediction of what kind of animial this is.
    :param classifier: classifier to use for classification
    :param track: the track to identify.
    :return: TrackPrediction object
    """

    segment = trackclassifier.TrackSegment()
    predictions = []
    weights = []

    # not enough frames for prediction
    if track.frames < 27:
        return (0, 0)

    # go through classifying each segment
    for i in range(track.frames):

        frame = track.get_frame(i)
        segment.append_frame(frame)

        if i >= 26:
            # segments with small mass are weighted less as we can assume the error is higher here.
            mass = np.float32(np.sum(segment.data[:, :, :, 4])) / 27

            # we use the squareroot here as the mass is in units squared.
            # this effectively means we are giving weight based on the diameter
            # of the object rather than the mass.
            weight = math.sqrt(np.clip(mass/30, 0.02, 1.0))

            predictions.append(classifier.predict(segment))
            weights.append(weight)

    return TrackPrediction(predictions, weights)

def paint_predictions(img, frame_index, predictions, label_names):

    # offset prediction to be at the end of the 27 frame window.
    frame_index -= 27

    # in the frames before don't paint anything.
    if frame_index < 0 or frame_index >= len(predictions):
        return

    prediction = predictions[frame_index]

    best_labels = np.argsort(-prediction)[:3]

    width, height = img.width, img.height

    for i, label in enumerate(best_labels):
        draw = pillow.ImageDraw.Draw(img)
        score = prediction[label]
        x = 10
        y = height - 100 + 10 + i*30
        draw.rectangle([x, y+16, width - 10, y + 26], outline=(0, 0, 0), fill=(0, 64, 0, 64))
        draw.rectangle([x, y+16, 10 + score * (width-20), y + 26], outline=(0, 0, 0), fill=(64, 255, 0, 250))
        draw.text([x,y],label_names[label])

def process_file(filename, enable_preview=True):
    """
    Process a file extracting tracks and identifying them.
    :param filename: filename to process
    :param enable_preview: if true an MPEG preview file is created.
    """

    if not os.path.exists(filename):
        raise Exception("File {} not found.".format(filename))

    # extract tracks from file
    tracker = trackextractor.TrackExtractor(filename)
    tracker.reduced_quality_optical_flow = True
    tracker.colormap = ml_tools.tools.load_colormap("custom_colormap.dat")

    tracker.extract()

    print(os.path.basename(filename)+":")

    # identify each track
    for i, track in enumerate(tracker.tracks):
        prediction = identify_track(classifier, track)
        prediction.save(os.path.join('temp',os.path.basename(filename)+"-"+str(i+1)+".txt"))

        if prediction.confidence() > 0.5:
            first_guess = "{} {:.1f} (clarity {:.1f})".format(
                classifier.classes[prediction.label()], prediction.confidence() * 100, prediction.clarity * 100)
        else:
            first_guess = "[nothing]"

        if prediction.confidence(2) > 0.5:
            second_guess = "[second guess - {} {:.1f}]".format(
                classifier.classes[prediction.label(2)], prediction.confidence(2)*100)
        else:
            second_guess = ""

        print(" - [{}/{}] prediction: {} {}".format(str(i + 1), str(len(tracker.tracks) + 1), first_guess, second_guess))

        export_filename = os.path.join('temp',os.path.basename(filename)) + '-'+str(i+1)+" " + (first_guess + " " + second_guess).strip() + ".mp4"
        tracker.display_track(track, export_filename, preview_scale=4.0, display_hook=paint_predictions, predictions=prediction.prediction_history, label_names = classifier.classes)

    # export a preview file showing classification over time.
    """
    if enable_preview:
        # todo write a custom display for this
        tracker.display(os.path.join('temp',os.path.basename(filename))+'.mpg')
    """


def process_folder(folder_path):
    """ Classifies all CPTV files in given folder. """
    print("Processing",folder_path)
    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        if os.path.isfile(full_path) and os.path.splitext(full_path )[1].lower() == '.cptv':
            start = time.time()
            process_file(full_path)
            print("took {:.2f}s".format(time.time() - start))


def main():

    # note, this really shouldn't be a global, better to make a class as this point and put all this stuff together.
    global classifier
    classifier = trackclassifier.TrackClassifier('./models/Model 4e-0.833', disable_GPU=False)

    parser = argparse.ArgumentParser()

    parser.add_argument('source',
                        help='a CPTV file to process, or a folder name')

    parser.add_argument('-p', '--enable-preview', action='count', help='Enables preview MPEG files (can be slow)')
    parser.add_argument('-v', '--verbose', action='count', help='Display additional information.')

    args = parser.parse_args()

    if os.path.splitext((args.source).lower()) == '.cptv':
        process_file(args.source)
    else:
        process_folder(args.source)


if __name__ == "__main__":
    # execute only if run as a script
    main()