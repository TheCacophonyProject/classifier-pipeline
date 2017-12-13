"""
Script to classify animals within a CPTV video file.
"""

import argparse
import os
import trackextractor
import trackclassifier
import numpy as np

class TrackPrediction():
    """ Class to hold the information about the predicted class of a track. """

    def __init__(self):

        # class label, integer.
        self.label = None

        # confidence of prediction, from 0 to 1.
        self.confidence = None

        # history of probability for each class at various intervals
        self.prob_history = []


def bayesian_update(predictions):
    """
    Processes classifier predictions and builds a beleif over time about the true class of the tracked object.
    :param predictions: list of predictions for each class at each timestep.
    :return: prediction of track class
    """

    # start with a uniform prior, could bias this if we want encourage certian classes.
    classes = len(predictions[0])
    confidence = [1/classes for _ in range(classes)]

    # apply a bayesian update for each prediction.
    for prediction in predictions:
        confidence = (confidence * prediction) / (confidence * prediction + (confidence * (1.0 - prediction)))
        # too much confidence can make updates slow, even with very strong evidence.
        confidence = np.clip(confidence, 0.05, 0.95)

        predicted_class = np.argmax(confidence)
        predicted_confidence = np.max(confidence) * 100
        print(" -", predicted_class, predicted_confidence)

    return confidence

def identify_track(classifier: trackclassifier.TrackClassifier, track: trackextractor.Track):
    """
    Runs through track identifying segments, and then returns it's prediction of what kind of animial this is.
    :param classifier: classifier to use for classification
    :param track: the track to identify.
    :return: TrackPrediction object
    """

    segment = trackclassifier.TrackSegment()
    predictions = []

    # todo: frames should be normalised... but I don't have normalisation constants yet... hmm maybe just normalise the
    # whole track??  Also, perhaps the batch norm helps fix this, cause it seems to actually be working...

    # go through classifying each segment
    for i in range(track.frames):
        frame = track.get_frame(i)
        segment.append_frame(frame)
        predictions.append(classifier.predict(segment))

    final_prediction = bayesian_update(predictions)

    predicted_class = np.argmax(final_prediction)
    predicted_confidence = np.max(final_prediction) * 100

    print("Prediction: {0} {1:.1f}".format(classifier.classes[predicted_class], predicted_confidence))


def process_file(filename, enable_preview=False):
    """
    Process a file extracting tracks and identifying them.
    :param filename: filename to process
    :param enable_preview: if true an MPEG preview file is created.
    """

    if not os.path.exists(filename):
        raise Exception("File {} not found.".format(filename))

    # extract tracks from file
    tracker = trackextractor.TrackExtractor(filename)

    tracker.extract()

    classifier = trackclassifier.TrackClassifier('./models/Model 4e-0.833')

    # identify each track
    track_info = []
    for track in tracker.tracks:
        identify_track(classifier, track)

    # export a preview file showing classification over time.
    if enable_preview:
        # NIY
        print("MPEG preview output has not been implemented yet.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('source',
                        help='CPTV file to process')

    parser.add_argument('-p', '--enable-preview', action='count', help='Enables preview MPEG files (can be slow)')
    parser.add_argument('-v', '--verbose', action='count', help='Display additional information.')

    args = parser.parse_args()

    process_file(args.source, args.enable_preview)


if __name__ == "__main__":
    # execute only if run as a script
    main()