"""
Script to classify animals within a CPTV video file.
"""

import argparse
import os
from trackextractor import TrackExtractor, Track
import trackclassifier
import numpy as np
import json
from ml_tools.tools import write_mpeg, load_colormap, convert_heat_to_img
import math
from PIL import Image, ImageDraw
import time
from multiprocessing import Value
from cptvfileprocessor import CPTVFileProcessor

DEFAULT_BASE_PATH = "c:\\cac"

class TrackPrediction():
    """
    Class to hold the information about the predicted class of a track.
    A list of predictions are transformed into 'confidences' by applying a bayesian update algorithim.
    The probabilities are also optionally weighted per example.  This can be useful if it is known that some
    of the examples are likely to be unhelpful (for example the track has minimal mass at that point).
    """

    def __init__(self, prediction_history, weights = None):

        # the highest confidence rating for each class
        self.class_best_confidence = [0.0]

        # history of raw probability for each class at every frame
        self.prediction_history = prediction_history.copy()

        # history of confidence over time for each class at every frame
        self.confidence_history = []

        self.weights = weights

        self._apply_bayesian_update()

    def label(self, n = 1):
        """ class label of nth best guess. """
        return int(np.argsort(self.class_best_confidence)[-n])

    def confidence(self, n = 1):
        """ class label of nth best guess. """
        return float(sorted(self.class_best_confidence)[-n])

    def label_at_time(self, frame_number, n = 1):
        """ class label of nth best guess at a point in time."""
        return int(np.argsort(self.confidence_history[frame_number])[-n])

    def confidence_at_time(self, frame_number, n = 1):
        """ class label of nth best guess at a point in time."""
        return float(sorted(self.confidence_history[frame_number])[-n])

    @property
    def clarity(self):
        """ The distance between our highest scoring class and second highest scoring class. """
        return self.confidence(1) - self.confidence(2)

    def save(self, filename):
        """ Saves prediction history to file. """
        save_file = {}
        save_file['label'] = int(self.label())
        save_file['confidence'] = float(self.confidence())
        save_file['clarity'] = float(self.clarity)
        save_file['predictions'] = list(self.class_best_confidence)
        save_file['prediction_history'] = [x.tolist() for x in self.prediction_history]
        save_file['confidence_history'] = [x.tolist() for x in self.confidence_history]
        if self.weights is not None: save_file['weights'] = [float(x) for x in self.weights]

        json.dump(save_file, open(filename, 'w'), indent=True)

    def _apply_bayesian_update(self):
        """
        Processes classifier predictions and builds a belief over time about the true class of the tracked object.
        """

        if len(self.prediction_history) == 0:
            return

        self.confidence_history = []

        # start with a uniform prior, could bias this if we want encourage certian classes.
        classes = len(self.prediction_history[0])
        confidence = np.asarray(np.float32([1/classes for _ in range(classes)]))

        # per class best confidence
        best = [float(0) for _ in range(classes)]

        # apply a bayesian update for each prediction.
        for i, prediction in enumerate(self.prediction_history):
            # too much confidence can make updates slow, even with very strong evidence.
            prediction = np.clip(prediction, 0.02, 0.98)

            # we scale down the predictions based on the weighting.  This will cause the algorithm to
            # give very low scores when the weight is low.
            if self.weights is not None:
                prediction *= self.weights[i]

            confidence = (confidence * prediction) / (confidence * prediction + (confidence * (1.0 - prediction)))

            self.confidence_history.append(confidence.copy())

            best = [float(max(best[i], confidence[i])) for i in range(classes)]

        self.class_best_confidence = best

class ClipClassifier(CPTVFileProcessor):
    """ Classifies tracks within CPTV files. """

    def __init__(self):
        """ Create an instance of a clip classifier"""

        super(ClipClassifier, self).__init__()

        # prediction record for each track
        self.track_prediction = {}

        # enables mpeg preview output
        self.enable_previews = False

    def identify_track(self, track: Track):
        """
        Runs through track identifying segments, and then returns it's prediction of what kind of animal this is.
        :param track: the track to identify.
        :return: TrackPrediction object
        """

        segment = trackclassifier.TrackSegment()
        predictions = []
        weights = []

        num_labels = len(self.classifier.classes)

        # preload the 27 frame buffer
        for i in range(min(27, track.frames)):
            frame = track.get_frame(i)
            segment.append_frame(frame)
            predictions.append(np.asarray(np.float32([1 / num_labels for _ in range(num_labels)])))
            weights.append(0.1)

        # go through making classifications at each frame
        # note: we should probably be doing this every 9 frames or so.
        for i in range(track.frames):
            frame = track.get_frame(i)
            segment.append_frame(frame)

            # segments with small mass are weighted less as we can assume the error is higher here.
            mass = np.float32(np.sum(segment.data[:, :, :, 4])) / 27

            # we use the squareroot here as the mass is in units squared.
            # this effectively means we are giving weight based on the diameter
            # of the object rather than the mass.
            weight = math.sqrt(np.clip(mass/30, 0.02, 1.0))

            predictions.append(self.classifier.predict(segment))

            weights.append(weight)

        return TrackPrediction(predictions, weights)

    def export_track_preview(self, filename, track):
        """
        Exports a clip showing tracking of one specific track with point in time predictions.
        """

        preview_scale = 4.0

        video_frames = []
        predictions = self.track_prediction[track].prediction_history

        for i in range(track.frames):
            # export a MPEG preview of the track
            frame = track.get_frame(i)
            draw_frame = np.float16(frame[:, :, 1])
            img = convert_heat_to_img(draw_frame, self.colormap, 0, 300)
            img = img.resize((int(img.width * preview_scale), int(img.height * preview_scale)), Image.NEAREST)

            # just in case we don't have as many predictions as frames.
            if i >= len(predictions):
                continue

            # draw predictions
            prediction = predictions[i]

            best_labels = np.argsort(-prediction)[:3]

            width, height = img.width, img.height

            for i, label in enumerate(best_labels):
                draw = ImageDraw.Draw(img)
                score = prediction[label]
                x = 10
                y = height - 100 + 10 + i * 30
                draw.rectangle([x, y + 16, width - 10, y + 26], outline=(0, 0, 0), fill=(0, 64, 0, 64))
                draw.rectangle([x, y + 16, 10 + score * (width - 20), y + 26], outline=(0, 0, 0),
                               fill=(64, 255, 0, 250))
                draw.text([x, y], self.classifier.classes[label])

            video_frames.append(np.asarray(img))

        write_mpeg(filename, video_frames)

    @property
    def classifier(self):
        """
        Returns a classifier object, which is creeated on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        global _classifier
        if '_classifer' not in globals():
            _classifier = trackclassifier.TrackClassifier('./models/Model 4e-0.833', disable_GPU=False)
        return _classifier

    def export_tracking_preview(self, filename, tracker:TrackExtractor):
        """
        Exports a clip showing the tracking and predictions for objects within the clip.
        """

        # increased resolution of video file.
        # videos look much better scaled up
        FRAME_SCALE = 3.0

        video_frames = []

        # write video
        for frame_number, filtered in enumerate(tracker.filtered_frames):

            frame = 3 * filtered + TrackExtractor.TEMPERATURE_MIN

            img = convert_heat_to_img(frame, self.colormap, tracker.TEMPERATURE_MIN, tracker.TEMPERATURE_MAX)
            img = img.resize((int(img.width * FRAME_SCALE), int(img.height * FRAME_SCALE)), Image.NEAREST)
            draw = ImageDraw.Draw(img)

            # look for any tracks that occur on this frame
            for id, track in enumerate(tracker.tracks):
                frame_offset = frame_number - track.first_frame
                if 0 < frame_offset < len(track.bounds_history)-1:

                    # display the track
                    rect = track.bounds_history[frame_offset]
                    rect_points = [int(p * FRAME_SCALE) for p in [rect.left, rect.top, rect.right, rect.top, rect.right, rect.bottom, rect.left, rect.bottom, rect.left, rect.top]]
                    draw.line(rect_points, (255, 64, 32))

                    # display prediction information
                    x = (rect.left) * FRAME_SCALE
                    y = (rect.bottom if rect.bottom < (img.height / FRAME_SCALE) - 8 else rect.top-10) * FRAME_SCALE

                    # get a string indicating our current prediction.
                    if frame_offset < 27:
                        # not enough information yet...
                        prediction_string = '...'
                    if track not in self.track_prediction :
                        # no information for this track just ignore
                        prediction_string = ''
                    else:
                        prediction = self.track_prediction[track]
                        label = self.classifier.classes[prediction.label_at_time(frame_offset)]
                        confidence = prediction.confidence_at_time(frame_offset)
                        if confidence >= 0.7:
                            prediction_string = "{} {:.1f}%".format(label, confidence * 100)
                        else:
                            prediction_string = "<not sure> {:.1f}%".format(confidence * 100)

                    draw.text((x,y),prediction_string)

            video_frames.append(np.asarray(img))

            frame_number += 1

            # we store the entire video in memory so we need to cap the frame count at some point.
            if frame_number > 9 * 60 * 10:
                break

        write_mpeg(filename, video_frames)

    def needs_processing(self, filename):
        """
        Returns True if this file needs to be processed, false otherwise.
        :param filename: the full path and filename of the cptv file in question.
        :return: returns true if file should be processed, false otherwise
        """

        # look to see of the destination file already exists.
        base_name = os.path.splitext(os.path.join(self.output_folder, os.path.basename(filename)))[0]
        meta_filename = base_name + '.txt'

        # if no stats file exists we haven't processed file, so reprocess

        # otherwise check what needs to be done.
        if self.overwrite_mode == self.OM_ALL:
            return True
        elif self.overwrite_mode == self.OM_NONE:
            return not os.path.exists(meta_filename)
        else:
            raise Exception("Overwrite mode {} not supported.".format(self.overwrite_mode))

    def process_file(self, filename):
        """
        Process a file extracting tracks and identifying them.
        :param filename: filename to process
        :param enable_preview: if true an MPEG preview file is created.
        """

        if not os.path.exists(filename):
            raise Exception("File {} not found.".format(filename))

        start = time.time()

        # extract tracks from file
        tracker = TrackExtractor(filename)

        tracker.reduced_quality_optical_flow = True
        tracker.colormap = load_colormap("custom_colormap.dat")

        tracker.extract()

        base_name = os.path.splitext(os.path.join(self.output_folder, os.path.basename(filename)))[0]

        mpeg_filename = base_name + '.mp4'
        meta_filename = base_name + '.txt'
        track_mpeg_filename = base_name + "-{} {} {}.mpg"
        track_meta_filename = base_name + "-{}.txt"

        print(os.path.basename(filename)+":")

        # identify each track
        for i, track in enumerate(tracker.tracks):

            prediction = self.identify_track(track)

            self.track_prediction[track] = prediction

            prediction.save(track_meta_filename.format(i+1))

            if prediction.confidence() > 0.5:
                first_guess = "{} {:.1f} (clarity {:.1f})".format(
                    self.classifier.classes[prediction.label()], prediction.confidence() * 100, prediction.clarity * 100)
            else:
                first_guess = "[nothing]"

            if prediction.confidence(2) > 0.5:
                second_guess = "[second guess - {} {:.1f}]".format(
                    self.classifier.classes[prediction.label(2)], prediction.confidence(2)*100)
            else:
                second_guess = ""

            print(" - [{}/{}] prediction: {} {}".format(str(i + 1), str(len(tracker.tracks) + 1), first_guess, second_guess))

            if self.enable_previews:
                self.export_track_preview(track_mpeg_filename.format(i + 1, first_guess, second_guess), track)

        if self.enable_previews:
            self.export_tracking_preview(mpeg_filename, tracker)

        # record results in text file.
        f = open(meta_filename,'w')
        save_file = {}
        save_file['tags'] = set([self.classifier.classes[prediction.label()] for prediction in self.track_prediction.values()])
        save_file['max_confidence'] = max([0.0]+[prediction.confidence() for prediction in self.track_prediction.values()])

        print("Took {:.1f}s".format(time.time() - start))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('source',
                        help='a CPTV file to process, or a folder name')

    parser.add_argument('-p', '--enable-preview', action='count', help='Enables preview MPEG files (can be slow)')
    parser.add_argument('-v', '--verbose', action='count', help='Display additional information.')
    parser.add_argument('-w', '--workers', default='0',help='Number of worker threads to use.  0 disables worker pool and forces a single thread.')
    parser.add_argument('-f', '--force-overwrite', default='none',help='Overwrite mode.  Options are all, old, or none.')
    parser.add_argument('-o', '--output-folder', default=os.path.join(DEFAULT_BASE_PATH, "autotagged"),help='Folder to output tracks to')
    parser.add_argument('-s', '--source-folder', default=os.path.join(DEFAULT_BASE_PATH, "clips"),help='Source folder root with class folders containing CPTV files')
    parser.add_argument('-c', '--color-map', default="custom_colormap.dat",help='Colormap to use when exporting MPEG files')

    args = parser.parse_args()

    clip_classifier = ClipClassifier()
    clip_classifier.enable_previews = args.enable_preview
    clip_classifier.output_folder = args.output_folder
    clip_classifier.source_folder = args.source_folder

    # apply the colormap
    clip_classifier.colormap = load_colormap(args.color_map)

    clip_classifier.workers_threads = int(args.workers)
    if clip_classifier.workers_threads >= 1:
        print("Using {0} worker threads".format(clip_classifier.workers_threads))

    # set overwrite mode
    if args.force_overwrite.lower() not in ['all', 'old', 'none']:
        raise Exception("Valid overwrite modes are all, old, or none.")
    clip_classifier.overwrite_mode = args.force_overwrite.lower()

    # set verbose
    clip_classifier.verbose = args.verbose

    if os.path.splitext(args.source)[-1].lower() == '.cptv':
        clip_classifier.process_file(os.path.join(args.source_folder, args.source))
    else:
        clip_classifier.process_folder(os.path.join(args.source_folder, args.source))


if __name__ == "__main__":
    # execute only if run as a script
    main()