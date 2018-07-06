"""
Script to classify animals within a CPTV video file.
"""

from typing import Dict

import argparse
import json
import os
import time
import logging
import sys
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from ml_tools import tools
from ml_tools.model import Model
from ml_tools.dataset import Preprocessor
from ml_tools.cptvfileprocessor import CPTVFileProcessor
from ml_tools.mpeg_creator import MPEGCreator
from ml_tools.trackextractor import TrackExtractor, Track, Region


DEFAULT_BASE_PATH = "c:/cac"
HERE = os.path.dirname(__file__)
RESOURCES_PATH = os.path.join(HERE, "resources")
MODEL_NAME = "model_lq-0.929"

# folders that are not processed when run with 'all'
IGNORE_FOLDERS = ['untagged','cat','dog','insect','unidentified','rabbit','hard','multi','moving','mouse',
                  'bird-kiwi','for_grant']


def resource_path(name):
    return os.path.join(RESOURCES_PATH, name)

# We store some cached shared objects as globals as they can not be passed around processes, and therefore would
# break the worker threads system.  Instead we load them on demand and store them in each processors global space.
_classifier = None
_classifier_font = None
_classifier_font_title = None

class TrackPrediction:
    """
    Class to hold the information about the predicted class of a track.
    Predictions are recorded for every frame, and methods provided for extracting the final predicted class of the
    track.
    """

    def __init__(self, prediction_history):
        """
        Setup track prediction with given prediction history
        :param prediction_history: list of predictions for each frame of this track.
        """
        self.prediction_history = prediction_history.copy()
        self.class_best_score = np.max(np.float32(prediction_history), axis=0).tolist()

    def label(self, n = 1):
        """ class label of nth best guess. """
        return int(np.argsort(self.class_best_score)[-n])

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


class ClipClassifier(CPTVFileProcessor):
    """ Classifies tracks within CPTV files. """

    # skips every nth frame.  Speeds things up a little, but reduces prediction quality.
    FRAME_SKIP = 1

    def __init__(self):
        """ Create an instance of a clip classifier"""

        super(ClipClassifier, self).__init__()

        # prediction record for each track
        self.track_prediction: Dict[Track, TrackPrediction] = {}

        # mpeg preview output
        self.enable_previews = False
        self.enable_side_by_side = False

        self.enable_gpu = True

        self.start_date = None
        self.end_date = None

        self.model_path = None

        # uses a higher quality version of the optical flow algorithm.
        self.high_quality_optical_flow = False

        # enables exports detailed information for each track.  If preview mode is enabled also enables track previews.
        self.enable_per_track_information = False

        # includes both original, and predicted tag in filename
        self.include_prediction_in_filename = False

        # writes metadata to standard out instead of a file.
        self.write_meta_to_stdout = False

    @property
    def font(self):
        """ gets default font. """
        global _classifier_font
        if not _classifier_font: _classifier_font = ImageFont.truetype(resource_path("Ubuntu-R.ttf"), 12)
        return _classifier_font

    @property
    def font_title(self):
        """ gets default title font. """
        global _classifier_font_title
        if not _classifier_font_title: _classifier_font_title = ImageFont.truetype(resource_path("Ubuntu-B.ttf"), 14)
        return _classifier_font_title

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

        num_labels = len(self.classifier.labels)
        prediction_smooth = 0.1

        smooth_prediction = None

        fp_index = self.classifier.labels.index('false-positive')

        # go through making clas sifications at each frame
        # note: we should probably be doing this every 9 frames or so.
        state = None
        for i in range(len(track)):

            # note: would be much better for the tracker to store the thermal references as it goes.
            thermal_reference = np.median(tracker.frame_buffer.thermal[track.start_frame + i])

            frame = tracker.get_track_channels(track, i)
            if i % self.FRAME_SKIP == 0:

                frame = Preprocessor.apply([frame], [thermal_reference])[0]

                prediction, state = self.classifier.classify_frame(frame, state)

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

            else:
                # just continue prediction and state along.
                pass

            if smooth_prediction is None:
                if UNIFORM_PRIOR:
                    smooth_prediction = np.ones([num_labels]) * (1 / num_labels)
                else:
                    smooth_prediction = prediction
            else:
                smooth_prediction = (1-prediction_smooth) * smooth_prediction + prediction_smooth * prediction
            predictions.append(smooth_prediction)

        return TrackPrediction(predictions)

    @property
    def classifier(self):
        """
        Returns a classifier object, which is created on demand.
        This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
        """
        global _classifier
        if _classifier is None:
            t0 = datetime.now()
            logging.info("classifier loading")
            _classifier = Model(tools.get_session(disable_gpu=not self.enable_gpu))
            _classifier.load(self.model_path)
            logging.info("classifier loaded ({})".format(datetime.now() - t0))

        return _classifier

    def get_clip_prediction(self):
        """ Returns list of class predictions for all tracks in this clip. """

        class_best_score = [0 for _ in range(len(self.classifier.labels))]

        # keep track of our highest confidence over every track for each class
        for track, prediction in self.track_prediction.items():
            for i in range(len(self.classifier.labels)):
                class_best_score[i] = max(class_best_score[i], prediction.class_best_score[i])

        results = []
        for n in range(1, 1+len(self.classifier.labels)):
            nth_label = int(np.argsort(class_best_score)[-n])
            nth_score = float(np.sort(class_best_score)[-n])
            results.append((self.classifier.labels[nth_label], nth_score))

        return results

    def fit_to_screen(self, rect:Region, screen_bounds:Region):
        """ Modifies rect so that rect is visible within bounds. """
        if rect.left < screen_bounds.left:
            rect.x = screen_bounds.left
        if rect.top < screen_bounds.top:
            rect.y = screen_bounds.top

        if rect.right > screen_bounds.right:
            rect.x = screen_bounds.right - rect.width

        if rect.bottom > screen_bounds.bottom:
            rect.y = screen_bounds.bottom - rect.height

    def export_clip_preview(self, filename, tracker:TrackExtractor):
        """
        Exports a clip showing the tracking and predictions for objects within the clip.
        """

        # increased resolution of video file.
        # videos look much better scaled up
        FRAME_SCALE = 4.0

        NORMALISATION_SMOOTH = 0.95

        # amount pad at ends of thermal range
        HEAD_ROOM = 25

        auto_min = np.min(tracker.frame_buffer.thermal[0])
        auto_max = np.max(tracker.frame_buffer.thermal[0])

        # setting quality to 30 gives files approximately the same size as the original CPTV MPEG previews
        # (but they look quite compressed)
        mpeg = MPEGCreator(filename)

        for frame_number, thermal in enumerate(tracker.frame_buffer.thermal):

            thermal_min = np.min(thermal)
            thermal_max = np.max(thermal)

            auto_min = NORMALISATION_SMOOTH * auto_min + (1 - NORMALISATION_SMOOTH) * (thermal_min-HEAD_ROOM)
            auto_max = NORMALISATION_SMOOTH * auto_max + (1 - NORMALISATION_SMOOTH) * (thermal_max+HEAD_ROOM)

            # sometimes we get an extreme value that throws off the autonormalisation, so if there are values outside
            # of the expected range just instantly switch levels
            if thermal_min < auto_min or thermal_max > auto_max:
                auto_min = thermal_min
                auto_max = thermal_max

            thermal_image = tools.convert_heat_to_img(thermal, self.colormap, auto_min, auto_max)
            thermal_image = thermal_image.resize((int(thermal_image.width * FRAME_SCALE), int(thermal_image.height * FRAME_SCALE)), Image.BILINEAR)

            if tracker.frame_buffer.filtered:
                if self.enable_side_by_side:
                    # put thermal & tracking images side by side
                    tracking_image = self.export_tracking_frame(tracker, frame_number, FRAME_SCALE)
                    side_by_side_image = Image.new('RGB', (tracking_image.width * 2, tracking_image.height))
                    side_by_side_image.paste(thermal_image, (0, 0))
                    side_by_side_image.paste(tracking_image, (tracking_image.width, 0))
                    mpeg.next_frame(np.asarray(side_by_side_image))
                else:
                    # overlay track rectanges on original thermal image
                    thermal_image = self.draw_track_rectangles(tracker, frame_number, FRAME_SCALE, thermal_image)
                    mpeg.next_frame(np.asarray(thermal_image))

            else:
                # no filtered frames available (clip too hot or
                # background moving?) so just output the original
                # frame without the tracking frame.
                mpeg.next_frame(np.asarray(thermal_image))

            # we store the entire video in memory so we need to cap the frame count at some point.
            if frame_number > 9 * 60 * 10:
                break

        mpeg.close()

    def export_tracking_frame(self, tracker: TrackExtractor, frame_number:int, frame_scale:float):

        mask = tracker.frame_buffer.mask[frame_number]

        filtered = tracker.frame_buffer.filtered[frame_number]
        tracking_image = tools.convert_heat_to_img(filtered / 200, self.colormap, temp_min=0, temp_max=1)

        tracking_image = tracking_image.resize((int(tracking_image.width * frame_scale), int(tracking_image.height * frame_scale)), Image.NEAREST)

        return self.draw_track_rectangles(tracker, frame_number, frame_scale, tracking_image)

    def draw_track_rectangles(self, tracker, frame_number, frame_scale, image):
        draw = ImageDraw.Draw(image)

        # look for any tracks that occur on this frame
        for id, track in enumerate(tracker.tracks):

            prediction = self.track_prediction[track]

            # find a track description, which is the final guess of what this class is.
            guesses = ["{} ({:.1f})".format(
                self.classifier.labels[prediction.label(i)], prediction.score(i) * 10) for i in range(1, 4)
                if prediction.score(i) > 0.5]

            track_description = "\n".join(guesses)
            track_description.strip()

            frame_offset = frame_number - track.start_frame
            if 0 < frame_offset < len(track.bounds_history) - 1:
                # display the track
                rect = track.bounds_history[frame_offset]
                rect_points = [int(p * frame_scale) for p in [rect.left, rect.top, rect.right, rect.top, rect.right,
                                                              rect.bottom, rect.left, rect.bottom, rect.left,
                                                              rect.top]]
                draw.line(rect_points, (255, 64, 32))

                if track not in self.track_prediction:
                    # no information for this track just ignore
                    current_prediction_string = ''
                else:
                    label = self.classifier.labels[prediction.label_at_time(frame_offset)]
                    score = prediction.score_at_time(frame_offset)
                    if score >= 0.7:
                        prediction_format = "({:.1f} {})"
                    else:
                        prediction_format = "({:.1f} {})?"
                    current_prediction_string = prediction_format.format(score * 10, label)

                header_size = self.font_title.getsize(track_description)
                footer_size = self.font.getsize(current_prediction_string)

                # figure out where to draw everything
                header_rect = Region(rect.left * frame_scale, rect.top * frame_scale - header_size[1], header_size[0], header_size[1])
                footer_center = ((rect.width * frame_scale) - footer_size[0]) / 2
                footer_rect = Region(rect.left * frame_scale + footer_center, rect.bottom * frame_scale, footer_size[0], footer_size[1])

                screen_bounds = Region(0, 0, image.width, image.height)

                self.fit_to_screen(header_rect, screen_bounds)
                self.fit_to_screen(footer_rect, screen_bounds)

                draw.text((header_rect.x, header_rect.y), track_description, font=self.font_title)
                draw.text((footer_rect.x, footer_rect.y), current_prediction_string, font=self.font)

        return image

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

        # otherwise check what needs to be done.
        if self.overwrite_mode == self.OM_ALL:
            return True
        elif self.overwrite_mode == self.OM_NONE:
            return not os.path.exists(meta_filename)
        else:
            raise Exception("Overwrite mode {} not supported.".format(self.overwrite_mode))

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
        """ Returns the base path and filename for an output filename from an input filename. """
        if self.include_prediction_in_filename:
            meta_data = self.get_meta_data(input_filename)
            tag_part = '[' + (meta_data["primary_tag"] if meta_data else "none") + '] '
        else:
            tag_part = ''
        return os.path.splitext(os.path.join(self.output_folder, tag_part + os.path.basename(input_filename)))[0]

    def process_all(self, root):
        for root, folders, files in os.walk(root):
            for folder in folders:
                if folder not in IGNORE_FOLDERS:
                    self.process_folder(os.path.join(root,folder), tag=folder.lower())

    def process_file(self, filename, **kwargs):
        """
        Process a file extracting tracks and identifying them.
        :param filename: filename to process
        :param enable_preview: if true an MPEG preview file is created.
        """

        if not os.path.exists(filename):
            raise Exception("File {} not found.".format(filename))

        start = time.time()

        # extract tracks from file
        tracker = TrackExtractor()
        tracker.WINDOW_SIZE = 48
        tracker.MAX_REGION_SIZE = 256 # enable larger regions
        tracker.load(filename)

        # turn up sensitivity on tracking so we can catch more animals.  The classifier will sort out the false
        # positives.
        tracker.track_min_duration = 0.0
        tracker.track_min_offset = 0.0
        tracker.track_min_delta = 1.0
        tracker.track_min_mass = 0.0

        tracker.min_threshold = 15
        tracker.include_cropped_regions = True # we can handle some cropping during classification.

        tracker.high_quality_optical_flow = self.high_quality_optical_flow

        tracker.extract_tracks()

        if len(tracker.tracks) > 10:
            logging.warning(" -warning, found too many tracks.  Using {} of {}".format(10, len(tracker.tracks)))
            tracker.tracks = tracker.tracks[:10]

        if len(tracker.tracks) > 0:
            # optical flow is not generated by default, if we have atleast one track we will need to generate it here.
            if not tracker.frame_buffer.has_flow:
                tracker.frame_buffer.generate_flow(tracker.opt_flow)

        base_name = self.get_base_name(filename)
        destination_folder = os.path.dirname(base_name)

        if not os.path.exists(destination_folder):
            logging.info("Creating folder {}".format(destination_folder))
            os.makedirs(destination_folder)

        if self.include_prediction_in_filename:
            mpeg_filename = base_name + "{}" + '.mp4'
        else:
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

        if self.enable_previews:
            prediction_string = ""
            for label, score in self.get_clip_prediction():
                if score > 0.5:
                    prediction_string = prediction_string + " {} {:.1f}".format(label, score * 10)
            self.export_clip_preview(mpeg_filename.format(prediction_string), tracker)

        # record results in text file.
        save_file = {}
        save_file['source'] = filename
        save_file['start_time'] = tracker.video_start_time.isoformat()
        save_file['end_time'] = (tracker.video_start_time + timedelta(seconds=len(tracker.frame_buffer.thermal) / 9.0)).isoformat()

        # read in original metadata
        meta_data = self.get_meta_data(filename)

        if meta_data:
            save_file['camera'] = meta_data['Device']['devicename']
            save_file['cptv_meta'] = meta_data
            save_file['original_tag'] = meta_data['primary_tag']
        save_file['tracks'] = []
        for track, prediction in self.track_prediction.items():
            track_info = {}
            save_file['tracks'].append(track_info)
            track_info['start_time'] = track.start_time.isoformat()
            track_info['end_time'] = track.end_time.isoformat()
            track_info['label'] = self.classifier.labels[prediction.label()]
            track_info['confidence'] = prediction.score()
            track_info['clarity'] = prediction.clarity
            track_info['class_confidence'] = prediction.class_best_score

        if self.write_meta_to_stdout:
            output = json.dumps(save_file, indent=4, cls=tools.CustomJSONEncoder)
            print(output)
        else:
            f = open(meta_filename, 'w')
            json.dump(save_file, f, indent=4, cls=tools.CustomJSONEncoder)

        ms_per_frame = (time.time() - start) * 1000 / max(1, len(tracker.frame_buffer.thermal))
        if self.verbose:
            logging.info("Took {:.1f}ms per frame".format(ms_per_frame))


def log_to_stdout():
    """ Outputs all log entries to standard out. """
    # taken from https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('source',help='a CPTV file to process, or a folder name, or "all" for all files within subdirectories of source folder.')

    parser.add_argument('-p', '--enable-preview', default=False, action='store_true', help='Enables preview MPEG files (can be slow)')
    parser.add_argument('-b', '--side-by-side', default=False, action='store_true', help='Output processed footage next to original output in preview MPEG')
    parser.add_argument('-q', '--high-quality-optical-flow', default=False, action='store_true', help='Enabled higher quality optical flow (slow)')
    parser.add_argument('-v', '--verbose', default=0, action='count', help='Display additional information.')
    parser.add_argument('-w', '--workers', default=0, help='Number of worker threads to use.  0 disables worker pool and forces a single thread.')
    parser.add_argument('-f', '--force-overwrite', default='none',help='Overwrite mode.  Options are all, old, or none.')
    parser.add_argument('-o', '--output-folder', default=os.path.join(DEFAULT_BASE_PATH, "runs"),help='Folder to output tracks to')
    parser.add_argument('-s', '--source-folder', default=os.path.join(DEFAULT_BASE_PATH, "clips"),help='Source folder root with class folders containing CPTV files')

    parser.add_argument('-m', '--model', default=os.path.join(HERE, "models", MODEL_NAME), help='Model to use for classification')
    parser.add_argument('-i', '--include-prediction-in-filename', default=False, action='store_true', help='Adds class scores to output files')
    parser.add_argument('--meta-to-stdout', default=False, action='store_true', help='Writes metadata to standard out instead of a file')

    parser.add_argument('--start-date', help='Only clips on or after this day will be processed (format YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Only clips on or before this day will be processed (format YYYY-MM-DD)')
    parser.add_argument('--disable-gpu', default=False, action='store_true', help='Disables GPU acclelerated classification')

    args = parser.parse_args()

    if not args.meta_to_stdout:
        log_to_stdout()

    clip_classifier = ClipClassifier()
    clip_classifier.enable_gpu = not args.disable_gpu
    clip_classifier.enable_previews = args.enable_preview
    clip_classifier.enable_side_by_side = args.side_by_side
    clip_classifier.output_folder = args.output_folder
    clip_classifier.source_folder = args.source_folder
    clip_classifier.model_path = args.model
    clip_classifier.high_quality_optical_flow = args.high_quality_optical_flow
    clip_classifier.include_prediction_in_filename = args.include_prediction_in_filename
    clip_classifier.write_meta_to_stdout = args.meta_to_stdout

    if clip_classifier.high_quality_optical_flow:
        logging.info("High quality optical flow enabled.")

    if not clip_classifier.enable_gpu:
        logging.info("GPU mode disabled.")

    if not os.path.exists(args.model+".meta"):
        logging.error("No model found named '{}'.".format(args.model+".meta"))
        exit(13)

    if args.start_date:
        clip_classifier.start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    if args.end_date:
        clip_classifier.end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # just fetch the classifier now so it doesn't impact the benchmarking on the first clip analysed.
    _ = clip_classifier.classifier

    # apply the colormap
    clip_classifier.colormap = tools.load_colormap(resource_path("custom_colormap.dat"))

    clip_classifier.workers_threads = int(args.workers)
    if clip_classifier.workers_threads >= 1:
        logging.info("Using {0} worker threads".format(clip_classifier.workers_threads))

    # set overwrite mode
    if args.force_overwrite.lower() not in ['all', 'old', 'none']:
        raise Exception("Valid overwrite modes are all, old, or none.")
    clip_classifier.overwrite_mode = args.force_overwrite.lower()

    # set verbose
    clip_classifier.verbose = args.verbose

    if args.source == "all":
        clip_classifier.process_all(args.source_folder)
    elif os.path.splitext(args.source)[-1].lower() == '.cptv':
        clip_classifier.process_file(os.path.join(args.source_folder, args.source))
    else:
        clip_classifier.process_folder(os.path.join(args.source_folder, args.source))


if __name__ == "__main__":
    main()
