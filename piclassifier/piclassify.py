#!/usr/bin/python3
import logging
from typing import Dict
from datetime import datetime
import numpy as np
import socket
import sys
from classify.trackprediction import RollingTrackPrediction, TrackPrediction
from load.clip import Clip, ClipTrackExtractor
from ml_tools import tools
from ml_tools.model import Model
from ml_tools.dataset import Preprocessor
from ml_tools.previewer import Previewer
from track.track import Track
from ml_tools.tools import Rectangle

from ml_tools.config import Config

SOCKET_NAME = "/var/run/lepton-frames"


def get_classifier(config):
    """
    Returns a classifier object, which is created on demand.
    This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
    """
    t0 = datetime.now()
    logging.info("classifier loading")
    classifier = Model(
        train_config=config.train,
        session=tools.get_session(disable_gpu=not config.use_gpu),
    )
    classifier.load(config.classify.model)
    logging.info("classifier loaded ({})".format(datetime.now() - t0))

    return classifier


def main():
    # Create a UDS socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Connect the socket to the port where the server is listening
    print("connecting to {}".format(SOCKET_NAME))
    try:
        sock.connect(SOCKET_NAME)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    config = Config.load_from_file()
    classifier = get_classifier(config)

    clip_classifier = PiClassifier(config, config.classify_tracking, classifier)

    clip = Clip(config.tracking, config.load.cache_to_disk, config.use_opt_flow)
    img_dtype = np.uint16
    i = 0
    try:
        p = -1
        while p != 0:
            i += 1
            thermal_frame = np.empty((120, 160), dtype=img_dtype)
            p = sock.recv_into(thermal_frame, 0, socket.MSG_WAITALL)
            if p != 120 * 160 * 2:
                print(f"got{p} expecting {120 * 160 * 2}")
            else:
                clip_classifier.process_frame(thermal_frame)
    finally:
        print("closing socket")
        sock.close()


class PiClassifier:
    """ Classifies tracks within CPTV files. """

    # skips every nth frame.  Speeds things up a little, but reduces prediction quality.
    FRAME_SKIP = 1

    def __init__(self, config, tracking_config, classifier):
        """ Create an instance of a clip classifier"""
        # prediction record for each track
        self.config = config
        self.track_prediction: Dict[Track, TrackPrediction] = {}
        self.classifier = classifier
        self.previewer = Previewer.create_if_required(config, config.classify.preview)
        self.num_labels = len(classifier.labels)

        self.start_date = None
        self.end_date = None
        self.cache_to_disk = config.classify.cache_to_disk
        # enables exports detailed information for each track.  If preview mode is enabled also enables track previews.
        self.enable_per_track_information = False
        self.prediction_per_track = {}
        try:
            self.fp_index = self.classifier.labels.index("false-positive")
        except ValueError:
            self.fp_index = None

        self.clip = Clip(
            self.config.tracking, self.cache_to_disk, self.config.use_opt_flow
        )

        self.clip.preview_frames = 7
        self.clip.track_extractor = ClipTrackExtractor(
            self.clip,
            self.config.tracking,
            self.cache_to_disk,
            False,
            self.config.tracking.high_quality_optical_flow,
        )
        edge = self.config.tracking.edge_pixels
        res_x = 160
        res_y = 120
        crop_rectangle = Rectangle(edge, edge, res_x - 2 * edge, res_y - 2 * edge)

        self.clip.track_extractor.crop_rectangle = Rectangle(edge, edge, res_x - 2 * edge, res_y - 2 * edge)

    def identify_last_frame(self):
        """
        Runs through track identifying segments, and then returns it's prediction of what kind of animal this is.
        One prediction will be made for every frame.
        :param track: the track to identify.
        :return: TrackPrediction object
        """

        prediction_smooth = 0.1

        smooth_prediction = None
        smooth_novelty = None

        prediction = 0.0
        novelty = 0.0

        active_tracks = self.clip.track_extractor.active_tracks
        frame = self.clip.track_extractor.frame_buffer.get_last_frame()
        if frame is None:
            return
        thermal_reference = np.median(frame.thermal)
        print(f"active tracks {len(active_tracks)}")
        for i, track in enumerate(active_tracks):
            track_prediction = self.prediction_per_track.setdefault(
                track.get_id(), RollingTrackPrediction(track.get_id())
            )
            region = track.bounds_history[-1]
            if region.frame_number != frame.frame_number:
                print("frame doesn't match last frame")
            else:
                track_data = track.crop_by_region(frame, region)
                # track_data = track.crop_by_region_at_trackframe(frame, i)
                if i % self.FRAME_SKIP == 0:
                    # we use a tigher cropping here so we disable the default 2 pixel inset
                    frames = Preprocessor.apply(
                        [track_data], [thermal_reference], default_inset=0
                    )

                    if frames is None:
                        logging.info(
                            "Frame {} of track could not be classified.".format(
                                region.frame_number
                            )
                        )
                        return

                    frame = frames[0]
                    prediction, novelty, state = self.classifier.classify_frame_with_novelty(
                        frame, track_prediction.state
                    )
                    track_prediction.state = state

                    if self.fp_index is not None:
                        prediction[self.fp_index] *= 0.8
                    state *= 0.98
                    mass = region.mass
                    mass_weight = np.clip(mass / 20, 0.02, 1.0) ** 0.5
                    cropped_weight = 0.7 if region.was_cropped else 1.0

                    prediction *= mass_weight * cropped_weight

                if smooth_prediction is None:
                    if track_prediction.uniform_prior:
                        smooth_prediction = np.ones([self.num_labels]) * (
                            1 / self.num_labels
                        )
                    else:
                        smooth_prediction = prediction
                    smooth_novelty = 0.5
                else:
                    smooth_prediction = (
                        1 - prediction_smooth
                    ) * smooth_prediction + prediction_smooth * prediction
                    smooth_novelty = (
                        1 - prediction_smooth
                    ) * smooth_novelty + prediction_smooth * novelty

                track_prediction.predictions.append(smooth_prediction)
                track_prediction.novelties.append(smooth_novelty)

    def get_clip_prediction(self):
        """ Returns list of class predictions for all tracks in this clip. """

        class_best_score = [0 for _ in range(len(self.classifier.labels))]

        # keep track of our highest confidence over every track for each class
        for _, prediction in self.track_prediction.items():
            for i in range(len(self.classifier.labels)):
                class_best_score[i] = max(
                    class_best_score[i], prediction.class_best_score[i]
                )

        results = []
        for n in range(1, 1 + len(self.classifier.labels)):
            nth_label = int(np.argsort(class_best_score)[-n])
            nth_score = float(np.sort(class_best_score)[-n])
            results.append((self.classifier.labels[nth_label], nth_score))

        return results

    def process_frame(self, thermal_frame):
        """
        Process a file extracting tracks and identifying them.
        :param filename: filename to process
        :param enable_preview: if true an MPEG preview file is created.
        """
        print("processing")
        self.clip.track_extractor.process_frame(thermal_frame)
        self.identify_last_frame()
        for key, value in self.prediction_per_track.items():
            print(f"Track {key} is {value.get_prediction(self.classifier.labels)}")
