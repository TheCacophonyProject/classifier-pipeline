"""
Author: Matthew Aitchison
Date: December 2017

Dataset used for training a tensorflow model from track data.

Tracks are broken into segments.  Filtered, and then passed to the trainer using a weighted random sample.

"""

import queue
import threading
import multiprocessing
import logging

import os
import datetime
import math
import random
import time
from dateutil import parser
from bisect import bisect

import numpy as np
import tensorflow as tf

from ml_tools import tools
from ml_tools.trackdatabase import TrackDatabase

class SegmentHeader():
    """ Header for segment. """

    def __init__(self, clip_id, track_number, start_frame, frames, weight, label, avg_mass):
        # reference to clip this segment came from
        self.clip_id = clip_id
        # reference to track this segment came from
        self.track_number = track_number
        # first frame of this segment
        self.start_frame = start_frame
        # length of segment in frames
        self.frames = frames
        # relative weight of the segment (higher is sampled more often)
        self.weight = weight
        # our label
        self.label = label.lower()
        self.avg_mass = avg_mass

    @property
    def name(self):
        """ Unique name of this segment. """
        return self.clip_id + '-' + str(self.track_number) + '-' + str(self.start_frame)

    @property
    def end_frame(self):
        """ end frame of sgement"""
        return self.start_frame+self.frames

    @property
    def track_id(self):
        """ Unique name of this segments track. """
        return TrackHeader.get_name(self.clip_id, self.track_number)

    def __str__(self):
        return "offset:{0} weight:{1:.1f}".format(self.start_frame, self.weight)


class TrackHeader():
    """ Header for track. """

    def __init__(self, clip_id, track_number, label, start_time, duration, camera):
        # reference to clip this segment came from
        self.clip_id = clip_id
        # reference to track this segment came from
        self.track_number = track_number
        # list of segments that belong to this track
        self.segments = []
        # label for this track
        self.label = label
        # date and time of the start of the track
        self.start_time = start_time
        # duration in seconds
        self.duration = duration
        # camera this track came from
        self.camera = camera

    @property
    def track_id(self):
        """ Unique name of this track. """
        return TrackHeader.get_name(self.clip_id, self.track_number)

    @property
    def bin_id(self):
        # name of the bin to assign this track to.
        return str(self.start_time.date())+'-'+str(self.camera)+'-'+self.label

    @property
    def weight(self):
        """ Returns total weight for all segments in this track"""
        return sum(segment.weight for segment in self.segments)

    @property
    def frames(self):
        """ Returns number of frames in this track"""
        return int(round(self.duration * 9))

    @staticmethod
    def get_name(clip_id, track_number):
        return str(clip_id) + '-' + str(track_number)

    @staticmethod
    def from_meta(clip_id, track_meta):
        """ Creates a track header from given metadata. """
        # kind of chacky way to get camera name from clip_id, in the future camera will be included in the metadata.
        camera = os.path.splitext(os.path.basename(clip_id))[0].split('-')[-1]
        result = TrackHeader(
            clip_id=clip_id, track_number=track_meta['id'], label=track_meta['tag'],
            start_time=parser.parse(track_meta['start_time']),
            duration=track_meta['duration'],
            camera=camera
        )
        return result

class Dataset():
    """
    Stores visit, clip, track, and segment information headers in memory, and allows track / segment streaming from
    disk.
    """

    def __init__(self, track_db: TrackDatabase, name="Dataset"):

        # database holding track data
        self.db = track_db

        # name of this dataset
        self.name = name

        # list of our tracks
        self.tracks = []
        self.track_by_id = {}
        self.tracks_by_label = {}
        self.tracks_by_bin = {}

        # cumulative distribution function for segments.  Allows for super fast weighted random sampling.
        self.segment_cdf = []

        # segments list
        self.segments = []

        # list of label names
        self.labels = []

        # number of frames each segment should be
        self.segment_width = 27
        # number of frames segments are spaced apart
        self.segment_spacing = 9
        # minimum mass of a segment frame for it to be included
        self.segment_min_mass = None
        # minimum average frame mass for segment to be included
        self.segment_avg_mass = None

        # constants used to normalise input
        self.normalisation_constants = None

        # this allows manipulation of data (such as scaling) during the sampling stage.
        self.enable_augmentation = False

        self.preloader_queue = None
        self.preloader_threads = None

        self.preloader_stop_flag = False

        # a copy of our entire dataset, if loaded.
        self.X = None
        self.y = None

    @property
    def rows(self):
        return len(self.segments)

    def next_batch(self, n, disable_async=False):
        """
        Returns a batch of n segments (X, y) from dataset.
        Applies augmentation and normalisation automatically.
        :param n: number of segments
        :return: X shape [n, channels, height, width], labels of shape [n]
        """

        # if async is enabled use it.
        if not disable_async and self.preloader_queue is not None:
            # get samples from queue
            batch_X = []
            batch_y = []
            for _ in range(n):
                X, y = self.preloader_queue.get()
                batch_X.append(X[0])
                batch_y.append(y[0])

            return np.asarray(batch_X), np.asarray(batch_y)

        segments = [self.sample_segment() for _ in range(n)]

        batch_X = []
        batch_y = []

        for segment in segments:
            data = self.fetch_segment(segment, normalise=True, augment=self.enable_augmentation)
            batch_X.append(data)
            batch_y.append(self.labels.index(segment.label))

        batch_X = np.float32(batch_X)
        batch_y = np.int32(batch_y)

        return(batch_X, batch_y)

    def load_tracks(self, track_filter=None):
        """
        Loads track headers from track database with optional filter
        :return: number of tracks added.
        """
        counter = 0
        for clip_id, track_number in self.db.get_all_track_ids():
            if self.add_track(clip_id, track_number, track_filter):
                counter += 1
        return counter

    def add_tracks(self, tracks, track_filter=None):
        """
        Adds list of tracks to dataset
        :param tracks: list of TrackHeader
        :param track_filter: optional filter
        """
        for track in tracks:
            self.add_track(track.clip_id, track.track_number, track_filter)

    def add_track(self, clip_id, track_number, track_filter=None):
        """
        Creates segments for track and adds them to the dataset
        :param track_filter: if provided a function filter(clip_meta, track_meta) that returns true when a track should be ignored)
        :return: True if track was added, false if it was filtered out.
        """

        # make sure we don't already have this track
        if TrackHeader.get_name(clip_id, track_number) in self.tracks:
            return

        clip_meta = self.db.get_clip_meta(clip_id)
        track_meta = self.db.get_track_meta(clip_id, track_number)
        if track_filter and track_filter(clip_meta, track_meta):
            return False

        track_header = TrackHeader.from_meta(clip_id, track_meta)

        self.tracks.append(track_header)
        self.track_by_id[track_header.track_id] = track_header

        if track_header.label not in self.tracks_by_label:
            self.tracks_by_label[track_header.label] = []
        self.tracks_by_label[track_header.label].append(track_header)

        if track_header.bin_id not in self.tracks_by_bin:
            self.tracks_by_bin[track_header.bin_id] = []
        self.tracks_by_bin[track_header.bin_id].append(track_header)

        # scan through track looking for good segments to add to our datset
        mass_history = track_meta['mass_history']
        for i in range(len(mass_history) // self.segment_spacing):
            segment_start = i * self.segment_spacing
            mass_slice = mass_history[segment_start:segment_start + self.segment_width]
            segment_min_mass = np.min(mass_slice)
            segment_avg_mass = np.median(mass_slice)
            segment_frames = len(mass_slice)

            if segment_frames != self.segment_width:
                continue

            if self.segment_min_mass and segment_min_mass < self.segment_min_mass:
                continue

            if self.segment_avg_mass and segment_avg_mass < self.segment_avg_mass:
                continue

            segment = SegmentHeader(
                clip_id=clip_id, track_number=track_number, start_frame=segment_start, frames=self.segment_width,
                weight=1.0, label=track_meta['tag'], avg_mass=segment_avg_mass)

            self.segments.append(segment)
            track_header.segments.append(segment)

        return True

    def filter_segments(self, avg_mass, ignore_labels=None):
        """
        Removes any segments with an average mass less than the given avg_mass
        :param avg_mass: segments with less avarage mass per frame than this will be removed from the dataset.
        :ignore_labels: these labels will not be filtered
        :return: number of segments removed
        """

        filtered = 0
        new_segments = []

        for segment in self.segments:
            if (ignore_labels and segment.label in ignore_labels) or segment.avg_mass >= avg_mass:
                new_segments.append(segment)
            else:
                filtered += 1

        self.segments = new_segments

        self._purge_track_segments()

        return filtered

    def fetch_all(self, normalise=True):
        """
        Fetches all segments
        :return: X of shape [n,27,channels,height,width], y of shape [n]
        """
        X = np.float32([self.fetch_segment(segment, normalise=normalise) for segment in self.segments])
        y = np.int32([self.labels.index(segment.label) for segment in self.segments])
        return X,y

    def fetch_segment(self, segment: SegmentHeader, normalise=False, augment=False):
        """
        Fetches data for segment.
        :param segment: The segment header to fetch
        :param normalise: if true normalises the channels in the segment according to normalisation_constants
        :param augment: if true applies data augmentation
        :return: segment of shape [frames, channels, height, width]
        """

        if augment:
            # jitter first frame
            prev_frames = segment.start_frame
            post_frames = self.track_by_id[segment.track_id].frames - segment.end_frame
            jitter = np.clip(np.random.randint(-5,5), -prev_frames, post_frames)
        else:
            jitter = 0

        data = self.db.get_track(segment.clip_id, segment.track_number, segment.start_frame + jitter, segment.end_frame + jitter)

        if len(data) != 27:
            print("ERROR, invalid segment length",len(data))

        # apply some thresholding.  This removes the noise from the background which helps a lot during training.
        # it is possiable that with enough data this will no longer be necessary.
        threshold = 10
        if threshold:
            data[:, 1, :, :] = np.clip(data[:, 1, :, :] - threshold, a_min=0, a_max=None)

        if augment:
            data = self.apply_augmentation(data)
        if normalise:
            data = self.apply_normalisation(data)

        return data

    def apply_normalisation(self, segment_data):
        """
        Applies a random augmentation to the segment_data.
        :param segment_data: array of shape [frames, channels, height, width]
        :return: normalised array
        """

        frames, channels, height, width = segment_data.shape

        segment_data = np.float32(segment_data)

        for channel in range(channels):
            mean, std = self.normalisation_constants[channel]
            segment_data[:, channel] -= mean
            if channel in [2,3]:
                segment_data[:, channel] = (np.sqrt(np.abs(segment_data[:, channel]))) * np.sign(segment_data[:, channel])
            segment_data[:, channel] *= (1.0/std)

        return segment_data

    def apply_augmentation(self, segment_data):
        """
        Applies a random augmentation to the segment_data.
        :param segment_data: array of shape [frames, channels, height, width]
        :return: augmented array
        """

        frames, channels, height, width = segment_data.shape

        segment_data = np.float32(segment_data)

        # apply scaling
        if random.randint(0, 1) == 0:
            mask = segment_data[:, 4, :, :]
            av_mass = np.sum(mask) / len(mask)
            scale_options = []
            if av_mass > 50: scale_options.append('down')
            if av_mass < 100: scale_options.append('up')
            scale_method = np.random.choice(scale_options)

            up_scale = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
            down_scale = [0.25, 0.33, 0.5, 0.75]

            if scale_method == 'up':
                scale = np.random.choice(up_scale)
            else:
                scale = np.random.choice(down_scale)

            for i in range(frames):
                # the image is expected to be in hwc format so we convert it here.
                segment_data[i] = tools.zoom_image(segment_data[i], scale=scale, channels_first=True)

        if random.randint(0,1) == 0:
            segment_data = np.flip(segment_data, axis = 3)

        return segment_data

    def sample_segment(self):
        """ Returns a random segment from weighted list. """
        roll = random.random()
        index = bisect(self.segment_cdf, roll)
        return self.segments[index]

    def load_all(self, force=False):
        """ Loads all X and y into dataset if required. """
        if self.X is None or force:
            self.X, self.y = self.fetch_all()

    def balance_weights(self, weight_modifiers=None):
        """
        Adjusts weights so that every class is evenly represented.
        :param weight_modifiers: if specified is a dictionary mapping from label to weight modifier,
            where < 1 sampled less frequently, and > 1 is sampled more frequently.
        :return:
        """

        class_weight = {}
        mean_class_weight = 0

        for class_name in self.labels:
            class_weight[class_name] = self.get_class_weight(class_name)
            mean_class_weight += class_weight[class_name] / len(self.labels)

        scale_factor = {}
        for class_name in self.labels:
            modifier = 1.0 if weight_modifiers is None else weight_modifiers.get(class_name, 1.0)
            if class_weight[class_name] == 0:
                scale_factor[class_name] = 1.0
            else:
                scale_factor[class_name] = mean_class_weight / class_weight[class_name] * modifier

        for segment in self.segments:
            segment.weight *= scale_factor.get(segment.label, 1.0)

        self.rebuild_cdf()

    def balance_bins(self, max_bin_weight):
        """
        Adjusts weights so that bins with a number number of segments aren't sampled so frequently.
        :param max_bin_weight: bins with more weight than this number will be scaled back to this weight.
        """

        for bin_name, tracks in self.tracks_by_bin.items():
            bin_weight = sum(track.weight for track in tracks)
            if bin_weight > max_bin_weight:
                scale_factor =  max_bin_weight / bin_weight
                for track in tracks:
                    for segment in track.segments:
                        segment.weight *= scale_factor

        self.rebuild_cdf()


    def balance_resample(self, required_samples, weight_modifiers=None):
        """ Removes segments until all classes have given number of samples (or less)"""

        new_segments = []

        for class_name in self.labels:
            segments = self.get_class_segments(class_name)
            required_class_samples = required_samples
            if weight_modifiers:
                required_class_samples = int(math.ceil(required_class_samples * weight_modifiers.get(class_name, 1.0)))
            if len(segments) > required_class_samples:
                # resample down
                segments = np.random.choice(segments, required_class_samples, replace=False).tolist()
            new_segments += segments

        self.segments = new_segments

        self._purge_track_segments()

        self.rebuild_cdf()

    def _purge_track_segments(self):
        """ Removes any segments from track_headers where the segment has been deleted """
        segment_set = set(self.segments)

        # remove segments from tracks
        for track in self.tracks:
            segments = track.segments
            segments = [segment for segment in segments if (segment in segment_set)]
            track.segments = segments

    def get_normalisation_constants(self, n=None):
        """
        Gets constants required for normalisation from dataset.  If n is specified uses a random sample of n segments.
        Segment weight is not taken into account during this sampling.  Otherrwise the entire dataset is used.
        :param n: If specified calculates constants from n samples
        :return: normalisation constants
        """

        # note:
        # we calculate the standard deviation and mean using the moments as this allows the calculation to be
        # done piece at a time.  Otherwise we'd need to load the entire dataset into memory, which might not be
        # possiable.

        if len(self.segments) == 0:
            raise Exception("No segments in dataset.")

        sample = self.segments if n is None or n >= len(self.segments) else random.sample(self.segments, n)

        # fetch a sample to see what the dims are
        example = self.fetch_segment(self.segments[0])
        _, channels, height, width = example.shape

        # we use float64 as this accumulator will get very large!
        first_moment = np.zeros((channels, height, width), dtype=np.float64)
        second_moment = np.zeros((channels, height, width), dtype=np.float64)

        for segment in sample:
            data = np.float64(self.fetch_segment(segment))
            first_moment += np.mean(data, axis=0)
            second_moment += np.mean(np.square(data), axis=0)

        # reduce down to channel only moments, in the future per pixel normalisation would be a good idea.
        first_moment = np.sum(first_moment, axis=(1,2)) / (len(sample) * width * height)
        second_moment = np.sum(second_moment, axis=(1,2)) / (len(sample) * width * height)

        mu = first_moment
        var = second_moment + (mu ** 2) - (2*mu*first_moment)

        normalisation_constants = [(mu[i], math.sqrt(var[i])) for i in range(channels)]

        return normalisation_constants

    def rebuild_cdf(self):
        """ Calculates the CDF used for fast random sampling """
        self.segment_cdf = []
        prob = 0
        for segment in self.segments:
            prob += segment.weight
            self.segment_cdf.append(prob)
        normalizer = self.segment_cdf[-1]
        self.segment_cdf = [x / normalizer for x in self.segment_cdf]

    def get_class_weight(self, label):
        """ Returns the total weight for all segments of given label. """
        return sum(segment.weight for segment in self.segments if segment.label == label)

    def get_class_segments_count(self, label):
        """ Returns the total weight for all segments of given class. """
        result = 0
        for track in self.tracks_by_label.get(label,[]):
            result += len(track.segments)
        return result

    def get_class_segments(self, label):
        """ Returns the total weight for all segments of given class. """
        result = []
        for track in self.tracks_by_label.get(label,[]):
            result.extend(track.segments)
        return result

    def start_async_load(self, buffer_size = 64):
        """
        Starts async load process.
        """

        # threading has limitations due to global lock
        # but processor ends up slow on windows as the numpy array needs to be pickled across processes which is
        # 2ms per process..
        # this could be solved either by using linux (with forking, which is copy on write) or with a shared ctype
        # array.

        WORKER_THREADS = 1
        PROCESS_BASED = False

        print("Starting async fetcher")
        if PROCESS_BASED:
            self.preloader_queue = multiprocessing.Queue(buffer_size)
            self.preloader_threads = [multiprocessing.Process(target=preloader, args=(self.preloader_queue, self, buffer_size)) for _ in range(WORKER_THREADS)]
        else:
            self.preloader_queue = queue.Queue(buffer_size)
            self.preloader_threads = [threading.Thread(target=preloader, args=(self.preloader_queue, self, buffer_size)) for _ in range(WORKER_THREADS)]

        self.preloader_stop_flag = False
        for thread in self.preloader_threads:
            thread.start()

    def stop_async_load(self, buffer_size = 64):
        """
        Stops async worker thread.
        """
        if self.preloader_threads is not None:
            self.preloader_stop_flag = True
            for thread in self.preloader_threads:
                thread.join()

# continue to read examples until queue is full
def preloader(q, dataset, buffer_size):
    """ add a segment into buffer """
    print("Async loader pool starting")
    loads = 0
    timer = time.time()
    while not dataset.preloader_stop_flag:
        if not q.full():
            q.put(dataset.next_batch(1, disable_async=True))
            loads += 1
            if (time.time() - timer) > 1.0:
                #logging.debug("{} segments per seconds {:.1f}".format(dataset.name, loads / (time.time() - timer)))
                timer = time.time()
                loads = 0
        else:
            time.sleep(0.01)

