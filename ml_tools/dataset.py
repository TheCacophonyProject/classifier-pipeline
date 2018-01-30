"""
Author: Matthew Aitchison
Date: December 2017

Dataset used for training a tensorflow model from track data.

Tracks are broken into segments.  Filtered, and then passed to the trainer using a weighted random sample.

"""

import queue
import threading
import multiprocessing
import cv2

import os
import math
import random
import time
from dateutil import parser
from bisect import bisect

import numpy as np

from ml_tools import tools
from ml_tools.trackdatabase import TrackDatabase

class TrackHeader:
    """ Header for track. """

    def __init__(self, clip_id, track_number, label, start_time, duration, camera, score):
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
        # score of track
        self.score = score
        # thermal reference point for each frame.
        self.thermal_reference_level = None
        # tracking frame movements for each frame, array of tuples (x-vel, y-vel)
        self.frame_velocity = None


    @property
    def track_id(self):
        """ Unique name of this track. """
        return TrackHeader.get_name(self.clip_id, self.track_number)

    @property
    def bin_id(self):
        # name of the bin to assign this track to.
        return str(self.start_time.date()) + '-' + str(self.camera) + '-' + self.label

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
    def from_meta(clip_id, clip_meta, track_meta):
        """ Creates a track header from given metadata. """

        # kind of checky way to get camera name from clip_id, in the future camera will be included in the metadata.
        camera = os.path.splitext(os.path.basename(clip_id))[0].split('-')[-1]

        # get the reference levels from clip_meta and load them into the track.
        track_start_frame = track_meta['start_frame']
        track_end_frame = track_meta['start_frame'] + int(round(track_meta['duration']*9))
        thermal_reference_level = np.float32(clip_meta['frame_temp_median'][track_start_frame:track_end_frame])

        # calculate the frame velocities
        bounds_history = track_meta['bounds_history']
        frame_center = [((left + right)/2, (top + bottom)/2) for left, top, right, bottom in bounds_history]
        frame_velocity = []
        prev = None
        for x, y in frame_center:
            if prev is None:
                frame_velocity.append((0.0,0.0))
            else:
                frame_velocity.append((x-prev[0], y-prev[1]))
            prev = (x, y)

        result = TrackHeader(
            clip_id=clip_id, track_number=track_meta['id'], label=track_meta['tag'],
            start_time=parser.parse(track_meta['start_time']),
            duration=float(track_meta['duration']),
            camera=camera,
            score=float(track_meta['score'])
        )
        result.thermal_reference_level = thermal_reference_level
        result.frame_velocity = frame_velocity

        return result

    def __repr__(self):
        return self.track_id

class SegmentHeader:
    """ Header for segment. """

    def __init__(self, track: TrackHeader, start_frame, frames, weight, avg_mass):
        # reference to track this segment came from
        self.track = track
        # first frame of this segment referenced by start of track
        self.start_frame = start_frame
        # length of segment in frames
        self.frames = frames
        # relative weight of the segment (higher is sampled more often)
        self.weight = weight
        # average mass of the segment
        self.avg_mass = avg_mass

    @property
    def clip_id(self):
        # reference to clip this segment came from
        return self.track.clip_id

    @property
    def track_number(self):
        # reference to track this segment came from
        return self.track.track_number

    @property
    def label(self):
        # label for this segment
        return self.track.label

    @property
    def name(self):
        """ Unique name of this segment. """
        return self.clip_id + '-' + str(self.track_number) + '-' + str(self.start_frame)

    @property
    def frame_velocity(self):
        # tracking frame velocity for each frame.
        return self.track.frame_velocity[self.start_frame:self.start_frame + self.frames]

    @property
    def thermal_reference_level(self):
        # thermal reference temperature for each frame (i.e. which temp is 0)
        return self.track.thermal_reference_level[self.start_frame:self.start_frame+self.frames]

    @property
    def end_frame(self):
        """ end frame of segment"""
        return self.start_frame + self.frames

    @property
    def track_id(self):
        """ Unique name of this segments track. """
        return TrackHeader.get_name(self.clip_id, self.track_number)

    def __str__(self):
        return "offset:{0} weight:{1:.1f}".format(self.start_frame, self.weight)

class Dataset:
    """
    Stores visit, clip, track, and segment information headers in memory, and allows track / segment streaming from
    disk.
    """

    # Number of threads to use for async loading
    WORKER_THREADS = 2

    # If true uses processes instead of threads.  Threads do not scale as well due to the GIL, however there is no
    # transfer time required per segment.  Processes scale much better but require ~1ms to pickling the segments
    # across processes.
    # In general if worker threads is one set this to False, if it is two or more set it to True.
    PROCESS_BASED = True

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

        # writes the frame motion into the center of the optical flow channels
        self.encode_frame_offsets_in_flow = False

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

        # dictionary used to apply label remapping during track load
        self.label_mapping = None

        # this allows manipulation of data (such as scaling) during the sampling stage.
        self.enable_augmentation = False
        # how often to scale during augmentation
        self.scale_frequency = 0.50

        self.preloader_queue = None
        self.preloader_threads = None
        self.preloader_stop_flag = False

        # a copy of our entire dataset, if loaded.
        self.X = None
        self.y = None

    @property
    def rows(self):
        return len(self.segments)

    def get_counts(self, label):
        """
        Gets number of examples for given label
        :label: label to check
        :return: (segments, tracks, bins, weight)
        """
        label_tracks = self.tracks_by_label.get(label, [])
        segments = sum(len(track.segments) for track in label_tracks)
        weight = self.get_class_weight(label)
        tracks = len(label_tracks)
        bins = len([tracks for bin_name, tracks in self.tracks_by_bin.items()
                    if len(tracks) > 0 and tracks[0].label == label])
        return segments, tracks, bins, weight

    def next_batch(self, n, disable_async=False, force_no_augmentation=False):
        """
        Returns a batch of n segments (X, y) from dataset.
        Applies augmentation and preprocessing automatically.
        :param n: number of segments
        :param disable_async: forces fetching of segment in this thread / process rather than collecting from
            an aync reader queue (if one exists)
        :param force_no_augmentation: forces augmentation off, may disable asyc loading.
        :return: X of shape [n, channels, height, width], y (labels) of shape [n]
        """

        # if async is enabled use it.
        if (not disable_async and self.preloader_queue is not None) or force_no_augmentation:
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

            data = self.fetch_segment(segment, augment=self.enable_augmentation and not force_no_augmentation)
            batch_X.append(data)
            batch_y.append(self.labels.index(segment.label))

            if np.isnan(data).any():
                print("Warning NaN found in data from source {}".format(segment.clip_id))

            # I've been getting some NaN's come through so I check here to make sure input is reasonable.
            # note: this is a really good idea, however my data has some '0' values in the thermal data which come
            # through as -reference level (i.e. -3000) so I've disabled this for now.
            #if np.max(np.abs(data)) > 2000:
            #    print("Extreme values found in batch from source {} with value +-{:.1f}".format(segment.clip_id,
            #                                                                              np.max(np.abs(data))))

        # Half float should be fine here.  When using process based async loading we have to pickle the batch between
        # processes, so having it half the size helps a lot.  Also it reduces the memory required for the read buffers
        batch_X = np.float16(batch_X)
        batch_y = np.int32(batch_y)

        return batch_X, batch_y

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
        result = 0
        for track in tracks:
            if self.add_track(track.clip_id, track.track_number, track_filter):
                result += 1
        return result

    def add_track(self, clip_id, track_number, track_filter=None):
        """
        Creates segments for track and adds them to the dataset
        :param clip_id: id of tracks clip
        :param track_number: track number
        :param track_filter: if provided a function filter(clip_meta, track_meta) that returns true when a track should
                be ignored)
        :return: True if track was added, false if it was filtered out.
        :return:
        """

        # make sure we don't already have this track
        if TrackHeader.get_name(clip_id, track_number) in self.track_by_id:
            return False

        clip_meta = self.db.get_clip_meta(clip_id)
        track_meta = self.db.get_track_meta(clip_id, track_number)
        if track_filter and track_filter(clip_meta, track_meta):
            return False

        track_header = TrackHeader.from_meta(clip_id, clip_meta, track_meta)
        if self.label_mapping and track_header.label in self.label_mapping:
            track_header.label = self.label_mapping[track_header.label]

        self.tracks.append(track_header)
        self.track_by_id[track_header.track_id] = track_header

        if track_header.label not in self.tracks_by_label:
            self.tracks_by_label[track_header.label] = []
        self.tracks_by_label[track_header.label].append(track_header)

        if track_header.bin_id not in self.tracks_by_bin:
            self.tracks_by_bin[track_header.bin_id] = []

        self.tracks_by_bin[track_header.bin_id].append(track_header)

        mass_history = track_meta['mass_history']
        segment_count = len(mass_history) // self.segment_spacing

        # scan through track looking for good segments to add to our datset

        for i in range(segment_count):
            segment_start = i * self.segment_spacing
            mass_slice = mass_history[segment_start:segment_start + self.segment_width]
            segment_avg_mass = np.mean(mass_slice)
            segment_frames = len(mass_slice)

            if segment_frames != self.segment_width:
                continue

            # try to sample the better segments more often
            if segment_avg_mass < 50:
                segment_weight_factor = 0.75
            elif segment_avg_mass < 100:
                segment_weight_factor = 1
            else:
                segment_weight_factor = 1.2

            segment = SegmentHeader(
                track=track_header,
                start_frame=segment_start, frames=self.segment_width,
                weight=segment_weight_factor, avg_mass=segment_avg_mass
            )

            segment.thermal_reference = np.float32(clip_meta['frame_temp_median'][segment.start_frame:segment.end_frame])

            self.segments.append(segment)
            track_header.segments.append(segment)

        return True

    def filter_segments(self, avg_mass, ignore_labels=None):
        """
        Removes any segments with an average mass less than the given avg_mass
        :param avg_mass: segments with less avarage mass per frame than this will be removed from the dataset.
        :param ignore_labels: these labels will not be filtered
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

    def fetch_all(self):
        """
        Fetches all segments
        :return: X of shape [n,f,channels,height,width], y of shape [n]
        """
        X = np.float32([self.fetch_segment(segment) for segment in self.segments])
        y = np.int32([self.labels.index(segment.label) for segment in self.segments])
        return X, y

    def fetch_track(self, track: TrackHeader):
        """
        Fetches data for an entire track
        :param track: the track to fetch
        :return: segment data of shape [frames, channels, height, width]
        """
        data = self.db.get_track(track.clip_id, track.track_number, 0, track.frames)
        data = self.apply_preprocessing(data, track.thermal_reference_level, track.frame_velocity)
        return data

    def fetch_segment(self, segment: SegmentHeader, augment=False):
        """
        Fetches data for segment.
        :param segment: The segment header to fetch
        :param augment: if true applies data augmentation
        :return: segment data of shape [frames, channels, height, width]
        """

        # if we are requesting a segment smaller than the default segment size take it from the middle.
        unused_frames = (segment.frames - self.segment_width)
        if unused_frames < 0:
            raise Exception("Maximum segment size for the dataset is {} frames, but requested {}".format(
                segment.frames, self.segment_width))
        first_frame = segment.start_frame + (unused_frames // 2)
        last_frame = segment.start_frame + (unused_frames // 2) + self.segment_width

        if augment:
            # jitter first frame
            prev_frames = first_frame
            post_frames = self.track_by_id[segment.track_id].frames - last_frame
            max_jitter = max(5, unused_frames)
            jitter = np.clip(np.random.randint(-max_jitter, max_jitter), -prev_frames, post_frames)
        else:
            jitter = 0

        data = self.db.get_track(segment.clip_id, segment.track_number, first_frame + jitter,
                                 last_frame + jitter)

        if len(data) != self.segment_width:
            print("ERROR, invalid segment length {}, expected {}", len(data), self.segment_width)

        data = self.apply_preprocessing(data, segment.thermal_reference_level, segment.frame_velocity)

        if augment:
            data = self.apply_augmentation(data)

        return data

    def apply_preprocessing(self, data, reference_level, frame_velocity=None):
        """
        Applies pre-processing to segment data.  For example mapping optical flow down to the right level.
        :param data: np array of shape [F, C, H, W]
        :param reference_level: thermal reference level for each frame in data
        :param frame_velocity: veloctiy (x,y) for each frame.
        """

        # the segment will be processed in float32 so we may aswell convert it here.
        # also optical flow is stored as a scaled integer, but we want it in float32 format.
        data = np.asarray(data, dtype=np.float32)

        # use the frame median as reference poitn.
        # this should be the better way to go, but sometimes doesn't work for strange reasons.
        data[:, 0, :, :] -= np.float32(reference_level)[:, np.newaxis, np.newaxis]

        # get reference level for thermal channel
        assert len(data) == len(reference_level), "Reference level shape and data shape not match."

        # map optical flow down to right level,
        # we pre-multiplied by 256 to fit into a 16bit int
        data[:, 2:3 + 1, :, :] *= (1/256)

        # write frame motion into center of frame
        if self.encode_frame_offsets_in_flow:
            F, C, H, W = data.shape
            for x in range(-2,2+1):
                for y in range(-2,2+1):
                    data[:, 2:3 + 1, H//2+y, W//2+x] = frame_velocity[:, :]

        # set filtered track to delta frames
        reference = np.clip(data[:, 0], 20, 999)
        data[0, 1] = 0
        data[1:, 1] = reference[1:] - reference[:-1]

        return data


    def apply_augmentation(self, segment_data, force_scale=None):
        """
        Applies a random augmentation to the segment_data.
        :param segment_data: array of shape [frames, channels, height, width]
        :param force_scale: (optional) force a specific scaling amount
        :return: augmented array
        """

        frames, channels, height, width = segment_data.shape

        if random.random() <= self.scale_frequency:
            # we will adjust contrast and levels, but only within these bounds.
            # that is a bright input may have brightness reduced, but not increased.
            MAX_LEVEL_OFFSET = 10

            # apply level and contrast shift
            level_adjust = (random.random() - 0.5) * MAX_LEVEL_OFFSET
            contrast_adjust = tools.random_log(0.8, 1.2)

            segment_data[:,0] *= contrast_adjust
            segment_data[:, 0] += level_adjust

        # apply scaling only some of time
        if random.random() <= self.scale_frequency or force_scale is not None:
            mask = segment_data[:, 4]
            av_mass = np.sum(mask) / len(mask)
            size = math.sqrt(av_mass + 4)

            # work out reasonable bounds so we don't scale too much
            # general idea is scale to a width of between 6 and 48 pixels
            max_scale_up = np.clip(36 / size, 1.0, 3.0)
            min_scale_down = np.clip(4 / size, 0.85, 1.0)

            if force_scale is not None:
                scale = force_scale
            else:
                scale = tools.random_log(min_scale_down, max_scale_up)

            # don't apply scaling if it's just a very small scale amount
            if scale != 1.0:
                position_jitter = 5 / scale
                xofs = random.normalvariate(0, position_jitter)
                yofs = random.normalvariate(0, position_jitter)
                for i in range(frames):
                    segment_data[i] = tools.zoom_image(segment_data[i], scale=scale, channels_first=True,
                                                       offset_x=xofs, offset_y=yofs,
                                                       interpolation=cv2.INTER_LINEAR)

        if random.randint(0, 1) == 0:
            # when we flip the frame remember to flip the horizontal velocity as well
            segment_data = np.flip(segment_data, axis=3)
            segment_data[:, 2] = -segment_data[:, 2]

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
                scale_factor = max_bin_weight / bin_weight
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

    def remove_label(self, label_to_remove):
        """
        Removes all segments of given label from dataset. Label remains in dataset.labels however, so as to not
        change the ordinal value of the labels.
        """
        if label_to_remove not in self.labels:
            return
        self.segments = [segment for segment in self.segments if segment.label != label_to_remove]
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
        first_moment = np.sum(first_moment, axis=(1, 2)) / (len(sample) * width * height)
        second_moment = np.sum(second_moment, axis=(1, 2)) / (len(sample) * width * height)

        mu = first_moment
        var = second_moment + (mu ** 2) - (2 * mu * first_moment)

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
        for track in self.tracks_by_label.get(label, []):
            result += len(track.segments)
        return result

    def get_class_segments(self, label):
        """ Returns the total weight for all segments of given class. """
        result = []
        for track in self.tracks_by_label.get(label, []):
            result.extend(track.segments)
        return result

    def start_async_load(self, buffer_size=64):
        """
        Starts async load process.
        """

        # threading has limitations due to global lock
        # but processor ends up slow on windows as the numpy array needs to be pickled across processes which is
        # 2ms per process..
        # this could be solved either by using linux (with forking, which is copy on write) or with a shared ctype
        # array.

        if self.PROCESS_BASED:
            self.preloader_queue = multiprocessing.Queue(buffer_size)
            self.preloader_threads = [multiprocessing.Process(target=preloader, args=(self.preloader_queue, self)) for _
                                      in range(self.WORKER_THREADS)]
        else:
            self.preloader_queue = queue.Queue(buffer_size)
            self.preloader_threads = [threading.Thread(target=preloader, args=(self.preloader_queue, self)) for _ in
                                      range(self.WORKER_THREADS)]

        self.preloader_stop_flag = False
        for thread in self.preloader_threads:
            thread.start()

    def stop_async_load(self):
        """
        Stops async worker thread.
        """
        if self.preloader_threads is not None:
            for thread in self.preloader_threads:
                if hasattr(thread, 'terminate'):
                    # note this will corrupt the queue, so reset it
                    thread.terminate()
                    self.preloader_queue = None
                else:
                    thread.exit()


# continue to read examples until queue is full
def preloader(q, dataset):
    """ add a segment into buffer """
    print(" -started async fetcher for {} with augment={} segment_width={}".format(
        dataset.name, dataset.enable_augmentation, dataset.segment_width))
    loads = 0
    timer = time.time()
    while not dataset.preloader_stop_flag:
        if not q.full():
            q.put(dataset.next_batch(1, disable_async=True))
            loads += 1
            if (time.time() - timer) > 1.0:
                # logging.debug("{} segments per seconds {:.1f}".format(dataset.name, loads / (time.time() - timer)))
                timer = time.time()
                loads = 0
        else:
            time.sleep(0.01)
