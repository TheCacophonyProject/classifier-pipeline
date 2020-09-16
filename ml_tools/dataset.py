"""
Author: Matthew Aitchison
Date: December 2017

Dataset used for training a tensorflow model from track data.

Tracks are broken into segments.  Filtered, and then passed to the trainer using a weighted random sample.

"""
from PIL import Image, ImageDraw, ImageFont, ImageColor
import itertools as it
import logging
import math
import multiprocessing
import os
import queue
import random
import threading
import time
from bisect import bisect
import json
import cv2
import dateutil
import numpy as np
import scipy.ndimage

# from load.clip import Clip
from ml_tools import tools
from ml_tools.trackdatabase import TrackDatabase
from track.region import Region

CPTV_FILE_WIDTH = 160
CPTV_FILE_HEIGHT = 120
FRAMES_PER_SECOND = 9


class TrackChannels:
    """ Indexes to channels in track. """

    thermal = 0
    filtered = 1
    flow_h = 2
    flow_v = 3
    mask = 4


class TrackHeader:
    """ Header for track. """

    def __init__(
        self,
        clip_id,
        track_id,
        label,
        start_time,
        num_frames,
        duration,
        camera,
        location,
        score,
        track_bounds,
        frame_temp_median,
        frames_per_second,
        predictions,
        correct_prediction,
        frame_mass,
        start_frame,
    ):
        self.predictions = predictions
        self.correct_prediction = correct_prediction
        self.filtered_stats = {"segment_mass": 0}
        # reference to clip this segment came from
        self.clip_id = clip_id
        # reference to track this segment came from
        self.track_id = track_id
        # list of segments that belong to this track
        self.segments = []
        # label for this track
        self.label = label
        # date and time of the start of the track
        self.start_time = start_time
        self.start_frame = start_frame
        # duration in seconds
        self.duration = duration
        # camera this track came from
        self.camera = camera

        self.location = location
        # score of track
        self.score = score
        # thermal reference point for each frame.
        self.frame_temp_median = frame_temp_median
        # tracking frame movements for each frame, array of tuples (x-vel, y-vel)
        self.frame_velocity = None
        # original tracking bounds
        self.track_bounds = track_bounds
        # what fraction of pixels are from out of bounds
        self.frame_crop = []
        self.num_frames = num_frames
        self.frames_per_second = frames_per_second
        self.calculate_velocity()
        self.calculate_frame_crop()
        self.important_frames = []
        self.important_predicted = 0
        self.frame_mass = frame_mass
        self.median_mass = np.median(frame_mass)
        self.mean_mass = np.mean(frame_mass)

    def toJSON(self):
        meta_dict = {}
        meta_dict["clip_id"] = int(self.clip_id)
        meta_dict["track_id"] = int(self.track_id)
        meta_dict["camera"] = self.camera
        meta_dict["num_frames"] = int(self.num_frames)
        positions = []
        for region in self.track_bounds:
            positions.append(region.tolist())

        meta_dict["frames_per_second"] = int(self.frames_per_second)
        meta_dict["track_bounds"] = positions
        meta_dict["start_frame"] = int(self.start_frame)
        if self.location is not None:
            meta_dict["location_hash"] = "{}{}".format(
                hash(self.location[0]), hash(self.location[1])
            )
        meta_dict["label"] = self.label

        return json.dumps(meta_dict, indent=3)

    def add_sample(self, sample, use_segments):
        if use_segments:
            self.segments.append(sample)
        else:
            self.important_frames.append(sample)

    def get_sample_frames(self):
        return self.important_frames

    def remove_sample_frame(self, f):
        self.important_frams.remove(f)

    def get_sample_frame(self, i=0, remove=False):
        if len(self.important_frames) == 0:
            return None
        f = self.important_frames[i]

        if remove:
            del self.important_frames[i]
        return f

    @property
    def frame_weight(self):
        return 1 / self.num_sample_frames

    @property
    def num_sample_frames(self):
        return len(self.important_frames)

    # trying to get only clear frames
    def set_important_frames(
        self, labels, min_mass=None, use_predictions=False, filtered_data=None
    ):
        # this needs more testing
        frames = []
        for i, mass in enumerate(self.frame_mass):
            if min_mass is None or mass >= min_mass:
                if self.label is not "false-positive" and filtered_data is not None:
                    filtered = filtered_data[i]
                    if not filtered_is_valid(filtered):
                        logging.debug(
                            "set_important_frames %s frame %s has no zeros in filtered frame",
                            self.unique_id,
                            i,
                        )
                        continue
                frames.append(i)
        np.random.shuffle(frames)

        if self.predictions is not None and use_predictions:
            fp_i = None
            # if self.label in labels:
            #     label_i = list(labels).index(self.label)
            if "false-positive" in labels:
                fp_i = list(labels).index("false-positive")
            best_preds = []
            for i in frames:
                pred = self.predictions[i]

                best = np.argsort(pred)[-1]

                if self.label != "false-positive" and fp_i and best == fp_i:
                    continue

                pred_percent = pred[best]
                if pred_percent >= MIN_PERCENT:
                    best_preds.append((i, pred_percent))

            pred_frames = list(f[0] for f in best_preds)
            frames = pred_frames
        for frame in frames:
            f = FrameSample(self.clip_id, self.track_id, frame, self.label)
            self.important_frames.append(f)

    def calculate_frame_crop(self):
        # frames are always square, but bounding rect may not be, so to see how much we clipped I need to create a square
        # bounded rect and check it against frame size.
        self.frame_crop = []
        for rect in self.track_bounds:
            rect = tools.Rectangle.from_ltrb(*rect)
            rx, ry = rect.mid_x, rect.mid_y
            size = max(rect.width, rect.height)
            adjusted_rect = tools.Rectangle(rx - size / 2, ry - size / 2, size, size)
            self.frame_crop.append(
                get_cropped_fraction(adjusted_rect, CPTV_FILE_WIDTH, CPTV_FILE_HEIGHT)
            )

    def calculate_velocity(self):
        frame_center = [
            ((left + right) / 2, (top + bottom) / 2)
            for left, top, right, bottom in self.track_bounds
        ]
        self.frame_velocity = []
        prev = None
        for x, y in frame_center:
            if prev is None:
                self.frame_velocity.append((0.0, 0.0))
            else:
                self.frame_velocity.append((x - prev[0], y - prev[1]))
            prev = (x, y)

    def calculate_segments(
        self,
        mass_history,
        segment_frame_spacing,
        segment_width,
        segment_min_mass=None,
        random=False,
    ):
        if random and self.num_sample_frames < segment_width / 2.0:
            # dont want to repeat too many frames
            return
        elif len(mass_history) < segment_width:
            return
        segment_count = (len(mass_history) - segment_width) // segment_frame_spacing
        segment_count += 1

        if random:
            remaining = segment_width - self.num_sample_frames
            sample_size = min(segment_width, self.num_sample_frames)

            segment_count = max(0, (self.num_sample_frames - segment_width) // 4)
            segment_count += 1
            self.segments = []
            # take any segment_width frames, this could be done each epoch
            for i in range(segment_count):
                frames = list(
                    np.random.choice(
                        self.important_frames,
                        min(segment_width, len(self.important_frames)),
                        replace=False,
                    )
                )
                # sample another batch
                if remaining > 0:
                    frames.extend(
                        np.random.choice(
                            self.important_frames, remaining, replace=False,
                        )
                    )
                frames = [frame.frame_num for frame in frames]
                frames.sort()
                mass_slice = mass_history[frames]
                segment_avg_mass = np.mean(mass_slice)
                if segment_avg_mass < 50:
                    segment_weight_factor = 0.75
                elif segment_avg_mass < 100:
                    segment_weight_factor = 1
                else:
                    segment_weight_factor = 1.2
                segment = SegmentHeader(
                    track=self,
                    start_frame=0,
                    frames=segment_width,
                    weight=segment_weight_factor,
                    avg_mass=segment_avg_mass,
                    frame_indices=frames,
                )
                self.segments.append(segment)

        # scan through track looking for good segments to add to our datset
        for i in range(segment_count):
            segment_start = i * segment_frame_spacing

            mass_slice = mass_history[segment_start : segment_start + segment_width]
            segment_avg_mass = np.mean(mass_slice)
            segment_frames = len(mass_slice)

            if segment_frames != segment_width:
                continue
            if segment_min_mass and segment_avg_mass < segment_min_mass:
                self.filtered_stats["segment_mass"] += 1
                continue
            # try to sample the better segments more often
            if segment_avg_mass < 50:
                segment_weight_factor = 0.75
            elif segment_avg_mass < 100:
                segment_weight_factor = 1
            else:
                segment_weight_factor = 1.2

            segment = SegmentHeader(
                track=self,
                start_frame=segment_start,
                frames=segment_width,
                weight=segment_weight_factor,
                avg_mass=segment_avg_mass,
            )

            self.segments.append(segment)

    @property
    def camera_id(self):
        """ Unique name of this track. """
        return "{}-{}".format(self.camera, self.location)

    @property
    def bin_id(self):
        """ Unique name of this track. """
        return "{}-{}".format(self.clip_id, self.track_id)

    @property
    def weight(self):
        """ Returns total weight for all segments in this track"""
        return sum(segment.weight for segment in self.segments)

    @property
    def unique_id(self):
        return "{}-{}".format(self.clip_id, self.track_id)

    @staticmethod
    def from_meta(clip_id, clip_meta, track_meta, predictions=None):
        """ Creates a track header from given metadata. """
        correct_prediction = track_meta.get("correct_prediction", None)

        start_time = dateutil.parser.parse(track_meta["start_time"])
        end_time = dateutil.parser.parse(track_meta["end_time"])
        duration = (end_time - start_time).total_seconds()
        location = clip_meta.get("location")
        num_frames = track_meta["frames"]
        camera = clip_meta["device"]
        frames_per_second = clip_meta.get("frames_per_second", FRAMES_PER_SECOND)
        # get the reference levels from clip_meta and load them into the track.
        track_start_frame = track_meta["start_frame"]
        track_end_frame = track_meta["end_frame"]
        frame_temp_median = np.float32(
            clip_meta["frame_temp_median"][
                track_start_frame : num_frames + track_start_frame
            ]
        )

        bounds_history = track_meta["bounds_history"]

        header = TrackHeader(
            clip_id=int(clip_id),
            track_id=int(track_meta["id"]),
            label=track_meta["tag"],
            start_time=start_time,
            num_frames=num_frames,
            duration=duration,
            camera=camera,
            location=location,
            score=float(track_meta["score"]),
            track_bounds=np.asarray(bounds_history),
            frame_temp_median=frame_temp_median,
            frames_per_second=frames_per_second,
            predictions=predictions,
            correct_prediction=correct_prediction,
            frame_mass=track_meta["mass_history"],
            start_frame=track_start_frame,
        )
        return header

    def __repr__(self):
        return self.unique_id


class Camera:
    def __init__(self, camera):
        self.label_to_bins = {}
        self.label_to_tracks = {}

        self.bins = {}
        self.camera = camera
        self.bin_segment_sum = {}
        self.segment_sum = 0
        self.segments = 0
        self.label_frames = {}
        # used to sample bins
        self.bin_i = -1

    def label_tracks(self, label):
        return len(self.label_to_bins[label])

    def sample_frame(self, label):
        bins = self.label_to_bins[label]
        if len(bins) == 0:
            return None, None
        self.bin_i += 1
        self.bin_i = self.bin_i % len(bins)

        bin_id = bins[self.bin_i]
        track = self.bins[bin_id][0]
        f = track.get_sample_frame()
        if len(track.important_frames) == 0 or f is None:
            del bins[self.bin_i]
            del self.bins[bin_id]

        return track, f

    def label_segment_count(
        self, label, max_segments_per_track=None,
    ):
        if label not in self.label_to_tracks:
            return 0
        tracks = self.label_to_tracks[label].values()
        frames = 0
        for track in tracks:
            if max_segments_per_track:
                frames += max(len(track.segments), max_segments_per_track)
            else:
                frames += len(track.segments)

        return frames

    def label_frame_count(
        self, label, max_frames_per_track=None,
    ):
        if label not in self.label_to_tracks:
            return 0
        tracks = self.label_to_tracks[label].values()
        frames = 0
        for track in tracks:
            if max_frames_per_track:
                frames += max(len(track.important_frames), max_frames_per_track)
            else:
                frames += len(track.important_frames)

        return frames

    def remove_track(self, track):
        self.segments -= 1
        self.segment_sum -= len(track.segments)
        del self.label_to_tracks["wallaby"][track.unique_id]

    def add_track(self, track_header):
        tracks = self.label_to_tracks.setdefault(track_header.label, {})
        tracks[track_header.unique_id] = track_header
        if track_header.bin_id not in self.bins:
            self.bins[track_header.bin_id] = []
            self.bin_segment_sum[track_header.bin_id] = 0

        if track_header.label not in self.label_to_bins:
            self.label_to_bins[track_header.label] = []
            self.label_frames[track_header.label] = 0

        if track_header.bin_id not in self.label_to_bins[track_header.label]:
            self.label_to_bins[track_header.label].append(track_header.bin_id)

        self.bins[track_header.bin_id].append(track_header)
        self.label_frames[track_header.label] += len(track_header.important_frames)

        segment_length = len(track_header.segments)
        self.bin_segment_sum[track_header.bin_id] += segment_length
        self.segment_sum += segment_length
        self.segments += 1


class FrameSample:
    def __init__(self, clip_id, track_id, frame_num, label):
        self.clip_id = clip_id
        self.track_id = track_id
        self.frame_num = frame_num
        self.label = label

    @property
    def unique_track_id(self):
        return "{}-{}".format(self.clip_id, self.track_id)


class SegmentHeader:
    """ Header for segment. """

    def __init__(
        self,
        track: TrackHeader,
        start_frame,
        frames,
        weight,
        avg_mass,
        frame_indices=None,
    ):
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
        self.frame_indices = frame_indices

    @property
    def unique_track_id(self):
        # reference to clip this segment came from
        return self.track.unique_id

    @property
    def clip_id(self):
        # reference to clip this segment came from
        return self.track.clip_id

    @property
    def label(self):
        # label for this segment
        return self.track.label

    @property
    def name(self):
        """ Unique name of this segment. """
        return self.clip_id + "-" + str(self.track_id) + "-" + str(self.start_frame)

    @property
    def frame_velocity(self):
        # tracking frame velocity for each frame.
        return self.track.frame_velocity[
            self.start_frame : self.start_frame + self.frames
        ]

    @property
    def track_bounds(self):
        # original location of this tracks bounds.
        return self.track.track_bounds[
            self.start_frame : self.start_frame + self.frames
        ]

    @property
    def frame_crop(self):
        # how much each frame has been cropped.
        return self.track.frame_crop[self.start_frame : self.start_frame + self.frames]

    @property
    def frame_temp_median(self):
        # thermal reference temperature for each frame (i.e. which temp is 0)
        return self.track.frame_temp_median[
            self.start_frame : self.start_frame + self.frames
        ]

    @property
    def end_frame(self):
        """ end frame of segment"""
        return self.start_frame + self.frames

    @property
    def track_bin(self):
        """ Unique name of this segments track. """
        return self.track.bin_id

    def __str__(self):
        return "{0} label {1} offset:{2} weight:{3:.1f}".format(
            self.unique_track_id, self.label, self.start_frame, self.weight
        )


class Preprocessor:
    """ Handles preprocessing of track data. """

    # size to scale each frame to when loaded.
    FRAME_SIZE = 48

    MIN_SIZE = 4

    @staticmethod
    def apply(
        frames,
        reference_level=None,
        frame_velocity=None,
        augment=False,
        encode_frame_offsets_in_flow=False,
        default_inset=0,
        keep_aspect=False,
        flip=None,
    ):
        """
        Preprocesses the raw track data, scaling it to correct size, and adjusting to standard levels
        :param frames: a list of np array of shape [C, H, W]
        :param reference_level: thermal reference level for each frame in data
        :param frame_velocity: velocity (x,y) for each frame.
        :param augment: if true applies a slightly random crop / scale
        :param default_inset: the default number of pixels to inset when no augmentation is applied.
        """

        # -------------------------------------------
        # first we scale to the standard size

        # adjusting the corners makes the algorithm robust to tracking differences.
        # gp changed to 0,1 maybe should be a percent of the frame size
        top_offset = random.randint(0, 1) if augment else default_inset
        bottom_offset = random.randint(0, 1) if augment else default_inset
        left_offset = random.randint(0, 1) if augment else default_inset
        right_offset = random.randint(0, 1) if augment else default_inset

        data = np.zeros(
            (
                len(frames),
                len(frames[0]),
                Preprocessor.FRAME_SIZE,
                Preprocessor.FRAME_SIZE,
            ),
            dtype=np.float32,
        )

        for i, frame in enumerate(frames):

            channels, frame_height, frame_width = frame.shape

            if (
                frame_height < Preprocessor.MIN_SIZE
                or frame_width < Preprocessor.MIN_SIZE
            ):
                return

            frame_bounds = tools.Rectangle(0, 0, frame_width, frame_height)

            # set up a cropping frame
            crop_region = tools.Rectangle.from_ltrb(
                left_offset,
                top_offset,
                frame_width - right_offset,
                frame_height - bottom_offset,
            )

            # if the frame is too small we make it a little larger
            while crop_region.width < Preprocessor.MIN_SIZE:
                crop_region.left -= 1
                crop_region.right += 1
                crop_region.crop(frame_bounds)
            while crop_region.height < Preprocessor.MIN_SIZE:
                crop_region.top -= 1
                crop_region.bottom += 1
                crop_region.crop(frame_bounds)

            cropped_frame = frame[
                :,
                crop_region.top : crop_region.bottom,
                crop_region.left : crop_region.right,
            ]

            if keep_aspect:

                height, width = cropped_frame[0].shape
                min_dim = max(width, height)
                scale = Preprocessor.FRAME_SIZE / min_dim
                target_size = (round(height * scale), round(width * scale))
                scaled_frames = data[i]
                for channel, scaled_frame in enumerate(scaled_frames):
                    cropped_data = cropped_frame[channel]
                    w, h = cropped_data.shape
                    if w > Preprocessor.FRAME_SIZE or h > Preprocessor.FRAME_SIZE:
                        cropped_data = cv2.resize(
                            cropped_data,
                            dsize=target_size,
                            interpolation=cv2.INTER_LINEAR
                            if channel != TrackChannels.mask
                            else cv2.INTER_NEAREST,
                        )

                    w, h = cropped_data.shape
                    scaled_frame[:w, :h] = cropped_data
                continue
                # print("target is", target_size)
            else:
                target_size = (Preprocessor.FRAME_SIZE, Preprocessor.FRAME_SIZE)
            scaled_frame = [
                cv2.resize(
                    cropped_frame[channel],
                    dsize=target_size,
                    interpolation=cv2.INTER_LINEAR
                    if channel != TrackChannels.mask
                    else cv2.INTER_NEAREST,
                )
                for channel in range(channels)
            ]
            height, width = scaled_frame[0].shape
            max_dim = max(height, width)
            if keep_aspect and max_dim > Preprocessor.FRAME_SIZE:

                crop_start = 0
                if augment:
                    crop = max_dim - Preprocessor.FRAME_SIZE
                    crop_start = random.randint(0, crop)
                crop_start = 0
                for i, channel in enumerate(scaled_frame):
                    if height > width:
                        scaled_frame[i] = channel[
                            crop_start : crop_start + Preprocessor.FRAME_SIZE, :
                        ]

                    else:
                        scaled_frame[i] = channel[
                            :, crop_start : crop_start + Preprocessor.FRAME_SIZE
                        ]
                scaled_frame = np.asarray(scaled_frame)

            data[i] = scaled_frame
        # convert back into [F,C,H,W] array.
        # data = np.float32(scaled_frames)

        # -------------------------------------------
        # next adjust temperature and flow levels
        # get reference level for thermal channel
        if reference_level is not None:
            assert len(data) == len(
                reference_level
            ), "Reference level shape and data shape not match."
            ref_avg = np.average(reference_level)

            # reference thermal levels to the reference level
            data[:, 0, :, :] -= np.float32(reference_level)[:, np.newaxis, np.newaxis]

        # map optical flow down to right level,
        # we pre-multiplied by 256 to fit into a 16bit int
        data[:, 2 : 3 + 1, :, :] *= 1.0 / 256.0

        # write frame motion into center of frame
        if encode_frame_offsets_in_flow:
            F, C, H, W = data.shape
            for x in range(-2, 2 + 1):
                for y in range(-2, 2 + 1):
                    data[:, 2 : 3 + 1, H // 2 + y, W // 2 + x] = frame_velocity[:, :]

        # set filtered track to delta frames
        reference = np.clip(data[:, 0], 20, 999)
        # data[0, 1] = 0
        # data[1:, 1] = reference[1:] - reference[:-1]
        flipped = False
        # -------------------------------------------
        # finally apply and additional augmentation
        if augment:
            if random.random() <= 0.75:
                # we will adjust contrast and levels, but only within these bounds.
                # that is a bright input may have brightness reduced, but not increased.
                LEVEL_OFFSET = 4

                # apply level and contrast shift
                level_adjust = random.normalvariate(0, LEVEL_OFFSET)
                contrast_adjust = tools.random_log(0.9, (1 / 0.9))

                data[:, 0] *= contrast_adjust
                data[:, 0] += level_adjust
                # gp will put back in but want to keep same for now so can test objectively
                # augment filtered, no need for brightness, as will normalize anyway
                data[:, 1] *= contrast_adjust
                # data[:, 1] += level_adjust
            if flip or (flip is None and random.random() <= 0.50):
                flipped = True
                # when we flip the frame remember to flip the horizontal velocity as well
                data = np.flip(data, axis=3)
                data[:, 2] = -data[:, 2]

        np.clip(data[:, 0, :, :], a_min=0, a_max=None, out=data[:, 0, :, :])
        return data, flipped


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

    # number of pixels to inset from frame edges by default
    DEFAULT_INSET = 2

    def __init__(
        self,
        track_db: TrackDatabase,
        name="Dataset",
        config=None,
        use_segments=True,
        use_predictions=False,
    ):
        self.camera_bins = {}
        self.use_segments = use_segments
        # database holding track data
        self.db = track_db
        self.label_mapping = None
        self.original_segments = None
        # name of this dataset
        self.name = name
        self.use_predictions = use_predictions
        self.original_samples = None
        # list of our tracks
        self.tracks = []
        self.tracks_by_label = {}
        self.tracks_by_bin = {}
        self.tracks_by_id = {}
        self.camera_names = set()
        self.cameras_by_id = {}

        # writes the frame motion into the center of the optical flow channels
        self.encode_frame_offsets_in_flow = False

        # cumulative distribution function for segments.  Allows for super fast weighted random sampling.
        self.segment_cdf = []
        self.segment_label_cdf = {}
        # segments list
        self.segments = []
        self.segments_by_label = {}

        self.frame_cdf = []
        self.frame_label_cdf = {}

        self.frame_samples = []
        self.clips_to_samples = {}
        self.frames_by_label = {}

        # list of label names
        self.labels = []

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
        self.label_caps = {}
        # a copy of our entire dataset, if loaded.
        self.X = None
        self.y = None

        if config:
            self.min_frame_mass = 3
            self.segment_length = config.build.segment_length
            # number of seconds segments are spaced apart
            self.segment_spacing = config.build.segment_spacing
            self.banned_clips = config.build.banned_clips
            self.included_labels = config.labels
            self.clip_before_date = config.build.clip_end_date
            self.segment_min_mass = config.build.train_min_mass
        else:
            # number of seconds each segment should be
            self.segment_length = 3
            # number of seconds segments are spaced apart
            self.segment_spacing = 1
        self.filtered_stats = {
            "confidence": 0,
            "trap": 0,
            "banned": 0,
            "date": 0,
            "tags": 0,
            "segment_mass": 0,
        }

    def set_read_only(self, read_only):
        self.db.set_read_only(read_only)

    @property
    def sample_count(self):
        return len(self.samples())

    def samples(self):
        if self.use_segments:
            return self.segments
        return self.frame_samples

    def set_samples_for(self, label, samples):
        if self.use_segments:
            self.segments_by_label[label] = samples
        else:
            self.frames_by_label[label] = samples

    def get_label_caps(self, labels, remapped=False):
        counts = []
        for label in labels:
            counts.append(len(self.samples_for(label, remapped=remapped)))
        index = math.floor(len(counts) * 0.40)
        counts.sort()
        birds = self.samples_for("bird", remapped=remapped)
        if len(birds) > 0:
            return len(birds)
        # return 4096

        return counts[index]
        # return int(np.percentile(counts, 25))

    def samples_for(self, label, remapped=False):
        labels = []
        if remapped and self.label_mapping:
            labels = [
                key for key, mapped in self.label_mapping.items() if mapped == label
            ]
            labels.sort()
        else:
            labels.append(label)
        samples = []
        for label in labels:
            if self.use_segments:
                samples.extend(self.segments_by_label.get(label, []))
            else:
                samples.extend(self.frames_by_label.get(label, []))
        return samples

    def get_counts(self, label):
        """
        Gets number of examples for given label
        :label: label to check
        :return: (segments, tracks, bins, weight)
        """
        segments = 0
        tracks = 0
        bins = 0
        weight = 0
        frames = 0
        if self.label_mapping:
            for key, value in self.label_mapping.items():
                if key == label or value == label:
                    label_tracks = self.tracks_by_label.get(key, [])
                    tracks += len(label_tracks)
                    segments += sum(len(track.segments) for track in label_tracks)
                    frames += sum(
                        len(track.get_sample_frames()) for track in label_tracks
                    )

        else:
            label_tracks = self.tracks_by_label.get(label, [])
            segments = sum(len(track.segments) for track in label_tracks)
            weight = self.get_label_weight(label)
            tracks = len(label_tracks)
            frames = sum(len(track.get_sample_frames()) for track in label_tracks)
            bins = len(
                [
                    tracks
                    for bin_name, tracks in self.tracks_by_bin.items()
                    if len(tracks) > 0 and tracks[0].label == label
                ]
            )
        return segments, frames, tracks, bins, weight

    def set_read_only(self, read_only):
        self.db.set_read_only(read_only)

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
        if (
            not disable_async
            and self.preloader_queue is not None
            and not force_no_augmentation
        ):
            # get samples from queue
            batch_X = []
            batch_y = []
            for _ in range(n):
                X, y = self.preloader_queue.get()
                batch_X.append(X[0])
                batch_y.append(y[0])

            return np.asarray(batch_X), np.asarray(batch_y)

        segments = self.sample_segments(n)

        batch_X = []
        batch_y = []

        for segment in segments:
            data = self.fetch_segment(
                segment, augment=self.enable_augmentation and not force_no_augmentation
            )
            batch_X.append(data)
            batch_y.append(self.labels.index(segment.label))

            if np.isnan(data).any():
                logging.warning("NaN found in data from source: %r", segment.clip_id)

        # Half float should be fine here.  When using process based async loading we have to pickle the batch between
        # processes, so having it half the size helps a lot.  Also it reduces the memory required for the read buffers
        batch_X = np.float16(batch_X)
        batch_y = np.int32(batch_y)

        return batch_X, batch_y

    def load_tracks(self, shuffle=False, before_date=None, after_date=None):
        """
        Loads track headers from track database with optional filter
        :return: [number of tracks added, total tracks].
        """
        labels = self.db.get_labels()
        counter = 0
        track_ids = self.db.get_all_track_ids(
            before_date=before_date, after_date=after_date
        )
        if shuffle:
            np.random.shuffle(track_ids)
        for clip_id, track_id in track_ids:
            if self.load_track(clip_id, track_id, labels):
                counter += 1
        return [counter, len(track_ids)]

    def add_tracks(self, tracks, max_segments_per_track=None):
        """
        Adds list of tracks to dataset
        :param tracks: list of TrackHeader
        :param track_filter: optional filter
        """
        result = 0
        for track in tracks:
            if self.add_track_header(track, max_segments_per_track):
                result += 1
        return result

    def add_track_header(self, track_header, max_segments_per_track=None):
        if track_header.bin_id in self.tracks_by_bin:
            return False

        # gp test less segments more tracks
        if max_segments_per_track is not None:
            if len(track_header.segments) > max_segments_per_track:
                segments = random.sample(track_header.segments, max_segments_per_track)
                track_header.segments = segments

        self.tracks.append(track_header)
        self.add_track_to_mappings(track_header)
        self.segments.extend(track_header.segments)
        # samples = track_header.get_sample_frames()
        # self.frame_samples.extend(samples)
        return True

    def load_track(self, clip_id, track_id, labels):
        """
        Creates segments for track and adds them to the dataset
        :param clip_id: id of tracks clip
        :param track_id: track number
        :param track_filter: if provided a function filter(clip_meta, track_meta) that returns true when a track should
                be ignored)
        :return: True if track was added, false if it was filtered out.
        :return:
        """

        # make sure we don't already have this track
        if "{}-{}".format(clip_id, track_id) in self.tracks_by_bin:
            return False

        clip_meta = self.db.get_clip_meta(clip_id)
        track_meta = self.db.get_track_meta(clip_id, track_id)
        predictions = self.db.get_track_predictions(clip_id, track_id)
        if self.filter_track(clip_meta, track_meta):
            return False
        track_header = TrackHeader.from_meta(
            clip_id, clip_meta, track_meta, predictions
        )
        self.tracks.append(track_header)

        track_data = self.db.get_track(
            clip_id, track_id, channel=TrackChannels.filtered
        )

        # if self.important_frames:
        track_header.set_important_frames(
            labels, self.min_frame_mass, self.use_predictions, filtered_data=track_data
        )
        segment_frame_spacing = round(
            self.segment_spacing * track_header.frames_per_second
        )
        segment_width = round(self.segment_length * track_header.frames_per_second)
        if len(track_header.important_frames) > segment_width / 3.0:
            track_header.calculate_segments(
                track_meta["mass_history"],
                segment_frame_spacing,
                segment_width,
                self.segment_min_mass,
            )

        self.filtered_stats["segment_mass"] += track_header.filtered_stats[
            "segment_mass"
        ]
        self.segments.extend(track_header.segments)
        self.add_track_to_mappings(track_header)

        return True

    def filter_track(self, clip_meta, track_meta):
        # some clips are banned for various reasons
        source = os.path.basename(clip_meta["filename"])
        if self.banned_clips and source in self.banned_clips:
            self.filtered_stats["banned"] += 1
            return True
        if "tag" not in track_meta:
            self.filtered_stats["tags"] += 1
            return True
        if track_meta["tag"] not in self.included_labels:
            self.filtered_stats["tags"] += 1
            return True

        # filter by date
        if (
            self.clip_before_date
            and dateutil.parser.parse(clip_meta["start_time"]).date()
            > self.clip_before_date.date()
        ):
            self.filtered_stats["date"] += 1
            return True

        # always let the false-positives through as we need them even though they would normally
        # be filtered out.
        if track_meta["tag"] == "false-positive":
            return False

        # for some reason we get some records with a None confidence?
        if track_meta.get("confidence", 0.0) <= 0.6:
            self.filtered_stats["confidence"] += 1
            return True

        # remove tracks of trapped animals
        if (
            "trap" in clip_meta.get("event", "").lower()
            or "trap" in clip_meta.get("trap", "").lower()
        ):
            self.filtered_stats["trap"] += 1
            return True

        return False

    def add_track_to_mappings(self, track_header):
        if self.label_mapping and track_header.label in self.label_mapping:
            track_header.label = self.mapped_label(track_header.label)

        self.tracks_by_id[track_header.unique_id] = track_header
        bins = self.tracks_by_bin.setdefault(track_header.bin_id, [])
        bins.append(track_header)

        if track_header.label not in self.tracks_by_label:

            self.labels.append(track_header.label)
            self.tracks_by_label[track_header.label] = []
        self.tracks_by_label[track_header.label].append(track_header)
        segs = self.segments_by_label.setdefault(track_header.label, [])
        segs.extend(track_header.segments)
        frames = self.frames_by_label.setdefault(track_header.label, [])
        samples = track_header.get_sample_frames()
        self.frame_samples.extend(samples)
        frames.extend(samples)
        camera = self.cameras_by_id.setdefault(
            track_header.camera_id, Camera(track_header.camera_id)
        )
        self.camera_names.add(track_header.camera_id)
        camera.add_track(track_header)

    def filter_segments(self, avg_mass, ignore_labels=None):
        """
        Removes any segments with an average mass less than the given avg_mass
        :param avg_mass: segments with less avarage mass per frame than this will be removed from the dataset.
        :param ignore_labels: these labels will not be filtered
        :return: number of segments removed
        """

        num_filtered = 0
        new_segments = []

        for segment in self.segments:

            pass_mass = segment.avg_mass >= avg_mass
            if (not ignore_labels and segment.label in ignore_labels) or (pass_mass):
                new_segments.append(segment)
            else:
                num_filtered += 1

        self.segments = new_segments

        self._purge_track_segments()

        self.rebuild_cdf()

        return num_filtered

    def fetch_all(self):
        """
        Fetches all segments
        :return: X of shape [n,f,channels,height,width], y of shape [n]
        """
        X = np.float32([self.fetch_segment(segment) for segment in self.segments])
        y = np.int32([self.labels.index(segment.label) for segment in self.segments])
        return X, y

    def fetch_track(self, track: TrackHeader, original=False, preprocess=True):
        """
        Fetches data for an entire track
        :param track: the track to fetch
        :return: segment data of shape [frames, channels, height, width]
        """
        data = self.db.get_track(track.clip_id, track.track_id, original=original)
        if preprocess:
            data = Preprocessor.apply(
                data,
                reference_level=track.frame_temp_median,
                frame_velocity=track.frame_velocity,
                encode_frame_offsets_in_flow=self.encode_frame_offsets_in_flow,
                default_inset=self.DEFAULT_INSET,
            )
        return data

    def fetch_frame(self, frame_sample, channels=None):
        data, label = self.db.get_frame(
            frame_sample.clip_id,
            frame_sample.track_id,
            frame_sample.frame_num,
            channels=channels,
        )
        label = self.mapped_label(label)
        return data, label

    def fetch_sample(self, sample, augment=False, channels=None):
        if isinstance(sample, SegmentHeader):
            label = sample.label
            if self.label_mapping:
                label = self.mapped_label(sample.label)
            return self.fetch_segment(sample, augment), label
        return self.fetch_frame(sample, channels=channels)

    def fetch_segment(
        self, segment: SegmentHeader, augment=False, frames=None, preprocess=True
    ):
        """
        Fetches data for segment.
        :param segment: The segment header to fetch
        :param augment: if true applies data augmentation
        :return: segment data of shape [frames, channels, height, width]
        """
        segment_width = round(self.segment_length * segment.track.frames_per_second)

        # if we are requesting a segment smaller than the default segment size take it from the middle.
        unused_frames = segment.frames - segment_width
        if unused_frames < 0:
            raise Exception(
                "Maximum segment size for the dataset is {} frames, but requested {}".format(
                    segment.frames, segment_width
                )
            )
        first_frame = segment.start_frame + (unused_frames // 2)
        last_frame = segment.start_frame + (unused_frames // 2) + segment_width
        if unused_frames != 0:
            raise "Unused frame"
        if augment and unused_frames > 0:
            # jitter first frame
            prev_frames = first_frame
            post_frames = segment.track.frames - 1 - last_frame
            max_jitter = max(5, unused_frames)
            jitter = np.clip(
                np.random.randint(-max_jitter, max_jitter), -prev_frames, post_frames
            )
        else:
            jitter = 0
        first_frame += jitter
        last_frame += jitter
        if frames:
            data = frames[first_frame:last_frame]
        else:
            data = self.db.get_track(
                segment.clip_id, segment.track.track_id, first_frame, last_frame
            )

        if len(data) != segment_width:
            logging.error(
                "invalid segment length %d, expected %d", len(data), len(segment_width)
            )
        if preprocess:
            data = Preprocessor.apply(
                data,
                segment.track.frame_temp_median[first_frame:last_frame],
                segment.track.frame_velocity[first_frame:last_frame],
                augment=augment,
                default_inset=self.DEFAULT_INSET,
            )
            return data
        else:
            return data

    def set_samples(self, cap_at=None, label_cap=None):
        samples = self.epoch_samples(
            cap_at=cap_at, cap_samples=True, label_cap=label_cap
        )
        self.segments_by_label = {}
        for seg in samples:
            segs = self.segments_by_label.setdefault(seg.track.label, [])
            segs.append(seg)
        self.segments = samples
        self.rebuild_cdf()

    def epoch_samples(
        self, cap_samples=None, replace=True, random=True, cap_at=None, label_cap=None
    ):
        if len(self.labels) == 0:
            return []
        labels = self.labels.copy()
        samples = []
        if cap_samples and label_cap is None:
            if cap_at:
                label_cap = len(self.samples_for(cap_at))
            else:
                label_cap = self.get_label_caps(labels, remapped=True)
        cap = None
        for label in labels:
            if cap_samples:
                cap = min(label_cap, len(self.samples_for(label, remapped=True)))
            new = self.get_sample(cap=cap, replace=replace, label=label, random=random)
            if new is not None and len(new) > 0:
                samples.extend(new)
        labels = [sample.label for sample in samples]
        # print(self.name, "sample is", labels)
        return samples

    def cdf(self):
        if self.use_segments:
            return self.segment_cdf
        return self.frame_cdf

    def label_cdf(self, label):
        if self.use_segments:
            return self.segment_label_cdf.get(label, [])
        return self.frame_label_cdf.get(label, [])

    def get_sample(self, cap=None, replace=True, label=None, random=True):
        """ Returns a random frames from weighted list. """
        if label:
            samples = self.samples_for(label, remapped=True)
            cdf = self.label_cdf(label)
        else:
            samples = self.samples()
            cdf = self.cdf()
        if not samples:
            return None
        if cap is None:
            return samples
        if random:
            return np.random.choice(samples, cap, replace=replace, p=cdf)
        else:
            cap = min(cap, len(samples))
            return samples[:cap]

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

        label_weight = {}
        mean_label_weight = 0

        for label in self.labels:
            label_weight[label] = self.get_label_weight(label)
            mean_label_weight += label_weight[label] / len(self.labels)

        scale_factor = {}
        for label in self.labels:
            modifier = (
                1.0 if weight_modifiers is None else weight_modifiers.get(label, 1.0)
            )
            if label_weight[label] == 0:
                scale_factor[label] = 1.0
            else:
                scale_factor[label] = mean_label_weight / label_weight[label] * modifier

        for segment in self.segments:
            segment.weight *= scale_factor.get(segment.label, 1.0)
        self.rebuild_cdf()

    def balance_labels_and_remove(
        self,
        labels=None,
        exclude_labels=None,
        high_tracks_first=False,
        use_segments=True,
    ):
        if labels is None:
            labels = self.labels
        if exclude_labels:
            for label in exclude_labels:
                if label in labels:
                    labels.remove(label)

        total = 0
        for label in labels:
            if use_segments:
                total += len(self.segments_by_label.get(label, []))
            else:
                total += len(self.frames_by_label.get(label, []))

        # this could be done smarter
        label_percent = 1 / len(self.labels)
        label_cap = label_percent * total
        for label in labels:
            label_tracks = self.tracks_by_label.get(label, []).copy()
            if use_segments:
                label_samples = len(self.segments_by_label.get(label, []))
            else:
                label_samples = len(self.frames_by_label.get(label, []))
                # label_segments = sum([len(track.segments) for track in label_tracks])
            samples_to_remove = label_samples - label_cap
            samples_removed = 0
            if high_tracks_first:
                tracks_by_segments = {}
                for track in label_tracks:
                    if use_segments:
                        samples = len(track.segments)
                    else:
                        samples = len(track.get_sample_frames())
                    seg_tracks = tracks_by_segments.setdefault(samples, [])
                    seg_tracks.append(track)
                seg_counts = list(tracks_by_segments.keys())
                seg_counts.sort(reverse=True)

            while samples_removed < samples_to_remove:
                if high_tracks_first:
                    seg_count = seg_counts[0]
                    high_tracks = tracks_by_segments[seg_count]
                    track = random.choice(high_tracks)
                    high_tracks.remove(track)
                    seg_tracks = tracks_by_segments.setdefault(seg_count - 1, [])
                    seg_tracks.append(track)
                    insert_at = 1
                    if len(high_tracks) == 0:
                        insert_at = 0
                        del seg_counts[0]
                    if len(seg_tracks) == 1:
                        seg_counts.insert(insert_at, seg_count - 1)
                else:
                    track = np.random.choice(label_tracks)

                if use_segments:
                    if len(track.segments) == 0:
                        label_tracks.remove(track)
                        continue
                    seg = random.choice(track.segments)
                    track.segments.remove(seg)
                    self.segments.remove(seg)
                    self.segments_by_label[label].remove(seg)
                    if len(track.segments) == 0:
                        label_tracks.remove(track)
                else:
                    if len(track.get_sample_frames()) == 0:
                        label_tracks.remove(track)
                        continue
                    frame = random.choice(track.get_sample_frames())
                    track.important_frames.remove(frame)
                    self.frame_samples.remove(frame)
                    self.frames_by_label[label].remove(frame)
                    if len(track.get_sample_frames()) == 0:
                        label_tracks.remove(track)
                samples_removed += 1

    def balance_bins(self, max_bin_weight=None):
        """
        Adjusts weights so that bins with a number number of segments aren't sampled so frequently.
        :param max_bin_weight: bins with more weight than this number will be scaled back to this weight.
        """

        for bin_name, tracks in self.tracks_by_bin.items():
            bin_weight = sum(track.weight for track in tracks)
            if bin_weight == 0:
                continue
            if max_bin_weight is None:
                scale_factor = 1 / bin_weight
                # means each bin has equal possiblity
            elif bin_weight > max_bin_weight:
                scale_factor = max_bin_weight / bin_weight
            else:
                scale_factor = 1
            for track in tracks:
                for segment in track.segments:
                    segment.weight *= scale_factor
        self.rebuild_cdf()

    def get_bin_segments_count(self, bin_id):
        return sum(len(track.segments) for track in self.tracks_by_bin[bin_id])

    def get_bin_max_track_duration(self, bin_id):
        return max(track.duration for track in self.tracks_by_bin[bin_id])

    def is_heavy_bin(self, bin_id, max_bin_segments, max_validation_set_track_duration):
        """
        heavy bins are bins with more tracks whiche xceed track duration or max bin_segments
        """
        bin_segments = self.get_bin_segments_count(bin_id)
        max_track_duration = self.get_bin_max_track_duration(bin_id)
        return (
            bin_segments > max_bin_segments
            or max_track_duration > max_validation_set_track_duration
        )

    def split_heavy_bins(
        self, bins, max_bin_segments, max_validation_set_track_duration
    ):
        """
        heavy bins are bins with more tracks whiche xceed track duration or max bin_segments
        """
        heavy_bins, normal_bins = [], []
        for bin_id in bins:
            if bin_id in self.tracks_by_bin:
                if self.is_heavy_bin(
                    bin_id, max_bin_segments, max_validation_set_track_duration
                ):
                    heavy_bins.append(bin_id)
                else:
                    normal_bins.append(bin_id)
        return normal_bins, heavy_bins

    def balance_resample(self, required_samples, weight_modifiers=None):
        """ Removes segments until all classes have given number of samples (or less)"""

        new_segments = []

        for label in self.labels:
            segments = self.get_label_segments(label)
            required_label_samples = required_samples
            if weight_modifiers:
                required_label_samples = int(
                    math.ceil(required_label_samples * weight_modifiers.get(label, 1.0))
                )
            if len(segments) > required_label_samples:
                # resample down
                segments = np.random.choice(
                    segments, required_label_samples, replace=False
                ).tolist()
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
        self.segments = [
            segment for segment in self.segments if segment.label != label_to_remove
        ]
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

        sample = (
            self.segments
            if n is None or n >= len(self.segments)
            else random.sample(self.segments, n)
        )

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
        first_moment = np.sum(first_moment, axis=(1, 2)) / (
            len(sample) * width * height
        )
        second_moment = np.sum(second_moment, axis=(1, 2)) / (
            len(sample) * width * height
        )

        mu = first_moment
        var = second_moment + (mu ** 2) - (2 * mu * first_moment)

        normalisation_constants = [(mu[i], math.sqrt(var[i])) for i in range(channels)]

        return normalisation_constants

    def mapped_label(self, label):
        if self.label_mapping:
            return self.label_mapping.get(label, label)
        return label

    def rebuild_frame_cdf(self, balance_labels=False, lbl_p=None):
        self.frame_cdf = []
        total = 0
        self.frame_label_cdf = {}

        for track in self.tracks:
            for frame in track.important_frames:
                frame_weight = track.frame_weight
                if lbl_p and track.label in lbl_p:
                    frame_weight *= lbl_p[track.label]
                total += frame_weight

                self.frame_cdf.append(frame_weight)

                cdf = self.frame_label_cdf.setdefault(track.label, [])
                cdf.append(track.frame_weight)

        if len(self.frame_cdf) > 0:
            self.frame_cdf = [x / total for x in self.frame_cdf]

        for key, cdf in self.frame_label_cdf.items():
            total = sum(cdf)
            self.frame_label_cdf[key] = [x / total for x in cdf]

        if self.label_mapping:
            mapped_cdf = {}
            labels = list(self.label_mapping.keys())
            labels.sort()
            for label in labels:
                if label not in self.frame_label_cdf:
                    continue
                label_cdf = self.frame_label_cdf[label]
                new_label = self.label_mapping[label]
                if balance_labels:
                    if lbl_p and label in lbl_p:
                        label_cdf = np.float64(label_cdf)
                        label_cdf *= lbl_p[label]
                cdf = mapped_cdf.setdefault(new_label, [])
                cdf.extend(label_cdf)

            for key, cdf in mapped_cdf.items():
                total = sum(cdf)
                mapped_cdf[key] = [x / total for x in cdf]
            self.frame_label_cdf = mapped_cdf

    def rebuild_cdf(self, balance_labels=False):
        """Calculates the CDF used for fast random sampling for frames and
        segments, if balance labels is set each label has an equal chance of
        being chosen
        """
        p = {
            "bird": 20,
            "possum": 20,
            "rodent": 20,
            "hedgehog": 20,
            "cat": 5,
            "insect": 1,
            "leporidae": 5,
            "mustelid": 5,
            "false-positive": 1,
            "wallaby": 10,
        }
        self.rebuild_segment_cdf(balance_labels=balance_labels, lbl_p=p)
        self.rebuild_frame_cdf(balance_labels=balance_labels, lbl_p=p)

    def rebuild_segment_cdf(self, balance_labels=False, lbl_p=None):
        """ Calculates the CDF used for fast random sampling """
        self.segment_cdf = []
        total = 0
        self.segment_label_cdf = {}
        for segment in self.segments:
            seg_weight = segment.weight
            if lbl_p and segment.track.label in lbl_p:
                seg_weight *= lbl_p[segment.track.label]
            total += seg_weight
            self.segment_cdf.append(seg_weight)
            label = segment.track.label
            cdf = self.segment_label_cdf.setdefault(label, [])
            cdf.append(segment.weight)
        if len(self.segment_cdf) > 0:
            self.segment_cdf = [x / total for x in self.segment_cdf]
        for key, cdf in self.segment_label_cdf.items():
            total = sum(cdf)
            if total > 0:
                self.segment_label_cdf[key] = [x / total for x in cdf]
            else:
                self.segment_label_cdf[key] = np.zeros((cdf.shape))
        # do this after so labels are balanced
        if self.label_mapping:
            mapped_cdf = {}
            labels = list(self.label_mapping.keys())
            labels.sort()
            for label in labels:
                if label not in self.frame_label_cdf:
                    continue
                label_cdf = self.segment_label_cdf[label]
                new_label = self.label_mapping[label]
                if balance_labels:
                    if lbl_p and label in lbl_p:
                        label_cdf = np.float64(label_cdf)
                        label_cdf *= lbl_p[label]
                cdf = mapped_cdf.setdefault(new_label, [])
                cdf.extend(label_cdf)

            for key, cdf in mapped_cdf.items():
                total = sum(cdf)

                mapped_cdf[key] = [x / total for x in cdf]
            self.segment_label_cdf = mapped_cdf

    def get_label_weight(self, label):
        """ Returns the total weight for all segments of given label. """
        tracks = self.tracks_by_label.get(label)
        return sum(track.weight for track in tracks) if tracks else 0

    def get_label_segments_count(self, label):
        """ Returns the total weight for all segments of given class. """
        tracks = self.tracks_by_label.get(label, [])
        result = sum([len(track.segments) for track in tracks])
        return result

    def get_label_segments(self, label):
        """ Returns the total weight for all segments of given class. """
        result = []
        for track in self.tracks_by_label.get(label, []):
            result.extend(track.segments)
        return result

    def start_async_load(self, buffer_size=128):
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
            self.preloader_threads = [
                multiprocessing.Process(
                    target=preloader, args=(self.preloader_queue, self)
                )
                for _ in range(self.WORKER_THREADS)
            ]
        else:
            self.preloader_queue = queue.Queue(buffer_size)
            self.preloader_threads = [
                threading.Thread(target=preloader, args=(self.preloader_queue, self))
                for _ in range(self.WORKER_THREADS)
            ]

        self.preloader_stop_flag = False
        for thread in self.preloader_threads:
            thread.start()

    def stop_async_load(self):
        """
        Stops async worker thread.
        """
        if self.preloader_threads is not None:
            for thread in self.preloader_threads:
                if hasattr(thread, "terminate"):
                    # note this will corrupt the queue, so reset it
                    thread.terminate()
                    self.preloader_queue = None
                else:
                    thread.exit()

    #
    # def resample(self, keep_all, ratio=1.1):
    #     if self.original_segments is None:
    #         self.original_segments = self.segments
    #
    #     self.segments = [
    #         sample for sample in self.original_segments if sample.label == keep_all
    #     ]
    #
    #     total_size = len(self.segments) * 1.1
    #     # labels to samples isn't updated when binarized, but cna be used for finding data labels
    #     other_labels = [
    #         sample.label
    #         for sample in self.original_segments
    #         if sample.label != keep_all
    #     ]
    #     other_labels = set(other_labels)
    #     amount_per = int(total_size / len(other_labels))
    #     other_labels = list(other_labels)
    #     np.random.shuffle(other_labels)
    #     for label in other_labels:
    #         segments = [
    #             sample for sample in self.original_segments if sample.label == label
    #         ]
    #         take = min(len(segments), amount_per)
    #         new_segments = np.random.choice(segments, take, replace=False)
    #         self.segments.extend(new_segments)
    #         logging.debug("Resample %s taking %s", len(new_segments), label)

    def binarize(
        self,
        set_one,
        lbl_one,
        set_two=None,
        lbl_two="other",
        scale=True,
        keep_fp=False,
        remove_labels=None,
        balance_labels=True,
        shuffle=True,
    ):
        set_one_count = 0
        self.label_mapping = {}
        for label in set_one:
            samples = len(self.samples_for(label))
            set_one_count += samples
            self.label_mapping[label] = lbl_one

        if set_two is None:
            set_two = set(self.labels) - set(set_one)
        if remove_labels:
            for label in remove_labels:
                set_two.discard(label)
        self.labels = [lbl_one, lbl_two]
        if keep_fp:
            set_two.discard("false-positive")
            self.label_mapping["false-positive"] = "false-positive"
            self.labels.append("false-positive")

        set_two_count = 0
        for label in set_two:
            samples = len(self.samples_for(label))
            set_two_count += samples
            self.label_mapping[label] = lbl_two
        percent = 1
        percent2 = 1
        if scale:
            if set_two_count > set_one_count:
                percent2 = set_one_count / set_two_count
                # allow 10% more
                if self.name != "validation":
                    percent2 += 0.05
                percent2 = min(1, percent2)
            else:
                percent = set_two_count / set_one_count
                percent += 0.1
                percent = min(1, percent)
        # set_one_cap = set_one_count / len()
        tracks_by_id, new_samples = self.rebalance(
            cap_percent=percent, labels=set_one, shuffle=shuffle
        )
        tracks_by_id2, new_samples2 = self.rebalance(
            cap_percent=percent2, labels=set_two, shuffle=shuffle
        )
        self.tracks_by_bin = tracks_by_id
        samples = new_samples
        self.tracks_by_bin.update(tracks_by_id2)
        samples.extend(new_samples2)
        if keep_fp:
            tracks_by_id3, new_samples3 = self.rebalance(
                label_cap=int(set_one_count * 0.5), labels=["false-positive"]
            )
            self.tracks_by_bin.update(tracks_by_id3)
            samples.extend(new_samples3)
        if self.use_segments:
            self.segments = new_samples
        else:
            self.frame_samples = new_samples
        if shuffle:
            np.random.shuffle(self.segments)
            np.random.shuffle(self.frame_samples)
        self.rebuild_cdf(balance_labels=balance_labels)

    def rebalance(
        self,
        label_cap=None,
        cap_percent=None,
        exclude=[],
        labels=None,
        update=False,
        shuffle=True,
    ):
        new_samples = []
        tracks_by_id = {}
        tracks = []
        if labels is None:
            labels = self.labels.copy()

        for label in labels:
            samples = self.samples_for(label)
            if len(samples) == 0:
                continue
            label_samples = []
            self.set_samples_for(label, label_samples)
            if label in exclude:
                labels.remove(label)
            track_ids = set()
            if shuffle:
                np.random.shuffle(samples)
            if label_cap:
                samples = samples[:label_cap]
            if cap_percent:
                new_length = int(len(samples) * cap_percent)
                samples = samples[:new_length]
            label_tracks = self.tracks_by_label.get(label, [])
            for track in label_tracks:
                if self.use_segments:
                    track.segments = []
                else:
                    track.important_frames = []
            for sample in samples:
                track = self.tracks_by_id[sample.unique_track_id]
                track_ids.add(track)
                track.add_sample(sample, self.use_segments)
                tracks_by_id[track.bin_id] = track
                label_samples.append(sample)
            self.tracks_by_label[label] = list(track_ids)
            new_samples.extend(label_samples)

        if update:
            self.tracks_by_bin = tracks_by_id
            if self.use_segments:
                self.segments = new_segments
            else:
                self.frame_samples = new_samples
        return tracks_by_id, new_samples

    def has_data(self):
        if self.use_segments:
            return len(self.segments) > 0
        else:
            return len(self.frame_samples) > 0

    def random_segments(self, balance_labels=True):
        self.segments = []
        self.segments_by_label = {}

        for track in self.tracks:
            segment_frame_spacing = round(
                self.segment_spacing * track.frames_per_second
            )
            segment_width = round(self.segment_length * track.frames_per_second)

            track.calculate_segments(
                track.frame_mass,
                segment_frame_spacing,
                segment_width,
                self.segment_min_mass,
                random=True,
            )
            self.segments.extend(track.segments)
            segs = self.segments_by_label.setdefault(track.label, [])
            segs.extend(track.segments)
        self.rebuild_cdf(balance_labels=balance_labels)


# continue to read examples until queue is full
def preloader(q, dataset):
    """ add a segment into buffer """
    logging.info(
        " -started async fetcher for %s with augment=%s segment_width=%s",
        dataset.name,
        dataset.enable_augmentation,
        dataset.segment_width,
    )
    loads = 0
    timer = time.time()
    while not dataset.preloader_stop_flag:
        if not q.full():
            q.put(dataset.next_batch(1, disable_async=True))
            loads += 1
            if (time.time() - timer) > 1.0:
                # logging.debug("{} segments per seconds {:.1f}".format(dataset.name, loads / (time.time() - timer)))
                loads = 0
        else:
            time.sleep(0.1)


def get_cropped_fraction(region: tools.Rectangle, width, height):
    """ Returns the fraction regions mass outside the rect ((0,0), (width, height)"""
    bounds = tools.Rectangle(0, 0, width - 1, height - 1)
    return 1 - (bounds.overlap_area(region) / region.area)


def dataset_db_path(config):
    return os.path.join(config.tracks_folder, "datasets.dat")


def filtered_is_valid(filtered):
    area = filtered.shape[0] * filtered.shape[1]
    percentile = int(100 - 100 * 16.0 / area)
    threshold = np.percentile(filtered, percentile)
    threshold = max(0, threshold - 40)
    num_less = len(filtered[filtered <= threshold])
    if num_less <= area * 0.05:
        return False
    return True
