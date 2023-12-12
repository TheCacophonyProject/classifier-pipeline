import math
import cv2
import json
import dateutil
import numpy as np
import logging
from ml_tools import tools
from track.region import Region
from abc import ABC, abstractmethod
from ml_tools.imageprocessing import resize_cv, rotate, normalize, resize_and_pad
from ml_tools.frame import Frame, TrackChannels
from ml_tools import imageprocessing
import datetime
from enum import Enum
import attr

FRAMES_PER_SECOND = 9

CPTV_FILE_WIDTH = 160
CPTV_FILE_HEIGHT = 120
FRAME_SIZE = 32
MIN_SIZE = 4


class SegmentType(Enum):
    IMPORTANT_RANDOM = 0
    ALL_RANDOM = 1
    IMPORTANT_SEQUENTIAL = 2
    ALL_SEQUENTIAL = 3
    TOP_SEQUENTIAL = 4
    ALL_SECTIONS = 5
    TOP_RANDOM = 6
    ALL_RANDOM_NOMIN = 7


class BaseSample(ABC):
    @property
    @abstractmethod
    def track_bounds(cls):
        """Get all regions for this sample"""
        ...

    @property
    @abstractmethod
    def source_file(self):
        """Source file."""
        ...

    @property
    @abstractmethod
    def track_median_mass(self):
        """The function gets median mass of the track."""
        ...

    @property
    @abstractmethod
    def frame_indices(self):
        """The function gets all frames indices for this sample."""
        ...

    @property
    @abstractmethod
    def unique_track_id(self):
        """Represent the unique identifier for this track."""
        ...

    @property
    @abstractmethod
    def unique_id(self):
        """Represent the unique identifier for this sample."""
        ...

    @property
    @abstractmethod
    def bin_id(self):
        """Represent the unique identifier for this sample."""
        ...

    @property
    @abstractmethod
    def sample_weight(self):
        """Represent the unique identifier for this sample."""
        ...

    @property
    @abstractmethod
    def mass(cls):
        """Get mass for this sample"""
        ...


class Sample(BaseSample):
    def __init__(self, label):
        self.original_label = label
        self.remapped_label = label

    @property
    def label(self):
        return self.remapped_label


EDGE = 1

res_x = 120
res_y = 160


@attr.s
class ClipHeader:
    clip_id = attr.ib()
    location = attr.ib()
    station_id = attr.ib()
    rec_time = attr.ib()
    source_file = attr.ib()
    frames_per_second = attr.ib()
    camera = attr.ib()
    events = attr.ib()
    trap = attr.ib()
    tracks = attr.ib()
    ffc_frames = attr.ib()
    frame_temp_median = attr.ib(default=None)

    def get_samples(self):
        samples = []
        for t in self.tracks:
            samples.extend(t.samples)
        return samples


class TrackHeader:
    """Header for track."""

    def __init__(
        self,
        clip_id,
        track_id,
        label,
        num_frames,
        regions,
        start_frame,
        ffc_frames=None,
        sample_frames_indices=None,
        station_id=None,
        rec_time=None,
        source_file=None,
        camera=None,
        confidence=None,
        human_tags=None,
        remapped_lbl=None,
        mega_missed_regions=None,
    ):
        # regions that megadetector found nothing in
        self.mega_missed_regions = mega_missed_regions
        self.station_id = station_id
        self.clip_id = clip_id
        self.source_file = source_file
        self.human_tags = human_tags
        self.confidence = confidence
        self.filtered_stats = {"segment_mass": 0}
        # reference to clip this segment came from
        # reference to track this segment came from
        self.track_id = track_id
        # label for this track
        # date and time of the start of the track
        self.start_frame = np.uint16(start_frame)

        self.original_label = label
        self.remapped_label = remapped_lbl

        # thermal reference point for each frame.
        # self.frame_temp_median = np.uint16(frame_temp_median)
        # tracking frame movements for each frame, array of tuples (x-vel, y-vel)
        self.frame_velocity = None
        # original tracking bounds
        self.regions_by_frame = regions
        # what fraction of pixels are from out of bounds
        self.frame_crop = None
        self.num_frames = num_frames
        self.important_predicted = 0

        mass_history = np.uint16(
            [region.mass for region in self.regions_by_frame.values()]
        )
        mass_history = [
            region.frame_number
            for region in self.regions_by_frame.values()
            if region.mass > 0
            and (
                ffc_frames is None
                or skip_ffc is False
                or region.frame_number not in ffc_frames
            )
            and not region.blank
            and region.width > 0
            and region.height > 0
        ]
        self.has_no_mass = np.sum(mass_history) == 0
        if len(mass_history) == 0:
            self.lower_mass = 0
            self.upper_mass = 0
            self.median_mass = 0
            self.mean_mass = 0
        else:
            self.lower_mass = np.uint16(np.percentile(mass_history, q=25))
            self.upper_mass = np.uint16(np.percentile(mass_history, q=75))
            self.median_mass = np.uint16(np.median(mass_history))
            self.mean_mass = np.uint16(np.mean(mass_history))
        self.samples = []

    @property
    def label(self):
        return (
            self.original_label if self.remapped_label is None else self.remapped_label
        )

    @property
    def bounds_history(self):
        regions = list(self.regions_by_frame.values())
        regions = sorted(regions, key=lambda r: r.frame_number)
        return regions

    def toJSON(self, clip_meta):
        meta_dict = {}
        ffc_frames = clip_meta.get("ffc_frames", [])
        meta_dict["ffc_frames"] = ffc_frames
        meta_dict["clip_id"] = int(self.clip_id)
        meta_dict["track_id"] = int(self.track_id)
        meta_dict["camera"] = self.camera
        meta_dict["num_frames"] = int(self.num_frames)
        meta_dict["frames_per_second"] = int(self.frames_per_second)
        meta_dict["track_bounds"] = self.regions
        meta_dict["start_frame"] = int(self.start_frame)
        if self.location is not None:
            try:
                meta_dict["location_hash"] = "{}{}".format(
                    hash(self.location[0]), hash(self.location[1])
                )
            except:
                pass
        meta_dict["label"] = self.label

        return json.dumps(meta_dict, indent=3, cls=tools.CustomJSONEncoder)

    def add_sample(self, sample):
        self.samples.append(sample)

    def calculate_sample_frames(
        self, min_mass=None, max_mass=None, ffc_frames=None, skip_last=None
    ):
        frame_numbers = list(self.regions_by_frame.keys())
        if skip_last is not None:
            skip_x = int(len(frame_numbers) * skip_last)
            frame_numbers = frame_numbers[:-skip_x]
        frame_numbers = [
            frame
            for frame in frame_numbers
            if (ffc_frames is None or frame not in ffc_frames)
            and (
                self.mega_missed_regions is None
                or frame not in self.mega_missed_regions
            )
        ]
        frame_numbers.sort()

        for frame_num in frame_numbers:
            region = self.regions_by_frame[frame_num]
            if region.mass == 0 or region.blank:
                continue
            if min_mass is not None and region.mass < min_mass:
                continue
            if max_mass is not None and region.mass > max_mass:
                continue
            f = FrameSample(
                self.clip_id,
                self.track_id,
                region.frame_number,
                self.label,
                None,
                region,
                source_file=self.source_file,
                weight=1,
                station_id=self.station_id,
                track_median_mass=self.median_mass,
            )
            self.samples.append(f)

    def remove_sample(self, f):
        self.samples.remove(f)

    #
    # def get_sample_frame(self, i=0, remove=False):
    #     if len(self.sample_frames) == 0:
    #         return None
    #     f = self.sample_frames[i]
    #
    #     if remove:
    #         del self.sample_frames[i]
    #     return f

    @property
    def frame_weight(self):
        return 1 / self.num_sample_frames

    #
    # @property
    # def num_samples(self):
    #     return len(self.sample_frames)

    def calculate_frame_crop(self):
        # frames are always square, but bounding rect may not be, so to see how much we clipped I need to create a square
        # bounded rect and check it against frame size.
        self.frame_crop = []
        for rect in self.regions:
            rx, ry = rect.mid_x, rect.mid_y
            size = max(rect.width, rect.height)
            adjusted_rect = tools.Rectangle(rx - size / 2, ry - size / 2, size, size)
            self.frame_crop.append(
                get_cropped_fraction(adjusted_rect, self.res_x, self.res_y)
            )

    def calculate_velocity(self):
        frame_center = [region.mid for region in self.track_bounds]
        self.frame_velocity = []
        prev = None
        for x, y in frame_center:
            if prev is None:
                self.frame_velocity.append((0.0, 0.0))
            else:
                self.frame_velocity.append((x - prev[0], y - prev[1]))
            prev = (x, y)

    def get_segments(
        self,
        segment_width,
        segment_frame_spacing=9,
        segment_type=SegmentType.ALL_RANDOM,
        segment_min_mass=None,
        repeats=1,
        max_segments=None,
        dont_filter=False,
        skip_ffc=True,
        ffc_frames=None,
        location=None,
        segment_frames=None,
        from_last=None,
    ):
        if segment_frames is not None:
            raise Exception("Have not implement this path")
        min_frames = segment_width
        if self.label == "vehicle" or self.label == "human":
            min_frames = segment_width / 4.0

        # in python3.7+ can just take the values and it guarantees order it was added to dict
        regions = self.bounds_history
        self.samples, self.filtered_stats = get_segments(
            self.clip_id,
            self.track_id,
            self.start_frame,
            segment_frame_spacing=segment_frame_spacing,
            segment_width=segment_width,
            label=self.label,
            regions=np.array(regions),
            segment_min_mass=segment_min_mass,
            ffc_frames=ffc_frames,
            lower_mass=self.lower_mass,
            repeats=repeats,
            min_frames=min_frames,
            segment_type=segment_type,
            max_segments=max_segments,
            station_id=self.station_id,
            source_file=self.source_file,
            dont_filter=dont_filter,
            skip_ffc=skip_ffc,
        )
        # GP could get this from the tracks when writing
        # but might be best to keep samples independent for ease
        for s in self.samples:
            s._track_median_mass = self.median_mass
        return self.samples

    @property
    def camera_id(self):
        """Unique name of this track."""
        return "{}-{}".format(self.camera, self.location)

    @property
    def bin_id(self):
        """Unique name of this track."""
        return "{}-{}".format(self.clip_id, self.location_id)

    @property
    def weight(self):
        """Returns total weight for all segments in this track"""
        return sum(segment.weight for segment in self.segments)

    @property
    def unique_id(self):
        return "{}-{}".format(self.clip_id, self.track_id)

    @staticmethod
    def from_meta(source_file, clip_id, clip_meta, track_meta, predictions=None):
        """Creates a track header from given metadata."""
        correct_prediction = track_meta.get("correct_prediction", None)
        rec_time = dateutil.parser.parse(clip_meta["start_time"])
        # end_time = dateutil.parser.parse(track_meta["end_time"])
        location = clip_meta.get("location")
        camera = clip_meta.get("device_id")
        # clip_meta["device"]
        station_id = clip_meta.get("station_id")
        frames_per_second = clip_meta.get("frames_per_second", FRAMES_PER_SECOND)
        # get the reference levels from clip_meta and load them into the track.
        track_start_frame = int(round(track_meta["start_frame"]))
        track_end_frame = int(round(track_meta["end_frame"]))
        num_frames = track_end_frame - track_start_frame + 1
        frame_temp_median = np.float32(
            clip_meta["frame_temp_median"][
                track_start_frame : num_frames + track_start_frame
            ]
        )

        ffc_frames = clip_meta.get("ffc_frames", [])
        sample_frames = None
        # sample_frames = track_meta.get("sample_frames")
        # if sample_frames is not None:
        #     sample_frames = sample_frames + track_start_frame
        regions = {}
        f_i = 0
        first_region = None
        for bounds in track_meta["regions"]:
            r = Region.region_from_array(bounds)
            if first_region is None:
                first_region = r
            regions[r.frame_number] = r
            f_i += 1
        if track_start_frame != first_region.frame_number:
            logging.warn(
                "Have wrong region or start frame for %s clip %s track %s adjusting regions first region %s becomes %s",
                source_file,
                clip_id,
                track_meta["id"],
                first_region.frame_number,
                track_start_frame,
            )

            region_list = list(regions.values())
            regions = {}
            frame_number = track_start_frame
            for r in region_list:
                r = Region.region_from_array(bounds)
                r.frame_number = frame_number
                frame_number += 1
                regions[r.frame_number] = r
        header = TrackHeader(
            clip_id=int(clip_id),
            track_id=int(track_meta["id"]),
            label=track_meta["human_tag"],
            num_frames=num_frames,
            location=location,
            regions=regions,
            frame_temp_median=frame_temp_median,
            frames_per_second=frames_per_second,
            predictions=predictions,
            correct_prediction=correct_prediction,
            start_frame=track_start_frame,
            res_x=clip_meta.get("res_x", CPTV_FILE_WIDTH),
            res_y=clip_meta.get("res_y", CPTV_FILE_HEIGHT),
            ffc_frames=ffc_frames,
            sample_frames_indices=sample_frames,
            station_id=station_id,
            rec_time=rec_time,
            source_file=source_file,
            camera=camera,
        )
        return header

    def __repr__(self):
        return self.unique_id

    def get_id(self):
        return self.track_id


class Camera:
    def __init__(self, camera):
        self.label_to_bins = {}
        self.label_to_tracks = {}
        self.label_to_samples = {}

        self.bins = {}
        self.camera = camera
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
        if len(track.sample_frames) == 0 or f is None:
            del bins[self.bin_i]
            del self.bins[bin_id]

        return track, f

    def label_track_count(
        self,
        label,
        max_segments_per_track=None,
    ):
        if label not in self.label_to_tracks:
            return 0
        tracks = self.label_to_tracks[label].values()
        return len(tracks)

    def label_segment_count(
        self,
        label,
        max_segments_per_track=None,
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
        self,
        label,
        max_frames_per_track=None,
    ):
        if label not in self.label_to_tracks:
            return 0
        tracks = self.label_to_tracks[label].values()
        frames = 0
        for track in tracks:
            if max_frames_per_track:
                frames += max(len(track.sample_frames), max_frames_per_track)
            else:
                frames += len(track.sample_frames)

        return frames

    def remove_label(self, label):
        if label in self.label_to_tracks:
            tracks = list(self.label_to_tracks[label].values())
            while len(tracks) > 0:
                track = tracks.pop()
                self.remove_track(track)
            del self.label_to_tracks[label]
            del self.label_to_bins[label]
            del self.label_frames[label]

    def remove_track(self, track):
        self.segments -= 1
        self.segment_sum -= len(track.segments)
        del self.label_to_tracks[track.label][track.unique_id]
        if track.bin_id in self.label_to_bins[track.label]:
            self.label_to_bins[track.label].remove(track.bin_id)
        self.label_frames[track.label] -= len(track.sample_frames)

    def add_track(self, track_header):
        tracks = self.label_to_tracks.setdefault(track_header.label, {})
        tracks[track_header.unique_id] = track_header
        if track_header.bin_id not in self.bins:
            self.bins[track_header.bin_id] = []

        if track_header.label not in self.label_to_bins:
            self.label_to_bins[track_header.label] = []
            self.label_frames[track_header.label] = 0

        if track_header.bin_id not in self.label_to_bins[track_header.label]:
            self.label_to_bins[track_header.label].append(track_header.bin_id)

        self.bins[track_header.bin_id].append(track_header)
        self.label_frames[track_header.label] += len(track_header.sample_frames)

        segment_length = len(track_header.segments)
        self.segment_sum += segment_length
        self.segments += 1

    def remove_sample(self, sample):
        del self.label_to_samples[sample.label][sample.unique_id]
        if track.bin_id in self.label_to_bins[sample.label]:
            self.label_to_bins[sample.label].remove(sample.bin_id)
        self.label_frames[sample.label] -= len(sample.sample_frames)

    def add_sample(self, sample):
        samples = self.label_to_samples.setdefault(sample.label, {})
        samples[sample.unique_id] = sample
        if sample.bin_id not in self.bins:
            self.bins[sample.bin_id] = []

        if sample.label not in self.label_to_bins:
            self.label_to_bins[sample.label] = []
            self.label_frames[sample.label] = 0

        if sample.bin_id not in self.label_to_bins[sample.label]:
            self.label_to_bins[sample.label].append(sample.bin_id)

        self.bins[sample.bin_id].append(sample)


class FrameSample(Sample):
    _frame_id = 1

    def __init__(
        self,
        clip_id,
        track_id,
        frame_num,
        label,
        velocity,
        region,
        weight,
        source_file,
        augment=False,
        track_median_mass=None,
        station_id=None,
    ):
        super().__init__(label)
        self.clip_id = clip_id
        self.id = FrameSample._frame_id
        FrameSample._frame_id += 1

        self.track_id = track_id
        self.frame_number = frame_num
        self.velocity = velocity
        self.region = region
        self.weight = weight
        self.augment = augment
        self._source_file = source_file
        self._track_median_mass = track_median_mass
        self.station_id = station_id

    @property
    def filtered(self):
        return False

    @property
    def track_median_mass(self):
        return self._track_median_mass

    @property
    def source_file(self):
        return self._source_file

    def copy(self):
        f = FrameSample(
            clip_id=self.clip_id,
            track_id=self.track_id,
            frame_num=self.frame_number,
            label=self.label,
            temp_median=self.temp_median,
            velocity=self.velocity,
            region=self.region,
            weight=self.weight,
            camera=self.camera,
            start_time=self.start_time,
            augment=self.augment,
            source_file=self.source_file,
            station_id=self.station_id,
            rec_time=self.rec_time,
            track_median_mass=self.track_median_mass,
        )
        return f

    @property
    def frame_numbers(self):
        return [self.frame_number]

    @property
    def mass(self):
        return self.region.mass

    @property
    def sample_weight(self):
        return self.weight

    @property
    def unique_track_id(self):
        return "{}-{}".format(self.clip_id, self.track_id)

    @property
    def track_bounds(self):
        return [self.region]

    @property
    def frame_indices(self):
        return [self.frame_number]

    @property
    def unique_id(self):
        return f"{self.clip_id}-{self.track_id}-{self.frame_number}"

    @property
    def bin_id(self):
        """Unique name of this segments track."""
        # break into 50 frame keys, since we dont have much data this means multiple sets can have same clip
        i = int(self.frame_number / 50)
        return f"{self.clip_id}"


class SegmentHeader(Sample):
    """Header for segment."""

    _segment_id = 1

    def __init__(
        self,
        clip_id,
        track_id,
        start_frame,
        frames,
        weight,
        mass,
        label,
        regions,
        frame_indices=None,
        movement_data=None,
        best_mass=False,
        top_mass=False,
        start_time=None,
        camera=None,
        location=None,
        station_id=None,
        rec_time=None,
        source_file=None,
        filtered=False,
        track_median_mass=None,
    ):
        super().__init__(label)
        self.filtered = filtered
        self.rec_time = rec_time
        self.location = location
        self.station_id = station_id
        self.movement_data = movement_data
        self.top_mass = top_mass
        self.best_mass = best_mass
        self.id = SegmentHeader._segment_id
        SegmentHeader._segment_id += 1
        # reference to track this segment came from
        self.clip_id = clip_id
        self.track_id = track_id
        self.frame_numbers = np.uint16(frame_indices)
        self.start_time = start_time
        self.regions = regions

        self.start_frame = start_frame
        # length of segment in frames
        self.frames = np.uint16(frames)
        # relative weight of the segment (higher is sampled more often)
        self.weight = np.float16(weight)

        self._mass = np.uint16(mass)
        self.camera = camera
        self._source_file = source_file
        self._track_median_mass = track_median_mass

    @property
    def track_median_mass(self):
        return self._track_median_mass

    @property
    def source_file(self):
        return self._source_file

    @property
    def mass(self):
        return self._mass

    @property
    def sample_weight(self):
        return self.weight

    @property
    def track_bounds(self):
        return self.regions

    @property
    def frame_indices(self):
        return self.frame_numbers

    @property
    def avg_mass(self):
        return self.mass / self.frames

    @property
    def unique_track_id(self):
        # reference to clip this segment came from
        return "{}-{}".format(self.clip_id, self.track_id)

    @property
    def name(self):
        """Unique name of this segment."""
        return self.clip_id + "-" + str(self.track_id) + "-" + str(self.start_frame)

    @property
    def frame_velocity(self):
        # tracking frame velocity for each frame.
        return self.track.frame_velocity[
            self.start_frame : self.start_frame + self.frames
        ]

    @property
    def frame_crop(self):
        # how much each frame has been cropped.
        return self.track.frame_crop[self.start_frame : self.start_frame + self.frames]

    @property
    def end_frame(self):
        """end frame of segment"""
        return self.start_frame + self.frames

    @property
    def bin_id(self):
        """Unique name of this segments track."""
        return f"{self.station_id}"

    def __str__(self):
        return "{0} label {1} offset:{2} weight:{3:.1f}".format(
            self.unique_track_id, self.label, self.start_frame, self.weight
        )

    @property
    def unique_id(self):
        return self.id

    def get_data(self, db):
        crop_rectangle = tools.Rectangle(2, 2, 160 - 2 * 2, 140 - 2 * 2)

        try:
            background = db.get_clip_background(self.clip_id)
            frames = db.get_track(
                self.clip_id,
                self.track_id,
                original=False,
                frame_numbers=self.frame_numbers - self.start_frame,
            )

            thermals = []  # np.empty(len(frames), dtype=object)
            filtered = []  # np.empty(len(frames), dtype=object)

            for i, frame in enumerate(frames):
                frame.float_arrays()
                frame.filtered = frame.thermal - frame.region.subimage(background)
                temp_index = np.where(self.frame_numbers == frame.frame_number)[0][0]
                temp = self.frame_temp_median[temp_index]
                frame.resize_with_aspect((32, 32), crop_rectangle, keep_edge=True)
                frame.thermal -= temp
                np.clip(frame.thermal, a_min=0, a_max=None, out=frame.thermal)

                frame.thermal, stats = imageprocessing.normalize(
                    frame.thermal, new_max=255
                )
                if not stats[0]:
                    frame.thermal = np.zeros((frame.thermal.shape))
                    # continue
                frame.filtered, stats = imageprocessing.normalize(
                    frame.filtered, new_max=255
                )
                if not stats[0]:
                    frame.filtered = np.zeros((frame.filtered.shape))

                # continue
                # frame.filtered =
                filtered.append(frame.filtered)
                thermals.append(frame.thermal)
            thermals = np.array(thermals)
            filtered = np.array(filtered)
            # thermal, success = imageprocessing.square_clip(thermals, 5, (32, 32))
            # if not success:
            #     logging.warn("Error making thermal square clip %s", self.clip_id)
            #     return None
            # filtered, success = imageprocessing.square_clip(filtered, 5, (32, 32))
            # if not success:
            #     logging.warn("Error making filtered square clip %s", filtered)
            #
            #     return None
        except:
            logging.error("Cant get segment %s", self, exc_info=True)
            raise "EX"
            return None
        return thermals, filtered


def get_cropped_fraction(region: tools.Rectangle, width, height):
    """Returns the fraction regions mass outside the rect ((0,0), (width, height)"""
    bounds = tools.Rectangle(0, 0, width - 1, height - 1)
    return 1 - (bounds.overlap_area(region) / region.area)


def get_movement_data(regions):
    areas = np.array([region.area for region in regions])
    centrex = np.array([region.mid_x for region in regions])
    centrey = np.array([region.mid_y for region in regions])
    xv = np.hstack((0, centrex[1:] - centrex[:-1]))
    yv = np.hstack((0, centrey[1:] - centrey[:-1]))
    axv = xv / areas**0.5
    ayv = yv / areas**0.5
    bounds = np.array([region.to_ltrb() for region in regions])
    mass = np.array([region.mass for region in regions])

    return np.hstack((bounds, np.vstack((mass, xv, yv, axv, ayv)).T))


# should use trackheader / track in creationg interface out common properties
def get_segments(
    clip_id,
    track_id,
    start_frame,
    regions,
    segment_frame_spacing=9,
    segment_width=25,
    label=None,
    segment_min_mass=None,
    ffc_frames=[],
    lower_mass=0,
    repeats=1,
    min_frames=None,
    segment_frames=None,
    segment_type=SegmentType.ALL_RANDOM,
    max_segments=None,
    location=None,
    station_id=None,
    camera=None,
    rec_time=None,
    source_file=None,
    dont_filter=False,
    skip_ffc=True,
):
    if segment_type == SegmentType.ALL_RANDOM_NOMIN:
        segment_min_mass = None
    if min_frames is None:
        min_frames = 25
    segments = []
    mass_history = np.uint16([region.mass for region in regions])
    filtered_stats = {"segment_mass": 0, "too short": 0}

    has_no_mass = np.sum(mass_history) == 0
    frame_indices = [
        region.frame_number
        for region in regions
        if (has_no_mass or region.mass > 0)
        and (
            ffc_frames is None
            or skip_ffc is False
            or region.frame_number not in ffc_frames
        )
        and not region.blank
        and region.width > 0
        and region.height > 0
    ]
    if len(frame_indices) == 0:
        logging.warn("Nothing to load for %s - %s", clip_id, track_id)
        return [], filtered_stats
    if segment_min_mass is not None:
        segment_min_mass = min(
            segment_min_mass,
            np.median(mass_history[frame_indices - start_frame]),
        )
    else:
        segment_min_mass = 1
        # remove blank frames

    if segment_type == SegmentType.TOP_RANDOM:
        # take top 50 mass frames
        frame_indices = sorted(
            frame_indices,
            key=lambda f_i: mass_history[f_i - start_frame],
            reverse=True,
        )
        frame_indices = frame_indices[:50]
        frame_indices.sort()
    if segment_type == SegmentType.TOP_SEQUENTIAL:
        return get_top_mass_segments(
            clip_id,
            track_id,
            label,
            camera,
            segment_width,
            segment_frame_spacing,
            mass_history,
            ffc_frames,
            regions,
            start_frame,
            lower_mass,
            segment_min_mass,
            source_file=source_file,
        )
    # if len(frame_indices) < min_frames:
    # filtered_stats["too short"] += 1
    # return segments, filtered_stats
    frame_indices = np.array(frame_indices)
    segment_count = max(1, len(frame_indices) // segment_frame_spacing)
    segment_count = int(segment_count)
    if max_segments is not None:
        segment_count = min(max_segments, segment_count)
    # take any segment_width frames, this could be done each epoch
    whole_indices = frame_indices
    random_frames = segment_type in [
        SegmentType.IMPORTANT_RANDOM,
        SegmentType.ALL_RANDOM,
        SegmentType.ALL_RANDOM_NOMIN,
        SegmentType.TOP_RANDOM,
        None,
    ]
    for _ in range(repeats):
        frame_indices = whole_indices.copy()
        if random_frames:
            # random_frames and not random_sections:
            np.random.shuffle(frame_indices)
        for i in range(segment_count):
            # always get atleast one segmnet
            if i > 0:
                if (len(frame_indices) < segment_width and len(segments) > 1) or len(
                    frame_indices
                ) < (segment_width / 4.0):
                    break

            if segment_type == SegmentType.ALL_SECTIONS:
                # random frames from section 2.2 * segment_width
                section = frame_indices[: int(segment_width * 2.2)]
                indices = np.random.choice(
                    len(section),
                    min(segment_width, len(section)),
                    replace=False,
                )
                frames = section[indices]
                frame_indices = frame_indices[segment_frame_spacing:]
            elif random_frames:
                # frame indices already randomized so just need to grab some
                frames = frame_indices[:segment_width]
                frame_indices = frame_indices[segment_width:]
            else:
                segment_start = i * segment_frame_spacing
                segment_end = segment_start + segment_width
                segment_end = min(len(frame_indices), segment_end)
                frames = frame_indices[segment_start:segment_end]

            remaining = segment_width - len(frames)
            # sample another same frames again if need be
            if remaining > 0:
                extra_frames = np.random.choice(
                    frames,
                    min(remaining, len(frames)),
                    replace=False,
                )
                frames = np.concatenate([frames, extra_frames])
            frames.sort()
            relative_frames = frames - start_frame
            mass_slice = mass_history[relative_frames]
            segment_mass = np.sum(mass_slice)
            segment_avg_mass = segment_mass / len(mass_slice)
            filtered = False
            if segment_min_mass and segment_avg_mass < segment_min_mass:
                if dont_filter:
                    filtered = True
                else:
                    filtered_stats["segment_mass"] += 1
                    continue

            # temp_slice = frame_temp_median[relative_frames]
            region_slice = regions[relative_frames]
            movement_data = None
            if segment_avg_mass < 50:
                segment_weight_factor = 0.75
            elif segment_avg_mass < 100:
                segment_weight_factor = 1
            else:
                segment_weight_factor = 1.2

            for z, f in enumerate(frames):
                assert region_slice[z].frame_number == f
            segment = SegmentHeader(
                clip_id,
                track_id,
                start_frame=start_frame,
                frames=segment_width,
                weight=segment_weight_factor,
                mass=segment_mass,
                label=label,
                regions=region_slice,
                frame_indices=frames,
                movement_data=movement_data,
                camera=camera,
                location=location,
                station_id=station_id,
                rec_time=rec_time,
                source_file=source_file,
                filtered=filtered,
            )
            segments.append(segment)
    return segments, filtered_stats


def get_top_mass_segments(
    clip_id,
    track_id,
    label,
    camera,
    segment_width,
    segment_frame_spacing,
    mass_history,
    ffc_frames,
    regions,
    start_frame,
    lower_mass,
    segment_min_mass,
    source_file=None,
):
    # Needs re testing if want to use
    logging.warning("Top mass segments hasn't been tested may not work")
    filtered_stats = {"segment_mass": 0, "too short": 0}

    segments = []
    segment_count = max(1, len(regions) // segment_frame_spacing)
    segment_count = int(segment_count)

    segment_mass = []
    for i in range(max(1, len(mass_history) - segment_width)):
        contains_ffc = False
        for z in range(segment_width):
            if (z + i + start_frame) in ffc_frames:
                contains_ffc = True
                break
        if contains_ffc:
            continue
        mass = np.sum(mass_history[i : i + segment_width])
        segment_mass.append((i, mass))

    sorted_mass = sorted(segment_mass, key=lambda x: x[1], reverse=True)
    best_mass = True
    segment_count = max(1, len(regions) // segment_frame_spacing)
    segment_count = int(min(len(sorted_mass), segment_count))

    for _ in range(segment_count):
        segment_info = sorted_mass[0]
        index = segment_info[0]
        avg_mass = segment_info[1] / segment_width
        if not best_mass and (avg_mass < lower_mass or avg_mass < segment_min_mass):
            break
        movement_data = get_movement_data(regions[index : index + segment_width])
        width = min(segment_width, len(regions))
        frames = np.arange(width) + index
        segment = SegmentHeader(
            clip_id,
            track_id,
            start_frame=start_frame,
            frames=segment_width,
            weight=1,
            mass=segment_info[1],
            label=label,
            regions=regions[frames],
            frame_indices=frames + start_frame,
            movement_data=movement_data,
            best_mass=best_mass,
            top_mass=True,
            camera=camera,
            source_file=source_file,
        )
        best_mass = False
        segments.append(segment)
        # remove those that start within this segment
        sorted_mass = [
            mass_info
            for mass_info in sorted_mass
            if mass_info[0] <= (index - segment_width / 3 * 2)
            or mass_info[0] >= (index + segment_width / 3 * 2)
        ]
        if len(segments) == segment_count or len(sorted_mass) == 0:
            break
    return segments, filtered_stats


# GP Not using at the moment, but handy for more complex training


class TrackingSample(Sample):
    _s_id = 1

    def __init__(
        self,
        clip_id,
        track_id,
        frame_num,
        labels,
        temp_median,
        region,
        camera,
        filename,
    ):
        self.id = TrackingSample._s_id
        TrackingSample._s_id += 1
        self.clip_id = clip_id
        self.filename = filename
        self.track_id = track_id
        # needed for original frame will change this GP
        self.frame_number = frame_num
        self.camera = camera
        if isinstance(labels, str):
            self.labels = [labels]
        else:
            self.labels = labels
        self.temp_median = temp_median
        if region is None:
            self.regions = []
        else:
            self.regions = [region]

    # asusming only one label for now GP should change
    @property
    def label(self):
        return self.labels[0]

    @property
    def region(self):
        return self.regions[0]

    def add_sample(self, f):
        self.labels.append(f.label)
        self.regions.append(f.region)

    def filename(self):
        return f"{self.clip_id}-{self.track_id}-{self.frame_number}"

    @classmethod
    def from_sample(cls, f):
        return cls(
            f.clip_id, f.track_id, f.frame_number, [f.label], f.temp_median, [f.region]
        )

    @property
    def unique_track_id(self):
        return "{}-{}".format(self.clip_id, self.track_id)

    @property
    def track_bounds(self):
        return self.regions

    @property
    def frame_indices(self):
        return [self.frame_number]

    @property
    def sample_weight(self):
        return 1

    @property
    def unique_id(self):
        return f"{self.clip_id}-{self.track_id}-{self.frame_number}"

    @property
    def bin_id(self):
        """Unique name of this segments track."""
        frames_numbers = int(self.frame_number / 20)
        # try and segment framesso 20 frames are in the same bin
        return f"{self.clip_id}-{self.track_id}-{frames_numbers}"
        # this should be used but dont have much data
        # return f"{self.clip_id}-{self.track_id}"
