import json
import dateutil
import numpy as np
import logging
import os
from ml_tools.imageprocessing import filtered_is_valid
from ml_tools import tools
from ml_tools.frame import Frame
from ml_tools.preprocess import MIN_SIZE
from track.track import TrackChannels

FRAMES_PER_SECOND = 9

CPTV_FILE_WIDTH = 160
CPTV_FILE_HEIGHT = 120


class NumpyMeta:
    def __init__(self, filename):
        self.filename = filename
        self.track_info = {}
        self.f = None
        self.mode = "rb"

    def __enter__(self):
        self.open(self.mode)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self, mode="rb"):
        if self.f is not None:
            return
        self.f = open(f"{self.filename}", mode)

    def close(self):
        if self.f is not None:
            self.f.close()
        self.f = None

    def save_tracks(self, db, tracks):
        logging.info("Writing %s tracks to %s", len(tracks), self.filename)
        self.open(mode="wb")
        try:
            for track in tracks:
                self.add_track(db, track, save_flow=False)
        except:
            logging.error("Error saving track info", exc_info=True)
        finally:
            self.close()
        # self.read_tracks(tracks)

    def read_tracks(self, tracks):
        self.open(mode="rb")
        for track in tracks:
            self.f.seek(track.track_info["background"])
            back = np.load(self.f)
            for frame_i in range(len(track.track_bounds)):
                frame_info = track.track_info[frame_i]
                # print("reading first at", track.track_id, frame_i, frame_info)
                self.f.seek(frame_info[TrackChannels.thermal])
                thermal = np.load(self.f)

        self.close()

    def add_track(self, db, track, save_flow=True):
        background = db.get_clip_background(track.clip_id)

        track_info = {}
        track_info["background"] = self.f.tell()
        np.save(self.f, background, allow_pickle=False)
        self.track_info[track.unique_id] = track_info
        frames = db.get_track(
            track.clip_id,
            track.track_id,
            original=False,
        )
        for frame in frames:
            frame_info = {}
            track_info[frame.frame_number] = frame_info
            frame_info[TrackChannels.thermal] = self.f.tell()
            np.save(self.f, frame.thermal, allow_pickle=False)
            if save_flow:
                frame.flow = np.float32(frame.flow)
                if frame.flow is not None and frame.flow_clipped:
                    frame.flow *= 1.0 / 256.0
                    frame.flow_clipped = False

                frame_info[TrackChannels.flow] = self.f.tell()
                np.save(self.f, frame.flow, allow_pickle=False)
        track.track_info = track_info


class TrackHeader:
    """Header for track."""

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
        res_x=CPTV_FILE_WIDTH,
        res_y=CPTV_FILE_HEIGHT,
        ffc_frames=None,
        important_frames=None,
    ):
        self.res_x = res_x
        self.res_y = res_y
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
        # camera this track came fromsegment
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
        self.frame_crop = None
        self.num_frames = num_frames
        self.frames_per_second = frames_per_second
        self.important_frames = None
        self.important_predicted = 0
        self.frame_mass = frame_mass
        self.lower_mass = np.percentile(frame_mass, q=25)
        self.upper_mass = np.percentile(frame_mass, q=75)
        self.median_mass = np.median(frame_mass)
        self.mean_mass = np.mean(frame_mass)
        self.ffc_frames = ffc_frames

        if important_frames is not None:
            self.important_frames = []
            for frame_num in important_frames:
                f = FrameSample(
                    self.clip_id, self.track_id, frame_num, self.label, None, None
                )
                self.important_frames.append(f)
        else:
            self.important_frames = []
            for frame_num, mass in enumerate(self.frame_mass):
                if mass == 0:
                    continue
                f = FrameSample(
                    self.clip_id, self.track_id, frame_num, self.label, None, None
                )
                self.important_frames.append(f)
        self.track_info = None

    def toJSON(self, clip_meta):
        meta_dict = {}
        ffc_frames = clip_meta.get("ffc_frames", [])
        json_safe = []
        for i in ffc_frames:
            json_safe.append(int(i))
        meta_dict["ffc_frames"] = json_safe
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

    # trying to get only clear frames, work in progrsss
    def set_important_frames(self, min_mass=None, frame_data=None, model=None):
        # this needs more testing
        frames = []
        self.important_frames = []
        for i, mass in enumerate(self.frame_mass):
            if self.ffc_frames is not None and i in self.ffc_frames:
                continue
            if (
                min_mass is None
                or (mass >= min_mass and mass >= self.lower_mass)
                # and mass <= self.upper_mass
            ):  # trying it out
                if frame_data is not None:
                    height, width = frame_data[i].shape
                    if height < MIN_SIZE or width < MIN_SIZE:
                        continue
                    if model and (self.label not in ["false-positive", "insect"]):
                        prediction = model.classify_frame(
                            frame_data[i], self.frame_temp_median[i]
                        )
                        if prediction is None:
                            logging.info(
                                "Couldnt predict Frame %d for clip %s track %s region %s",
                                i + self.start_frame,
                                self.clip_id,
                                self.track_id,
                                self.label,
                                self.track_bounds[i],
                            )
                        predicted_label = model.labels[np.argmax(prediction)]
                        if predicted_label == "false-positive":
                            logging.debug(
                                "Frame %d for clip %s track %s is suspected to be a FP instead of %s",
                                i + self.start_frame,
                                self.clip_id,
                                self.track_id,
                                self.label,
                            )
                            continue
                frames.append(i)
        np.random.shuffle(frames)
        for frame in frames:
            f = FrameSample(
                self.clip_id,
                self.track_id,
                frame,
                self.label,
                self.frame_temp_median[frame],
                self.frame_velocity[frame],
            )
            self.important_frames.append(f)

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
        use_important=False,
        random_frames=True,
        scale=1,
        top_frames=False,
        random_sections=False,
        repeats=1,
    ):
        self.segments = []
        self.filtered_stats = {"segment_mass": 0}
        if use_important:
            frame_indices = [frame.frame_num for frame in self.important_frames]
        else:
            frame_indices = [
                i
                for i, mass in enumerate(mass_history)
                if mass > 0 and (i + self.start_frame) not in self.ffc_frames
            ]
            if segment_min_mass is not None:
                segment_min_mass = min(
                    segment_min_mass, np.median(mass_history[frame_indices])
                )
            else:
                segment_min_mass = 1
                # remove blank frames

            if top_frames and random_frames:
                # take top 50 mass frames
                frame_indices = sorted(
                    frame_indices, key=lambda f_i: mass_history[f_i], reverse=True
                )
                frame_indices = frame_indices[:50]
                frame_indices.sort()

        if len(frame_indices) < segment_width:
            if self.label == "vehicle" or self.label == "human":
                if len(frame_indices) < (segment_width / 4.0):
                    return
            else:
                return
        frame_indices = np.array(frame_indices)
        segment_count = max(1, len(frame_indices) // segment_frame_spacing)
        segment_count = int(scale * segment_count)

        # i3d segments get all segments above min mass sequentially
        if top_frames and not random_frames:
            segment_mass = []

            for i in range(max(1, len(mass_history) - segment_width)):
                contains_ffc = False
                for z in range(segment_width):
                    if (z + i + self.start_frame) in self.ffc_frames:
                        contains_ffc = True
                        break
                if contains_ffc:
                    continue
                mass = np.sum(mass_history[i : i + segment_width])
                segment_mass.append((i, mass))

            sorted_mass = sorted(segment_mass, key=lambda x: x[1], reverse=True)
            best_mass = True

            for _ in range(segment_count):
                segment_info = sorted_mass[0]
                index = segment_info[0]
                avg_mass = segment_info[1] / segment_width
                if not best_mass and (
                    avg_mass < self.lower_mass or avg_mass < segment_min_mass
                ):
                    break
                movement_data = get_movement_data(
                    self.track_bounds[index : index + segment_width],
                    mass_history[index : index + segment_width],
                )
                segment = SegmentHeader(
                    track=self,
                    start_frame=index,
                    frames=segment_width,
                    weight=1,
                    avg_mass=avg_mass,
                    frame_indices=np.arange(segment_width) + index,
                    movement_data=movement_data,
                    best_mass=best_mass,
                    top_mass=True,
                )
                best_mass = False
                self.segments.append(segment)
                sorted_mass = [
                    mass_info
                    for mass_info in sorted_mass
                    if mass_info[0] <= (index - segment_width / 3 * 2)
                    or mass_info[0] >= (index + segment_width / 3 * 2)
                ]
                if len(self.segments) == segment_count or len(sorted_mass) == 0:
                    break
            return

        # give it slightly more than segment_width frames to choose some from
        extra_frames = 2
        # take any segment_width frames, this could be done each epoch
        whole_indices = frame_indices.copy()
        for _ in range(repeats):
            frame_indices = whole_indices
            for i in range(segment_count):
                if (
                    len(frame_indices) < segment_width and len(self.segments) > 1
                ) or len(frame_indices) < (segment_width / 4.0):
                    break

                if random_frames:
                    if random_sections:
                        section = frame_indices[: int(segment_width * 2.2)]
                        indices = np.random.choice(
                            len(section),
                            min(segment_width, len(section)),
                            replace=False,
                        )
                        frames = section[indices]
                        frame_indices = [
                            f_num
                            for f_num in frame_indices
                            if f_num > frames[0] + segment_frame_spacing
                        ]
                    else:
                        indices = np.random.choice(
                            len(frame_indices),
                            min(segment_width, len(frame_indices)),
                            replace=False,
                        )
                        frames = frame_indices[indices]
                        frame_indices = np.delete(frame_indices, indices)
                else:
                    i = int(i // scale)
                    segment_start = i * segment_frame_spacing
                    segment_end = segment_start + segment_width + extra_frames
                    if i > 0:
                        segment_start -= extra_frames
                    else:
                        segment_end += extra_frames
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

                mass_slice = mass_history[frames]
                segment_avg_mass = np.mean(mass_slice)
                if segment_min_mass and segment_avg_mass < segment_min_mass:
                    self.filtered_stats["segment_mass"] += 1
                    continue
                if segment_avg_mass < 50:
                    segment_weight_factor = 0.75
                elif segment_avg_mass < 100:
                    segment_weight_factor = 1
                else:
                    segment_weight_factor = 1.2
                # if we want to use movement_data
                # movement_data = get_movement_data(
                #     self.track_bounds[frames],
                #     mass_history[frames],
                # )
                movement_data = None
                segment = SegmentHeader(
                    track=self,
                    start_frame=frames[0],
                    frames=segment_width,
                    weight=segment_weight_factor,
                    avg_mass=segment_avg_mass,
                    frame_indices=frames,
                    movement_data=movement_data,
                )
                self.segments.append(segment)

        return

    @property
    def camera_id(self):
        """Unique name of this track."""
        return "{}-{}".format(self.camera, self.location)

    @property
    def bin_id(self):
        """Unique name of this track."""
        return "{}".format(self.clip_id)

        # return "{}-{}".format(self.clip_id, self.track_id)

    @property
    def weight(self):
        """Returns total weight for all segments in this track"""
        return sum(segment.weight for segment in self.segments)

    @property
    def unique_id(self):
        return "{}-{}".format(self.clip_id, self.track_id)

    @staticmethod
    def from_meta(clip_id, clip_meta, track_meta, predictions=None):
        """Creates a track header from given metadata."""
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
        frame_temp_median = np.float32(
            clip_meta["frame_temp_median"][
                track_start_frame : num_frames + track_start_frame
            ]
        )

        bounds_history = track_meta["bounds_history"]
        ffc_frames = clip_meta.get("ffc_frames", [])
        important_frames = track_meta.get("important_frames")

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
            res_x=clip_meta.get("res_x", CPTV_FILE_WIDTH),
            res_y=clip_meta.get("res_y", CPTV_FILE_HEIGHT),
            ffc_frames=ffc_frames,
            important_frames=important_frames,
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
                frames += max(len(track.important_frames), max_frames_per_track)
            else:
                frames += len(track.important_frames)

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
        self.label_frames[track.label] -= len(track.important_frames)

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
        self.label_frames[track_header.label] += len(track_header.important_frames)

        segment_length = len(track_header.segments)
        self.segment_sum += segment_length
        self.segments += 1


class FrameSample:
    _frame_id = 1

    def __init__(self, clip_id, track_id, frame_num, label, temp_median, velocity):
        self.id = FrameSample._frame_id
        FrameSample._frame_id += 1
        self.clip_id = clip_id
        self.track_id = track_id
        self.frame_num = frame_num
        self.label = label
        self.temp_median = temp_median
        self.velocity = velocity

    @property
    def unique_track_id(self):
        return "{}-{}".format(self.clip_id, self.track_id)


class SegmentHeader:
    """Header for segment."""

    _segment_id = 1

    def __init__(
        self,
        track: TrackHeader,
        start_frame,
        frames,
        weight,
        avg_mass,
        frame_indices=None,
        movement_data=None,
        best_mass=False,
        top_mass=False,
    ):
        self.movement_data = movement_data
        self.top_mass = top_mass
        self.best_mass = best_mass
        self.id = SegmentHeader._segment_id
        SegmentHeader._segment_id += 1
        # reference to track this segment came from
        self.clip_id = track.clip_id
        self.track_id = track.track_id
        self.frame_indices = frame_indices

        self.track_bounds = {}
        self.frame_temp_median = {}
        for i in frame_indices:
            self.track_bounds[i] = track.track_bounds[i]
            self.frame_temp_median[i] = track.frame_temp_median[i]
        self.track_info = track.track_info
        self.label = track.label
        # first frame of this segment referenced by start of track
        self.start_frame = start_frame
        # length of segment in frames
        self.frames = frames
        # relative weight of the segment (higher is sampled more often)
        self.weight = weight
        # average mass of the segment
        self.avg_mass = avg_mass

    @property
    def unique_track_id(self):
        # reference to clip this segment came from
        return "{}-{}".format(self.clip_id, self.track_id)

    #
    # @property
    # def clip_id(self):
    #     # reference to clip this segment came from
    #     return self.track.clip_id

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

    #
    # @property
    # def track_bounds(self):
    #     # original location of this tracks bounds.
    #     return self.track.track_bounds[
    #         self.start_frame : self.start_frame + self.frames
    #     ]

    @property
    def frame_crop(self):
        # how much each frame has been cropped.
        return self.track.frame_crop[self.start_frame : self.start_frame + self.frames]

    #
    # @property
    # def frame_temp_median(self):
    #     # thermal reference temperature for each frame (i.e. which temp is 0)
    #     return self.track.frame_temp_median[
    #         self.start_frame : self.start_frame + self.frames
    #     ]

    @property
    def end_frame(self):
        """end frame of segment"""
        return self.start_frame + self.frames

    @property
    def track_bin(self):
        """Unique name of this segments track."""
        return self.track.bin_id

    def __str__(self):
        return "{0} label {1} offset:{2} weight:{3:.1f}".format(
            self.unique_track_id, self.label, self.start_frame, self.weight
        )


def get_cropped_fraction(region: tools.Rectangle, width, height):
    """Returns the fraction regions mass outside the rect ((0,0), (width, height)"""
    bounds = tools.Rectangle(0, 0, width - 1, height - 1)
    return 1 - (bounds.overlap_area(region) / region.area)


def get_movement_data(b_h, m_h):
    areas = (b_h[:, 2] - b_h[:, 0]) * (b_h[:, 3] - b_h[:, 1])
    centrex = (b_h[:, 2] + b_h[:, 0]) / 2
    centrey = (b_h[:, 3] + b_h[:, 1]) / 2
    xv = np.hstack((0, centrex[1:] - centrex[:-1]))
    yv = np.hstack((0, centrey[1:] - centrey[:-1]))
    axv = xv / areas ** 0.5
    ayv = yv / areas ** 0.5
    return np.hstack((b_h, np.vstack((m_h, xv, yv, axv, ayv)).T))
