from abc import ABC, abstractmethod
import logging
import time
import math
import cv2
import numpy as np

from track.track import Track
from track.region import Region
from ml_tools.imageprocessing import detect_objects, normalize, hist_diff


class ClipTracker(ABC):
    def __init__(
        self,
        config,
        cache_to_disk=False,
        keep_frames=True,
        calc_stats=True,
        verbose=False,
        do_tracking=True,
        scale=None,
    ):
        config = config.get(self.type)
        self.scale = scale
        # if scale:
        # config.rescale(scale)
        self.do_tracking = do_tracking
        self.verbose = verbose
        self.config = config
        self.stats = None
        self.cache_to_disk = cache_to_disk
        self.max_tracks = config.max_tracks
        # frame_padding < 3 causes problems when we get small areas...
        self.frame_padding = max(3, self.config.frame_padding)
        # the dilation effectively also pads the frame so take it into consideration.
        # self.frame_padding = max(0, self.frame_padding - self.config.dilation_pixels)
        self.keep_frames = keep_frames
        self.calc_stats = calc_stats
        self._tracking_time = None
        self.min_dimension = config.min_dimension
        # if self.config.dilation_pixels > 0:
        #     size = self.config.dilation_pixels * 2 + 1
        #     self.dilate_kernel = np.ones((size, size), np.uint8)

    @abstractmethod
    def type(self):
        """Tracker type IR or Thermal"""

    @abstractmethod
    def parse_clip(self, clip, process_background=False):
        """parse_clip version"""

    @property
    @abstractmethod
    def tracker_version(self):
        """Tracker version"""
        ...

    @property
    @abstractmethod
    def tracking_time(self):
        """Tracker time"""
        ...

    @property
    @abstractmethod
    def start_tracking(self, clip, preview_frames, track_frames=True):
        """start_tracking"""
        ...

    @abstractmethod
    def process_frame(self, clip, rawframe, ffc_affected=False):
        """Get all regions for this sample"""
        ...

    def apply_track_filtering(self, clip):
        self.filter_tracks(clip)
        # apply smoothing if required
        if self.config.track_smoothing and clip.current_frame > 0:
            for track in clip.active_tracks:
                track.smooth(Rectangle(0, 0, clip.res_x, clip.res_y))

    def _get_filtered_frame(self, clip, thermal):
        """
        Calculates filtered frame from thermal
        :param thermal: the thermal frame
        :param background: (optional) used for background subtraction
        :return: uint8 filtered frame and adjusted clip threshold for normalized frame
        """

        filtered = np.float32(thermal.copy())
        avg_change = int(round(np.average(thermal) - clip.stats.mean_background_value))
        np.clip(filtered - clip.background - avg_change, 0, None, out=filtered)
        filtered, stats = normalize(filtered, new_max=255)
        if self.config.denoise:
            filtered = cv2.fastNlMeansDenoising(np.uint8(filtered), None)
        if stats[1] == stats[2]:
            mapped_thresh = clip.background_thresh
        else:
            mapped_thresh = clip.background_thresh / (stats[1] - stats[2]) * 255
        return filtered, mapped_thresh

    def _apply_region_matchings(self, clip, regions):
        """
        Work out the best matchings between tracks and regions of interest for the current frame.
        Create any new tracks required.
        """
        unmatched_regions, matched_tracks = self._match_existing_tracks(clip, regions)
        new_tracks = self._create_new_tracks(clip, unmatched_regions)

        unactive_tracks = clip.active_tracks - matched_tracks - new_tracks
        clip.active_tracks = matched_tracks | new_tracks
        self._filter_inactive_tracks(clip, unactive_tracks)

    def _match_existing_tracks(self, clip, regions):
        scores = []
        used_regions = set()
        unmatched_regions = set(regions)
        active = list(clip.active_tracks)
        active.sort(key=lambda x: x.get_id())
        for track in active:
            scores.extend(track.match(regions))

        # makes tracking consistent by ordering by score then by frame since target then track id
        scores.sort(
            key=lambda record: record[1].frames_since_target_seen
            + float(".{}".format(record[1]._id))
        )
        scores.sort(key=lambda record: record[0])
        matched_tracks = set()
        blanked_tracks = set()
        cur_frame = clip.frame_buffer.current_frame
        for score, track, region in scores:
            if (
                track in matched_tracks
                or region in used_regions
                or track in blanked_tracks
            ):
                continue
            logging.debug(
                "frame# %s matched %s to track %s", clip.current_frame, region, track
            )
            used_regions.add(region)
            unmatched_regions.remove(region)
            if not self.config.filter_regions_pre_match:
                if self.config.min_hist_diff is not None:
                    background = clip.background
                    # if self.scale:
                    #     background = clip.rescaled_background(
                    #         (int(self.res_x), int(self.res_y))
                    #     )
                    hist_v = hist_diff(region, background, cur_frame.thermal)
                    if hist_v > self.config.min_hist_diff:
                        logging.warn(
                            "%s filtering region %s because of hist diff %s track %s ",
                            region.frame_number,
                            region,
                            hist_v,
                            track,
                        )

                        blanked_tracks.add(track)
                        continue
                if (
                    region.pixel_variance < self.config.aoi_pixel_variance
                    or region.mass < self.config.aoi_min_mass
                ):
                    # this will force a blank frame to be added, rather than if we filter earlier
                    # and match this track to a different region
                    logging.debug(
                        "%s filtering region %s because of variance %s and mass %s track %s",
                        region.frame_number,
                        region,
                        region.pixel_variance,
                        region.mass,
                        track,
                    )
                    blanked_tracks.add(track)
                    continue
            track.add_region(region)
            matched_tracks.add(track)

        return unmatched_regions, matched_tracks

    def _create_new_tracks(self, clip, unmatched_regions):
        """Create new tracks for any unmatched regions"""
        new_tracks = set()
        for region in unmatched_regions:
            # make sure we don't overlap with existing tracks.  This can happen if a tail gets tracked as a new object
            overlaps = [
                track.last_bound.overlap_area(region) for track in clip.active_tracks
            ]
            if len(overlaps) > 0 and max(overlaps) > (region.area * 0.25):
                continue

            track = Track.from_region(
                clip,
                region,
                self.tracker_version,
                tracking_config=self.config,
            )
            new_tracks.add(track)
            clip._add_active_track(track)
            self.print_if_verbose(
                "Creating a new track {} with region {} mass{} area {} frame {}".format(
                    track.get_id(),
                    region,
                    track.last_bound.mass,
                    track.last_bound.area,
                    region.frame_number,
                )
            )
        return new_tracks

    def _filter_inactive_tracks(self, clip, unactive_tracks):
        """Filters tracks which are or have become inactive"""
        for track in unactive_tracks:
            track.add_blank_frame()
            if track.tracking:
                clip.active_tracks.add(track)
                logging.debug(
                    "frame {} adding a blank frame to {} ".format(
                        clip.current_frame, track.get_id()
                    )
                )

    def get_delta_frame(self, clip):
        frame = clip.frame_buffer.current_frame
        prev_frame = clip.frame_buffer.prev_frame
        if prev_frame is None:
            return None, None
        filtered, _ = normalize(frame.filtered, new_max=255)
        prev_filtered, _ = normalize(prev_frame.filtered, new_max=255)
        delta_filtered = np.abs(np.float32(filtered) - np.float32(prev_filtered))

        thermal, _ = normalize(frame.thermal, new_max=255)
        prev_thermal, _ = normalize(prev_frame.thermal, new_max=255)
        delta_thermal = np.abs(np.float32(thermal) - np.float32(prev_thermal))
        return delta_thermal, delta_filtered

    def _get_regions_of_interest(self, clip, component_details, centroids=None):
        """
        Calculates pixels of interest mask from filtered image, and returns both the labeled mask and their bounding
        rectangles.
        :param filtered: The filtered frame
        :return: regions of interest, mask frame
        """
        delta_thermal, delta_filtered = self.get_delta_frame(
            clip,
        )

        # we enlarge the rects a bit, partly because we eroded them previously, and partly because we want some context.
        padding = self.frame_padding
        # find regions of interest
        regions = []
        for i, component in enumerate(component_details):
            if centroids is None:
                centroid = [
                    int(component[0] + component[2] / 2),
                    int(component[1] + component[3] / 2),
                ]
            else:
                centroid = centroids[i]
            region = Region(
                component[0],
                component[1],
                component[2],
                component[3],
                mass=component[4],
                id=i,
                frame_number=clip.current_frame,
                centroid=centroid,
            )

            if self.scale:
                region.rescale(1 / self.scale)
            if region.width < self.min_dimension or region.height < self.min_dimension:
                continue
            # GP this needs to be checked for themals 29/06/2022
            if clip.type == "IR":
                if delta_thermal is not None:
                    # filtered only 0 or 255
                    sub_delta = region.subimage(delta_thermal)
                    previous_delta_mass = len(
                        sub_delta[sub_delta > clip.background_thresh]
                    )
                    # if previous_delta_mass == 0:
                    #     logging.info("No mass from previous so skipping")
                    #     continue
                    region.pixel_variance = np.var(sub_delta)

            elif delta_filtered is not None:
                region_difference = region.subimage(delta_filtered)
                region.pixel_variance = np.var(region_difference)
            old_region = region.copy()
            region.crop(clip.crop_rectangle)
            region.was_cropped = str(old_region) != str(region)

            if self.config.cropped_regions_strategy == "cautious":
                crop_width_fraction = (
                    old_region.width - region.width
                ) / old_region.width
                crop_height_fraction = (
                    old_region.height - region.height
                ) / old_region.height
                if crop_width_fraction > 0.25 or crop_height_fraction > 0.25:
                    continue
            elif (
                self.config.cropped_regions_strategy == "none"
                or self.config.cropped_regions_strategy is None
            ):
                if region.was_cropped:
                    continue
            elif self.config.cropped_regions_strategy != "all":
                raise ValueError(
                    "Invalid mode for CROPPED_REGIONS_STRATEGY, expected ['all','cautious','none'] but found {}".format(
                        self.config.cropped_regions_strategy
                    )
                )

            # filter out regions that are probably just noise
            if self.config.filter_regions_pre_match and (
                region.pixel_variance < self.config.aoi_pixel_variance
                and region.mass < self.config.aoi_min_mass
            ):
                logging.debug(
                    "%s filtering region %s because of variance %s and mass %s",
                    region.frame_number,
                    region,
                    region.pixel_variance,
                    region.mass,
                )
                continue

            region.enlarge(padding, max=clip.crop_rectangle)
            # gp dunno if we should use this feels like we already have the edge
            # extra_edge = 0
            extra_edge = math.ceil(clip.crop_rectangle.width * 0.03)
            region.set_is_along_border(clip.crop_rectangle, edge=extra_edge)
            regions.append(region)
        return regions

    def filter_tracks(self, clip):
        for track in clip.tracks:
            track.trim()
            track.set_end_s(clip.frames_per_second)

        track_stats = [(track.get_stats(), track) for track in clip.tracks]
        track_stats.sort(reverse=True, key=lambda record: record[0].score)
        if self.verbose:
            for stats, track in track_stats:
                start_s, end_s = clip.start_and_end_in_secs(track)
                logging.info(
                    " - track %s duration: %.1fsec, number of frames:%s, stats %s",
                    track.get_id(),
                    end_s - start_s,
                    len(track),
                    stats,
                )
        # filter out tracks that probably are just noise.
        good_tracks = []
        self.print_if_verbose(
            "{} {}".format("Number of tracks before filtering", len(clip.tracks))
        )

        for stats, track in track_stats:
            # discard any tracks that overlap too often with other tracks.  This normally means we are tracking the
            # tail of an animal.
            if not self.filter_track(clip, track, stats):
                good_tracks.append(track)

        clip.tracks = good_tracks
        self.print_if_verbose(
            "{} {}".format("Number of 'good' tracks", len(clip.tracks))
        )
        # apply max_tracks filter
        # note, we take the n best tracks.
        if self.max_tracks is not None and self.max_tracks < len(clip.tracks):
            logging.warning(
                " -using only {0} tracks out of {1}".format(
                    self.max_tracks, len(clip.tracks)
                )
            )
            clip.filtered_tracks.extend(
                [("Too many tracks", track) for track in clip.tracks[self.max_tracks :]]
            )
            clip.tracks = clip.tracks[: self.max_tracks]

        for key in clip.filtered_tracks:
            self.print_if_verbose(
                "filtered track {} because {}".format(key[1].get_id(), key[0])
            )

    def filter_track(self, clip, track, stats):
        # discard any tracks that are less min_duration
        # these are probably glitches anyway, or don't contain enough information.
        if len(track) < self.config.min_duration_secs * clip.frames_per_second:
            self.print_if_verbose("Track filtered. Too short, {}".format(len(track)))
            clip.filtered_tracks.append(("Track filtered.  Too short", track))
            return True

        # discard tracks that do not move enough

        if (
            stats.max_offset < self.config.track_min_offset
            or stats.frames_moved < self.config.min_moving_frames
        ):
            self.print_if_verbose(
                "Track filtered.  Didn't move {}".format(stats.max_offset)
            )
            clip.filtered_tracks.append(("Track filtered.  Didn't move", track))
            return True

        if stats.blank_percent > self.config.max_blank_percent:
            self.print_if_verbose("Track filtered.  Too Many Blanks")
            clip.filtered_tracks.append(("Track filtered. Too Many Blanks", track))
            return True
        if stats.region_jitter > self.config.max_jitter:
            self.print_if_verbose("Track filtered.  Too Jittery")
            clip.filtered_tracks.append(("Track filtered.  Too Jittery", track))
            return True
        # discard tracks that do not have enough delta within the window (i.e. pixels that change a lot)
        if stats.delta_std < clip.track_min_delta:
            self.print_if_verbose(
                "Track filtered.  Too static {}".format(stats.delta_std)
            )
            clip.filtered_tracks.append(("Track filtered.  Too static", track))
            return True
        if stats.delta_std > clip.track_max_delta:
            self.print_if_verbose("Track filtered.  Too Dynamic")
            clip.filtered_tracks.append(("Track filtered.  Too Dynamic", track))
            return True
        # discard tracks that do not have enough enough average mass.
        if stats.average_mass < self.config.track_min_mass:
            self.print_if_verbose(
                "Track filtered.  Mass too small ({})".format(stats.average_mass)
            )
            clip.filtered_tracks.append(("Track filtered.  Mass too small", track))

            return True
        # GP Removed as dont think this does much but filter good tracks, if the tracking
        # chose 2 tracks lets keep them, perhaps usefull for short tracks where 90% overlaps
        # but these will probably be filtered by being too short
        # highest_ratio = 0
        # for other in clip.tracks:
        #     if track == other:
        #         continue
        #     highest_ratio = max(track.get_overlap_ratio(other), highest_ratio)
        #
        # if highest_ratio > self.config.track_overlap_ratio:
        #     self.print_if_verbose(
        #         "Track filtered.  Too much overlap {}".format(highest_ratio)
        #     )
        #     clip.filtered_tracks.append(("Track filtered.  Too much overlap", track))
        #     return True

        return False

    def print_if_verbose(self, info_string):
        if self.verbose:
            logging.info(info_string)


class Background(ABC):
    TRIGGER_FRAMES = 2

    def __init__(self):
        self.rescaled = None
        self.prev_triggered = False
        self.triggered = 0
        self.movement_detected = False
        self.kernel_trigger = np.ones(
            (15, 15), "uint8"
        )  # kernel for erosion when not recording
        self.kernel_recording = np.ones(
            (10, 10), "uint8"
        )  # kernel for erosion when recording

    @abstractmethod
    def set_background(self, background, frames=1):
        """set_background version"""
        ...

    @abstractmethod
    def update_background(self, thermal, filtered):
        """update_background version"""
        ...

    @abstractmethod
    def compute_filtered(self, thermal, threshold):
        """compute_filtered version"""
        ...

    @property
    @abstractmethod
    def background(self):
        """Background image"""
        ...

    @property
    @abstractmethod
    def frames(self):
        """frames used"""
        ...

    @property
    def frames(self):
        return self._frames

    def get_kernel(self):
        if self.movement_detected:
            return self.kernel_recording
        else:
            return self.kernel_trigger

    def detect_motion(self):
        fg = self.compute_filtered(None)
        erosion_image = cv2.erode(fg, self.get_kernel())
        erosion_pixels = len(erosion_image[erosion_image > 0])

        self.prev_triggered = erosion_pixels > 0
        if erosion_pixels > 0:
            self.triggered += 1
            self.triggered = min(self.triggered, 2)
        else:
            self.triggered -= 1
            self.triggered = max(self.triggered, 0)
        self.movement_detected = self.triggered >= Background.TRIGGER_FRAMES
        return self.movement_detected


class CVBackground(Background):
    def __init__(self, tracking_alg="mog2"):
        super().__init__()
        self.use_subsense = False
        # knn doesnt respect learning rate, but maybe mog2 is better anyway
        if tracking_alg == "subsense":
            import pybgs as bgs

            self.use_subsense = True

            self.algorithm = bgs.SuBSENSE()
        elif tracking_alg == "mog2":
            self.algorithm = cv2.createBackgroundSubtractorMOG2(
                history=1000, detectShadows=False
            )
        else:
            raise Exception(f"No algorihtm details found for {tracking_alg}")
            # print(self.algorithm.getBackgroundRatio(), "RATION")
            # 1 / 0
        # self.algorithm = cv2.createBackgroundSubtractorKNN(
        #     history=1000, detectShadows=False
        # )
        self._frames = 0
        self._background = None

    def set_background(self, background, frames=1):
        # seems to be better to do x times rather than just set background
        if self.use_subsense:
            for _ in range(10):
                # doesnt have a learning rate
                self.update_background(background, learning_rate=1)
        else:
            self.update_background(background, learning_rate=1)

            # return

    def update_background(self, thermal, filtered=None, learning_rate=-1):
        if self.use_subsense:
            self._background = self.algorithm.apply(thermal)
        else:
            self._background = self.algorithm.apply(thermal, None, learning_rate)

        self._frames += 1

    @property
    def background(self):
        if self.use_subsense:
            return self.algorithm.getBackgroundModel()
        else:
            return self.algorithm.getBackgroundImage()

    def compute_filtered(self, thermal):
        return self._background


class DiffBackground(Background):
    def __init__(self, background_thresh):
        super().__init__()
        self._frames = 1
        self._background = None
        self.background_thresh = background_thresh

    def set_background(self, background, frames=1):
        self._frames = frames
        self._background = np.float32(background) * self.frames
        return

    def update_background(self, thermal):
        background = self.background
        filtered = get_diff_back_filtered(
            self.background,
            thermal,
            self.background_thresh,
        )
        new_thermal = np.where(filtered > 0, background, thermal)
        self._background += new_thermal
        self._frames += 1

    def compute_filtered(self, thermal=None):
        filtered = get_diff_back_filtered(
            self.background,
            thermal,
            self.background_thresh,
        )
        return filtered

    @property
    def background(self):
        return self._background / self.frames

    @property
    def frames(self):
        return self._frames


def get_diff_back_filtered(background, frame, back_thresh):
    """
    Calculates filtered frame from thermal
    :param frame: the frame
    :param background: (optional) used for background subtraction
    :return: uint8 filtered frame and adjusted clip threshold for normalized frame
    """

    filtered = np.float32(frame.copy())
    filtered = abs(filtered - background)
    filtered[filtered < back_thresh] = 0
    filtered, stats = normalize(filtered, new_max=255)
    return filtered
