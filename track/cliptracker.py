from abc import ABC, abstractmethod
import logging
import time
import math
import cv2
import numpy as np

from track.track import Track
from track.region import Region
from ml_tools.imageprocessing import detect_objects, normalize


class ClipTracker(ABC):
    def __init__(
        self,
        config,
        cache_to_disk=False,
        keep_frames=True,
        calc_stats=True,
        verbose=False,
        do_tracking=True,
    ):
        config = config.get(self.type)
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
    def start_tracking(self, clip, preview_frames):
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

        for (score, track, region) in scores:
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
            if not self.config.filter_regions_pre_match and (
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

    def _get_regions_of_interest(self, clip, component_details, time_spent):
        """
        Calculates pixels of interest mask from filtered image, and returns both the labeled mask and their bounding
        rectangles.
        :param filtered: The filtered frame
        :return: regions of interest, mask frame
        """
        s = time.time()
        delta_thermal, delta_filtered = self.get_delta_frame(
            clip,
        )
        time_spent["delta"] += time.time() - s

        # we enlarge the rects a bit, partly because we eroded them previously, and partly because we want some context.
        padding = self.frame_padding
        # find regions of interest
        regions = []
        for i, component in enumerate(component_details):

            if component[2] < self.min_dimension or component[3] < self.min_dimension:
                continue
            region = Region(
                component[0],
                component[1],
                component[2],
                component[3],
                mass=component[4],
                id=i,
                frame_number=clip.current_frame,
            )
            # GP this needs to be checked for themals 29/06/2022
            if clip.type == "IR":
                s = time.time()
                if delta_thermal is not None:
                    sub_delta = region.subimage(delta_thermal)
                    previous_delta_mass = len(
                        sub_delta[sub_delta > clip.background_thresh]
                    )
                    # GP TEST NOt sure if this will work
                    region.mass = previous_delta_mass
                    # if previous_delta_mass == 0:
                    #     logging.info("No mass from previous so skipping")
                    #     continue
                    region.pixel_variance = np.var(sub_delta)
                time_spent["delta2"] += time.time() - s

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
            print(
                "track didnt move",
                stats.max_offset,
                stats.frames_moved,
                self.config.track_min_offset,
                self.config.min_moving_frames,
            )
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

        highest_ratio = 0
        for other in clip.tracks:
            if track == other:
                continue
            highest_ratio = max(track.get_overlap_ratio(other), highest_ratio)

        if highest_ratio > self.config.track_overlap_ratio:
            self.print_if_verbose(
                "Track filtered.  Too much overlap {}".format(highest_ratio)
            )
            clip.filtered_tracks.append(("Track filtered.  Too much overlap", track))
            return True

        return False

    def print_if_verbose(self, info_string):

        if self.verbose:
            logging.info(info_string)
