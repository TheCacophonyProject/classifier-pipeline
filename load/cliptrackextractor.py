"""
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""


import logging
import numpy as np
import yaml

from cptv import CPTVReader
import cv2

from .clip import Clip
from ml_tools.tools import Rectangle
from track.region import Region
from track.track import Track
from piclassifier.motiondetector import is_affected_by_ffc
from ml_tools.imageprocessing import normalize, detect_objects

from matplotlib import pyplot as plt


class ClipTrackExtractor:
    BASE_DISTANCE_CHANGE = 450
    BASE_MASS_CHANGE = 10
    MAX_DISTANCE = 2000
    PREVIEW = "preview"

    def __init__(
        self,
        config,
        use_opt_flow,
        cache_to_disk,
        keep_frames=True,
        calc_stats=True,
        high_quality_optical_flow=False,
    ):
        self.config = config
        self.use_opt_flow = use_opt_flow
        self.high_quality_optical_flow = high_quality_optical_flow
        self.stats = None
        self.cache_to_disk = cache_to_disk
        self.max_tracks = config.max_tracks
        # frame_padding < 3 causes problems when we get small areas...
        self.frame_padding = max(3, self.config.frame_padding)
        # the dilation effectively also pads the frame so take it into consideration.
        self.frame_padding = max(0, self.frame_padding - self.config.dilation_pixels)
        self.keep_frames = keep_frames
        self.calc_stats = calc_stats

        if self.config.dilation_pixels > 0:
            size = self.config.dilation_pixels * 2 + 1
            self.dilate_kernel = np.ones((size, size), np.uint8)

    def parse_clip(self, clip):
        """
        Loads a cptv file, and prepares for track extraction.
        """
        clip.set_frame_buffer(
            self.high_quality_optical_flow,
            self.cache_to_disk,
            self.use_opt_flow,
            self.keep_frames,
        )

        with open(clip.source_file, "rb") as f:
            reader = CPTVReader(f)
            clip.set_res(reader.x_resolution, reader.y_resolution)

            camera_model = None
            if reader.model:
                camera_model = reader.model.decode()
            clip.set_model(camera_model)

            # if we have the triggered motion threshold should use that
            # maybe even override dynamic threshold with this value
            if reader.motion_config:
                motion = yaml.safe_load(reader.motion_config)
                temp_thresh = motion.get("triggeredthresh")
                if temp_thresh:
                    clip.temp_thresh = temp_thresh

            video_start_time = reader.timestamp.astimezone(Clip.local_tz)
            clip.num_preview_frames = (
                reader.preview_secs * clip.frames_per_second
                - self.config.preview_ignore_frames
            )
            clip.set_video_stats(video_start_time)
            clip.calculate_background(reader)

        with open(clip.source_file, "rb") as f:
            reader = CPTVReader(f)
            for frame in reader:
                self.process_frame(clip, frame.pix, is_affected_by_ffc(frame))

        if not clip.from_metadata:
            self.apply_track_filtering(clip)

        if self.calc_stats:
            clip.stats.completed(clip.frame_on, clip.res_y, clip.res_x)

        return True

    def process_frame(self, clip, frame, ffc_affected=False):
        if ffc_affected:
            self.print_if_verbose("{} ffc_affected".format(clip.frame_on))
        clip.ffc_affected = ffc_affected

        self._process_frame(clip, frame, ffc_affected)
        clip.frame_on += 1

    def apply_track_filtering(self, clip):
        self.filter_tracks(clip)
        # apply smoothing if required
        if self.config.track_smoothing and clip.frame_on > 0:
            for track in clip.active_tracks:
                track.smooth(Rectangle(0, 0, clip.res_x, clip.res_y))

    def _get_filtered_frame(self, clip, thermal):
        """
        Calculates filtered frame from thermal
        :param thermal: the thermal frame
        :param background: (optional) used for background subtraction
        :return: the filtered frame
        """

        # has to be a signed int so we dont get overflow
        filtered = np.float32(thermal.copy())
        avg_change = int(round(np.average(thermal) - clip.stats.mean_background_value))
        np.clip(filtered - clip.background - avg_change, 0, None, out=filtered)
        max = np.amax(filtered)
        min = np.amin(filtered)
        filtered, _ = normalize(filtered, min=min, max=max, new_max=255)
        filtered = cv2.fastNlMeansDenoising(np.uint8(filtered), None)
        # if clip.frame_on > 90:
        #     plt.imshow(filtered)
        #     plt.show()
        #     plt.imshow(clip.background)
        #     plt.show()
        # print(
        #     "clip threshold is",
        #     clip.background_thresh,
        #     "after norm this hsould be",
        #     min,
        #     max,
        #     clip.background_thresh * 255 / (max - min),
        # )
        # print("filtered values", np.amax(filtered), avg_change, np.amax(thermal))
        return filtered, clip.background_thresh * 255 / (max - min)

    def _process_frame(self, clip, thermal, ffc_affected=False):
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
            If specified background subtraction algorithm will be used.
        """
        filtered, thresh = self._get_filtered_frame(clip, thermal)

        _, mask, component_details = detect_objects(
            filtered.copy(), otsus=False, threshold=thresh, iterations=2
        )
        prev_filtered = clip.frame_buffer.get_last_filtered()
        clip.add_frame(thermal, filtered, mask, ffc_affected)

        if clip.from_metadata:
            for track in clip.tracks:
                if clip.frame_on in track.frame_list:
                    track.add_frame_for_existing_region(
                        clip.frame_buffer.get_last_frame(),
                        clip.background_thresh,
                        prev_filtered,
                    )
        else:
            regions = self._get_regions_of_interest(
                clip, component_details, filtered, prev_filtered
            )
            clip.region_history.append(regions)
            self._apply_region_matchings(
                clip, regions, create_new_tracks=not ffc_affected
            )

    def _apply_region_matchings(self, clip, regions, create_new_tracks=True):
        """
        Work out the best matchings between tracks and regions of interest for the current frame.
        Create any new tracks required.
        """
        unmatched_regions, matched_tracks = self._match_existing_tracks(clip, regions)
        if create_new_tracks:
            new_tracks = self._create_new_tracks(clip, unmatched_regions)
        else:
            new_tracks = set()

        self._filter_inactive_tracks(clip, new_tracks, matched_tracks)

    def _match_existing_tracks(self, clip, regions):

        scores = []
        used_regions = set()
        unmatched_regions = set(regions)
        for track in clip.active_tracks:
            for region in regions:
                distance, size_change = track.get_region_score(region)
                # we give larger tracks more freedom to find a match as they might move quite a bit.

                max_distance = get_max_distance_change(track)
                max_size_change = get_max_size_change(track, region)
                if distance > max_distance:
                    self.print_if_verbose(
                        "track {} distance score {} bigger than max distance {}".format(
                            track.get_id(), distance, max_distance
                        )
                    )

                    continue
                if size_change > max_size_change:
                    self.print_if_verbose(
                        "track {} size_change {} bigger than max size_change {}".format(
                            track.get_id(), size_change, max_size_change
                        )
                    )
                scores.append((distance, track, region))

        # makes tracking consistent by ordering by score then by frame since target then track id
        scores.sort(
            key=lambda record: record[1].frames_since_target_seen
            + float(".{}".format(record[1]._id))
        )
        scores.sort(key=lambda record: record[0])

        matched_tracks = set()
        for (score, track, region) in scores:
            if track in matched_tracks or region in used_regions:
                continue
            track.add_region(region)
            matched_tracks.add(track)
            used_regions.add(region)
            unmatched_regions.remove(region)

        return unmatched_regions, matched_tracks

    def _create_new_tracks(self, clip, unmatched_regions):
        """ Create new tracks for any unmatched regions """
        new_tracks = set()
        for region in unmatched_regions:
            # make sure we don't overlap with existing tracks.  This can happen if a tail gets tracked as a new object
            overlaps = [
                track.last_bound.overlap_area(region) for track in clip.active_tracks
            ]
            if len(overlaps) > 0 and max(overlaps) > (region.area * 0.25):
                continue

            track = Track.from_region(clip, region)
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

    def _filter_inactive_tracks(self, clip, new_tracks, matched_tracks):
        """ Filters tracks which are or have become inactive """

        unactive_tracks = clip.active_tracks - matched_tracks - new_tracks
        clip.active_tracks = matched_tracks | new_tracks
        for track in unactive_tracks:
            # need sufficient frames to allow insertion of excess blanks
            remove_after = min(
                2 * (len(track) - track.blank_frames),
                self.config.remove_track_after_frames,
            )
            if track.frames_since_target_seen + 1 < remove_after:
                track.add_blank_frame(clip.frame_buffer)
                clip.active_tracks.add(track)
                self.print_if_verbose(
                    "frame {} adding a blank frame to {} ".format(
                        clip.frame_on, track.get_id()
                    )
                )

    def _get_regions_of_interest(
        self, clip, component_details, filtered, prev_filtered
    ):
        """
                Calculates pixels of interest mask from filtered image, and returns both the labeled mask and their bounding
                rectangles.
                :param filtered: The filtered frame
        =        :return: regions of interest, mask frame
        """

        if prev_filtered is not None:
            delta_frame = np.abs(filtered - prev_filtered)
        else:
            delta_frame = None

        # we enlarge the rects a bit, partly because we eroded them previously, and partly because we want some context.
        padding = self.frame_padding
        # find regions of interest
        regions = []
        for i, component in enumerate(component_details[1:]):

            region = Region(
                component[0],
                component[1],
                component[2],
                component[3],
                mass=component[4],
                id=i,
                frame_number=clip.frame_on,
            )

            old_region = region.copy()
            region.crop(clip.crop_rectangle)
            region.was_cropped = str(old_region) != str(region)
            region.set_is_along_border(clip.crop_rectangle)

            if self.config.cropped_regions_strategy == "cautious":
                crop_width_fraction = (
                    old_region.width - region.width
                ) / old_region.width
                crop_height_fraction = (
                    old_region.height - region.height
                ) / old_region.height
                if crop_width_fraction > 0.25 or crop_height_fraction > 0.25:
                    continue
            elif self.config.cropped_regions_strategy == "none":
                if region.was_cropped:
                    continue
            elif self.config.cropped_regions_strategy != "all":
                raise ValueError(
                    "Invalid mode for CROPPED_REGIONS_STRATEGY, expected ['all','cautious','none'] but found {}".format(
                        self.config.cropped_regions_strategy
                    )
                )
            region.enlarge(padding, max=clip.crop_rectangle)

            if delta_frame is not None:
                region_difference = region.subimage(delta_frame)
                region.pixel_variance = np.var(region_difference)

            # filter out regions that are probably just noise
            if (
                region.pixel_variance < self.config.aoi_pixel_variance
                and region.mass < self.config.aoi_min_mass
            ):
                continue
            regions.append(region)
        return regions

    def filter_tracks(self, clip):

        for track in clip.tracks:
            track.trim()
            track.set_end_s(clip.frames_per_second)

        track_stats = [(track.get_stats(), track) for track in clip.tracks]
        track_stats.sort(reverse=True, key=lambda record: record[0].score)
        if self.config.verbose:
            for stats, track in track_stats:
                start_s, end_s = clip.start_and_end_in_secs(track)
                logging.info(
                    " - track duration: %.1fsec, number of frames:%s, offset:%.1fpx, delta:%.1f, mass:%.1fpx",
                    end_s - start_s,
                    len(track),
                    stats.max_offset,
                    stats.delta_std,
                    stats.average_mass,
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
            clip.filtered_tracks.append(("Track filtered.  Too much overlap", track))
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
            self.print_if_verbose("Track filtered.  Too static")
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

        if self.config.verbose:
            logging.info(info_string)


def get_max_size_change(track, region):
    exiting = region.is_along_border and not track.last_bound.is_along_border
    entering = not exiting and track.last_bound.is_along_border
    region_percent = 1.5
    if len(track) < 5:
        # may increase at first
        region_percent = 2
    if entering or exiting:
        region_percent = 2

    return region_percent


def get_max_distance_change(track):
    x, y = track.velocity
    velocity_distance = (2 * x) ** 2 + (2 * y) ** 2
    pred_vel = track.predicted_velocity()
    pred_distance = pred_vel[0] ** 2 + pred_vel[1] ** 2

    max_distance = np.clip(
        ClipTrackExtractor.BASE_DISTANCE_CHANGE + max(velocity_distance, pred_distance),
        0,
        ClipTrackExtractor.MAX_DISTANCE,
    )
    return max_distance
