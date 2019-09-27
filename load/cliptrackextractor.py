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

from cptv import CPTVReader
import cv2

from .clip import Clip
import ml_tools.tools as tools
from ml_tools.tools import Rectangle
from track.region import Region
from track.track import Track


class ClipTrackExtractor:
    def __init__(
        self, config, use_opt_flow, cache_to_disk, keep_frames=True, calc_stats=True
    ):

        self.config = config
        self.use_opt_flow = use_opt_flow
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
            self.config.high_quality_optical_flow,
            self.cache_to_disk,
            self.use_opt_flow,
            self.keep_frames,
        )

        with open(clip.source_file, "rb") as f:
            reader = CPTVReader(f)
            clip.set_res(reader.x_resolution, reader.y_resolution)
            video_start_time = reader.timestamp.astimezone(Clip.local_tz)
            clip.num_preview_frames = (
                reader.preview_secs * clip.frames_per_second - self.config.ignore_frames
            )

            clip.set_video_stats(video_start_time)
            # we need to load the entire video so we can analyse the background.

            if clip.background_is_preview:
                for frame in reader:
                    self.process_frame(clip, frame.pix)
            else:
                self.process_frames(clip, [np.float32(frame.pix) for frame in reader])

        if self.calc_stats:
            clip.stats.completed(clip.frame_on, clip.res_y, clip.res_x)

    def process_frame(self, clip, frame):
        if clip.on_preview():
            clip.calculate_preview_from_frame(frame)
            if clip.background_calculated:
                for i, back_frame in enumerate(clip.preview_frames):
                    clip.frame_on = i
                    self._process_frame(clip, back_frame)
                clip.preview_frames = None
        else:
            self._process_frame(clip, frame)
        clip.frame_on += 1

    def _whole_clip_stats(self, clip, frames):
        filtered = np.float32(
            [self._get_filtered_frame(clip, frame) for frame in frames]
        )

        delta = np.asarray(frames[1:], dtype=np.float32) - np.asarray(
            frames[:-1], dtype=np.float32
        )
        average_delta = float(np.mean(np.abs(delta)))

        # take half the max filtered value as a threshold
        threshold = float(
            np.percentile(
                np.reshape(filtered, [-1]), q=self.config.threshold_percentile
            )
        )

        # cap the threshold to something reasonable
        threshold = max(self.config.min_threshold, threshold)
        threshold = min(self.config.max_threshold, threshold)

        clip.threshold = threshold
        if self.calc_stats:
            clip.stats.threshold = threshold
            clip.stats.temp_thresh = self.config.temp_thresh
            clip.stats.average_delta = float(average_delta)
            clip.stats.filtered_deviation = float(np.mean(np.abs(filtered)))
            clip.stats.is_static_background = (
                clip.stats.filtered_deviation < clip.config.static_background_threshold
            )

            if (
                not clip.stats.is_static_background
                or clip.disable_background_subtraction
            ):
                clip.background = None

    def process_frames(self, clip, frames):
        # for now just always calculate as we are using the stats...
        # background np.float64[][] filtered calculated here and stats
        clip.background_from_whole_clip(frames)
        self._whole_clip_stats(clip, frames)
        if clip.background_is_preview:
            if clip.preview_frames > 0:
                # background np.int32[][]
                clip.background_from_frames(frames)
            else:
                logging.info(
                    "No preview secs defined for CPTV file - using statistical background measurement"
                )

        # process each frame
        for frame_number, frame in enumerate(frames):
            self._process_frame(clip, frame_number, frame)

        if not clip.from_metadata:
            self.apply_track_filtering(clip)

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
        if clip.background is None:
            filtered = filtered - np.median(filtered) - 40
            filtered[filtered < 0] = 0
        elif clip.background_is_preview:
            avg_change = int(
                round(np.average(thermal) - clip.stats.mean_background_value)
            )
            filtered[filtered < clip.temp_thresh] = 0
            np.clip(filtered - clip.background - avg_change, 0, None, out=filtered)

        else:
            filtered = filtered - clip.background
            filtered = filtered - np.median(filtered)
            filtered[filtered < 0] = 0
        return filtered

    def _process_frame(self, clip, thermal):
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
            If specified background subtraction algorithm will be used.
        """

        filtered = self._get_filtered_frame(clip, thermal)
        frame_height, frame_width = filtered.shape
        mask = np.zeros(filtered.shape)
        edge = self.config.edge_pixels

        # remove the edges of the frame as we know these pixels can be spurious value
        edgeless_filtered = clip.crop_rectangle.subimage(filtered)
        thresh, mass = tools.blur_and_return_as_mask(
            edgeless_filtered, threshold=clip.threshold
        )
        thresh = np.uint8(thresh)
        dilated = thresh

        # Dilation groups interested pixels that are near to each other into one component(animal/track)
        if self.config.dilation_pixels > 0:
            dilated = cv2.dilate(dilated, self.dilate_kernel, iterations=1)

        labels, small_mask, stats, _ = cv2.connectedComponentsWithStats(dilated)
        mask[edge : frame_height - edge, edge : frame_width - edge] = small_mask

        clip.add_frame(thermal, filtered, mask)

        prev_filtered = clip.frame_buffer.get_last_filtered()
        if clip.from_metadata:
            for track in clip.active_tracks:
                if clip.frame_on in track.frame_list:
                    track.add_frame_for_existing_region(
                        clip.frame_buffer.get_last_frame(),
                        clip.threshold,
                        prev_filtered,
                    )
        else:
            regions = self._get_regions_of_interest(
                clip, labels, stats, thresh, filtered, prev_filtered, mass
            )
            clip.region_history.append(regions)
            self._apply_region_matchings(clip, regions)

    def _apply_region_matchings(self, clip, regions):
        """
        Work out the best matchings between tracks and regions of interest for the current frame.
        Create any new tracks required.
        """
        unmatched_regions, matched_tracks = self._match_existing_tracks(clip, regions)
        new_tracks = self._create_new_tracks(clip, unmatched_regions)
        self._filter_inactive_tracks(clip, new_tracks, matched_tracks)

    def get_max_size_change(self, track, region):
        exiting = region.is_along_border and not track.last_bound.is_along_border
        entering = not exiting and track.last_bound.is_along_border

        min_change = 50
        if entering or exiting:
            self.print_if_verbose("entering {} or exiting {}".format(entering, exiting))
            min_change = 100
        max_size_change = np.clip(track.last_mass, min_change, 500)
        return max_size_change

    def _match_existing_tracks(self, clip, regions):

        scores = []
        used_regions = set()
        unmatched_regions = set(regions)
        for track in clip.active_tracks:
            for region in regions:
                score, size_change = track.get_track_region_score(
                    region, self.config.moving_vel_thresh
                )
                # we give larger tracks more freedom to find a match as they might move quite a bit.
                max_distance = np.clip(7 * track.last_mass, 900, 9025)
                max_size_change = self.get_max_size_change(track, region)

                if score > max_distance:
                    self.print_if_verbose(
                        "track {} distance score {} bigger than max score {}".format(
                            track.get_id(), score, max_distance
                        )
                    )

                    continue
                if size_change > max_size_change:
                    self.print_if_verbose(
                        "track {} size_change {} bigger than max size_change {}".format(
                            track.get_id(), size_change, max_size_change
                        )
                    )
                    continue
                scores.append((score, track, region))
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
                "Creating a new track {} with region {} mass{} area {}".format(
                    track.get_id(), region, track.last_bound.mass, track.last_bound.area
                )
            )
        return new_tracks

    def _filter_inactive_tracks(self, clip, new_tracks, matched_tracks):
        """ Filters tracks which are or have become inactive """

        unactive_tracks = clip.active_tracks - matched_tracks - new_tracks
        clip.active_tracks = matched_tracks | new_tracks
        for track in unactive_tracks:
            if (
                track.frames_since_target_seen + 1
                < self.config.remove_track_after_frames
            ):
                track.add_blank_frame(clip.frame_buffer)
                clip.active_tracks.add(track)
                self.print_if_verbose(
                    "frame {} adding a blacnk frame to {} ".format(
                        clip.frame_on, track.get_id()
                    )
                )

    def _get_regions_of_interest(
        self, clip, labels, stats, thresh, filtered, prev_filtered, mass
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
        edge = self.config.edge_pixels

        # find regions of interest
        regions = []
        for i in range(1, labels):

            region = Region(
                stats[i, 0],
                stats[i, 1],
                stats[i, 2],
                stats[i, 3],
                stats[i, 4],
                0,
                i,
                clip.frame_on,
            )
            # want the real mass calculated from before the dilation
            # region.mass = np.sum(region.subimage(thresh))
            region.mass = mass
            # Add padding to region and change coordinates from edgeless image -> full image
            region.x += edge - padding
            region.y += edge - padding
            region.width += padding * 2
            region.height += padding * 2

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
            if not self.filter_track(track, stats):
                good_tracks.append(track)

        clip.tracks = good_tracks
        self.print_if_verbose(
            "{} {}".format("Number of 'good' tracks", len(self.tracks))
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
                [("Too many tracks", track) for track in self.tracks[self.max_tracks :]]
            )
            clip.tracks = clip.tracks[: self.max_tracks]

    def filter_track(self, clip, track, stats):
        # discard any tracks that are less min_duration
        # these are probably glitches anyway, or don't contain enough information.
        if len(track) < self.config.min_duration_secs * 9:
            self.print_if_verbose("Track filtered. Too short, {}".format(len(track)))
            clip.filtered_tracks.append(("Track filtered.  Too much overlap", track))
            return True

        # discard tracks that do not move enough
        if stats.max_offset < self.config.track_min_offset:
            self.print_if_verbose("Track filtered.  Didn't move")
            clip.filtered_tracks.append(("Track filtered.  Didn't move", track))

            return True

        # discard tracks that do not have enough delta within the window (i.e. pixels that change a lot)
        if stats.delta_std < self.config.track_min_delta:
            self.print_if_verbose("Track filtered.  Too static")
            clip.filtered_tracks.append(("Track filtered.  Too static", track))
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
