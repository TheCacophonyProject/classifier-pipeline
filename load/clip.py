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


import datetime
import logging
import numpy as np
import os
import pytz

from cptv import CPTVReader
import cv2

import ml_tools.tools as tools
from ml_tools.tools import Rectangle
from track.region import Region
from track.framebuffer import FrameBuffer
from track.track import Track


class Clip:
    PREVIEW = "preview"
    FRAMES_PER_SECOND = 9
    local_tz = pytz.timezone("Pacific/Auckland")
    VERSION = 7

    def __init__(self, trackconfig, cache_to_disk=False, calc_flow=False):
        self._id = None
        self.preview_frames = None
        self.from_metadata = False
        self.background_is_preview = trackconfig.background_calc == Clip.PREVIEW
        self.config = trackconfig
        self.frames_per_second = Clip.FRAMES_PER_SECOND
        # start time of video
        self.video_start_time = None
        # name of source file
        self.source_file = None
        self.location = None
        self.stats = ClipStats()
        # per frame temperature statistics for thermal channel

        # this buffers store the entire video in memory and are required for fast track exporting
        self.mean_background_value = 0.0
        self.device = None
        self.cache_to_disk = cache_to_disk
        self.track_extractor = None
        self.frames = 0
        self.threshold = trackconfig.delta_thresh
        self.flow = calc_flow

    def get_id(self):
        return str(self._id)

    def parse_clip_meta(
        self, filename, metadata, include_filtered_channel, tag_precedence
    ):
        tracks = self.load_metadata(metadata, include_filtered_channel, tag_precedence)
        self.parse_clip(filename, tracks)

    def parse_clip(self, filename, tracks=None):
        """
        Loads a cptv file, and prepares for track extraction.
        """

        self.track_extractor = ClipTrackExtractor(
            self,
            self.config,
            self.cache_to_disk,
            self.from_metadata,
            self.config.high_quality_optical_flow,
            meta_tracks=tracks,
        )

        self.source_file = filename
        res_x = None
        res_y = None
        # self.frame_buffer = FrameBuffer(filename, self.opt_flow, self.cache_to_disk)
        with open(filename, "rb") as f:
            reader = CPTVReader(f)
            self.video_start_time = reader.timestamp.astimezone(Clip.local_tz)
            self.preview_frames = (
                reader.preview_secs * self.frames_per_second - self.config.ignore_frames
            )
            self.set_video_stats()
            # we need to load the entire video so we can analyse the background.
            edge = self.config.edge_pixels
            res_x = reader.x_resolution
            res_y = reader.y_resolution
            edge = self.config.edge_pixels

            crop_rectangle = Rectangle(edge, edge, res_x - 2 * edge, res_y - 2 * edge)
            self.track_extractor.crop_rectangle = crop_rectangle
            if self.background_is_preview:
                for frame in reader:
                    self.track_extractor.process_frame(frame.pix)
            else:
                self.track_extractor.extract_tracks(
                    [np.float32(frame.pix) for frame in reader]
                )
            # frames = [np.float32(frame.pix) for frame in reader]
            # self.frame_buffer.thermal = frames
        self.stats.completed(self.track_extractor.frame_on, res_y, res_x)

    def set_video_stats(self):
        """
        Extracts useful statics from video clip.
        """
        self.stats.date_time = self.video_start_time.astimezone(Clip.local_tz)
        self.stats.is_night = (
            self.video_start_time.astimezone(Clip.local_tz).time().hour >= 2
        )

    def load_metadata(self, metadata, include_filtered_channel, tag_precedence):
        self._id = metadata["id"]
        device_meta = metadata.get("Device")
        if device_meta:
            self.device = device_meta.get("devicename")
        else:
            self.device = os.path.splitext(os.path.basename(self.source_file))[0].split(
                "-"
            )[-1]

        self.location = metadata.get("location")
        tracks = self.load_tracks_meta(
            metadata, include_filtered_channel, tag_precedence
        )
        self.from_metadata = True
        return tracks

    def load_tracks_meta(self, metadata, include_filtered_channel, tag_precedence):
        tracks_meta = metadata["Tracks"]
        tracks = []
        # get track data
        for track_meta in tracks_meta:
            track = Track(self.get_id())
            if track.load_track_meta(
                track_meta,
                self.frames_per_second,
                include_filtered_channel,
                tag_precedence,
                self.config.min_tag_confidence,
            ):
                tracks.append(track)
        return tracks

    def start_and_end_in_secs(self, track):
        if not track.start_s:
            track.start_s = track.start_frame / self.frames_per_second

        if not track.end_s:
            track.end_s = (track.end_frame + 1) / self.frames_per_second

        return (track.start_s, track.end_s)

    def start_and_end_time_absolute(self, start_s=0, end_s=None):
        if not end_s:
            end_s = (
                len(self.track_extractor.frame_buffer.frames) / self.frames_per_second
            )
        return (
            self.video_start_time + datetime.timedelta(seconds=start_s),
            self.video_start_time + datetime.timedelta(seconds=end_s),
        )

    def print_if_verbose(self, info_string):
        if self.config.verbose:
            logging.info(info_string)


class ClipTrackExtractor:
    def __init__(
        self,
        clip,
        config,
        cache_to_disk,
        from_metadata,
        high_quality_flow,
        meta_tracks=None,
    ):
        self.clip = clip
        self.cache_to_disk = cache_to_disk
        self.from_metadata = from_metadata
        self.frame_buffer = FrameBuffer(
            clip.source_file, high_quality_flow, cache_to_disk, clip.flow
        )
        self.frame_on = 0
        self.preview_frames = []
        self.background = None
        self.config = config
        self.stats = clip.stats
        self.tracks = meta_tracks if meta_tracks else []
        self.active_tracks = []
        self.crop_rectangle = None
        self.max_tracks = config.max_tracks
        self.filtered_tracks = []
        self.threshold = config.delta_thresh
        self.stats.threshold = self.threshold
        self.region_history = []

        # frame_padding < 3 causes problems when we get small areas...
        self.frame_padding = max(3, self.config.frame_padding)
        # the dilation effectively also pads the frame so take it into consideration.
        self.frame_padding = max(0, self.frame_padding - self.config.dilation_pixels)

    def process_frame(self, frame):
        if self.clip.preview_frames > 0 and self.frame_on < self.clip.preview_frames:
            self._calculate_preview_from_frame(frame)
        else:
            self._process_frame(frame, self.frame_on, self.background)
        self.frame_on += 1

    def _calculate_preview_from_frame(self, frame):
        self.preview_frames.append(frame)
        if self.background is None:
            self.background = frame
        else:
            self.background = np.minimum(self.background, frame)

        if self.frame_on == (self.clip.preview_frames - 1):
            self.stats.mean_background_value = np.average(self.background)
            self.set_config_from_background()
            for i, back_frame in enumerate(self.preview_frames):
                self._process_frame(back_frame, i, self.background)
            self.preview_frames = None

    def _background_from_preview(self, frame_list):
        number_frames = self.clip.preview_frames
        if not number_frames < len(frame_list):
            logging.error("Video consists entirely of preview")
            number_frames = len(frame_list)
        frames = frame_list[0:number_frames]
        background = np.min(frames, axis=0)
        background = np.int32(np.rint(background))
        self.stats.mean_background_value = np.average(background)
        self.set_config_from_background()
        return background

    def set_config_from_background(self):
        if self.config.dynamic_thresh:
            self.config.temp_thresh = min(
                self.config.temp_thresh, self.stats.mean_background_value
            )
        self.stats.temp_thresh = self.config.temp_thresh

    def extract_tracks(self, frames):
        # for now just always calculate as we are using the stats...
        # background np.float64[][] filtered calculated here and stats
        background = self._background_from_whole_clip(frames)

        if self.clip.background_is_preview:
            if self.clip.preview_frames > 0:
                # background np.int32[][]
                background = self._background_from_preview(frames)
            else:
                logging.info(
                    "No preview secs defined for CPTV file - using statistical background measurement"
                )

        # process each frame
        for frame_number, frame in enumerate(frames):
            self._process_frame(frame, frame_number, background)

        if not self.from_metadata:
            self.filter_tracks()
            # apply smoothing if required
            if self.config.track_smoothing and len(frames) > 0:
                frame_height, frame_width = frames[0].shape
                for track in self.tracks:
                    track.smooth(Rectangle(0, 0, frame_width, frame_height))

    def _background_from_whole_clip(self, frames):
        """
        Runs through all provided frames and estimates the background, consuming all the source frames.
        :param frames_list: a list of numpy array frames
        :return: background
        """

        # note: unfortunately this must be done before any other processing, which breaks the streaming architecture
        # for this reason we must return all the frames so they can be reused

        # [][] array
        background = np.percentile(frames, q=10, axis=0)
        filtered = np.float32(
            [self._get_filtered_frame(frame, background) for frame in frames]
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

        self.threshold = threshold
        self.stats.threshold = threshold
        self.stats.temp_thresh = self.config.temp_thresh
        self.stats.average_delta = float(average_delta)
        self.stats.filtered_deviation = float(np.mean(np.abs(filtered)))
        self.stats.is_static_background = (
            self.stats.filtered_deviation < self.config.static_background_threshold
        )

        if (
            not self.stats.is_static_background
            or self.clip.disable_background_subtraction
        ):
            background = None
        self.set_config_from_background()
        return background

    def _get_filtered_frame(self, thermal, background=None):
        """
        Calculates filtered frame from thermal
        :param thermal: the thermal frame
        :param background: (optional) used for background subtraction
        :return: the filtered frame
        """

        # has to be a signed int so we dont get overflow
        filtered = np.float32(thermal.copy())
        if background is None:
            filtered = filtered - np.median(filtered) - 40
            filtered[filtered < 0] = 0
        elif self.clip.background_is_preview:
            avg_change = int(
                round(np.average(thermal) - self.stats.mean_background_value)
            )
            filtered[filtered < self.stats.temp_thresh] = 0
            filtered = filtered - background - avg_change
            # filtered[filtered < 0] = 0
        else:
            filtered = filtered - background
            filtered[filtered < 0] = 0
            filtered = filtered - np.median(filtered)
            filtered[filtered < 0] = 0

        return filtered

    def _process_frame(self, thermal, frame_number, background=None):
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
        :param background: (optional) Background image, a numpy array of shape (height, width) and type uint16
            If specified background subtraction algorithm will be used.
        """

        filtered = self._get_filtered_frame(thermal, background)
        frame_height, frame_width = filtered.shape

        mask = np.zeros(filtered.shape)
        edge = self.config.edge_pixels

        # remove the edges of the frame as we know these pixels can be spurious value
        edgeless_filtered = self.crop_rectangle.subimage(filtered)
        thresh = np.uint8(
            tools.blur_and_return_as_mask(edgeless_filtered, threshold=self.threshold)
        )

        dilated = thresh

        # Dilation groups interested pixels that are near to each other into one component(animal/track)
        if self.config.dilation_pixels > 0:
            size = self.config.dilation_pixels * 2 + 1
            kernel = np.ones((size, size), np.uint8)
            dilated = cv2.dilate(dilated, kernel, iterations=1)

        labels, small_mask, stats, _ = cv2.connectedComponentsWithStats(dilated)

        mask[edge : frame_height - edge, edge : frame_width - edge] = small_mask
        self.stats.add_frame(thermal, filtered)
        # save history
        prev_filtered = self.frame_buffer.get_last_filtered()
        self.frame_buffer.add_frame(thermal, filtered, mask)
        if self.from_metadata:
            for track in self.tracks:
                if frame_number in track.frame_list:
                    track.add_frame(
                        self.frame_buffer.get_last_frame(),
                        self.threshold,
                        prev_filtered,
                    )
        else:
            regions = self._get_regions_of_interest(
                labels, stats, thresh, filtered, prev_filtered, frame_number
            )
            self.region_history.append(regions)
            self._apply_region_matchings(regions, frame_number)

    def _apply_region_matchings(self, regions, frame_number):
        """
        Work out the best matchings between tracks and regions of interest for the current frame.
        Create any new tracks required.
        """
        used_regions, matched_tracks = self._match_existing_tracks(regions)
        self._create_new_tracks(regions, used_regions, matched_tracks, frame_number)

    def _match_existing_tracks(self, regions):

        scores = []
        used_regions = []
        matched_tracks = []
        for track in self.active_tracks:
            for region in regions:
                distance, size_change = track.get_track_region_score(
                    region, self.config.moving_vel_thresh
                )

                # we give larger tracks more freedom to find a match as they might move quite a bit.
                max_distance = np.clip(7 * (track.last_mass ** 0.5), 30, 95)
                max_size_change = np.clip(track.last_mass, 50, 500)
                if distance > max_distance:
                    continue
                if size_change > max_size_change:
                    continue
                scores.append((distance, track, region))

        scores.sort(key=lambda record: record[0])

        for (score, track, region) in scores:
            if track in matched_tracks or region in used_regions:
                continue
            track.add_frame_from_region(region, self.frame_buffer.prev_frame)
            matched_tracks.append(track)
            used_regions.append(region)

        return used_regions, matched_tracks

    def _create_new_tracks(self, regions, used_regions, matched_tracks, frame_number):
        """ Create new tracks for any unmatched regions """
        new_tracks = []
        for region in regions:
            if region in used_regions:
                continue
            # make sure we don't overlap with existing tracks.  This can happen if a tail gets tracked as a new object
            overlaps = [
                track.last_bound.overlap_area(region) for track in self.active_tracks
            ]
            if len(overlaps) > 0 and max(overlaps) > (region.area * 0.25):
                continue

            track = Track(self.clip.get_id())
            track.start_frame = frame_number
            track.start_s = frame_number * self.clip.frames_per_second
            track.add_frame_from_region(region, self.frame_buffer.prev_frame)
            new_tracks.append(track)
            self.active_tracks.append(track)
            self.tracks.append(track)

        # check if any tracks did not find a matched region
        for track in [
            track
            for track in self.active_tracks
            if track not in matched_tracks and track not in new_tracks
        ]:
            # we lost this track.  start a count down, and if we don't get it back soon remove it
            track.frames_since_target_seen += 1
            track.add_blank_frame(self.frame_buffer)

        # remove any tracks that have not seen their target in a while
        self.active_tracks = [
            track
            for track in self.active_tracks
            if track.frames_since_target_seen < self.config.remove_track_after_frames
        ]

    def _get_regions_of_interest(
        self, labels, stats, thresh, filtered, prev_filtered, frame_number
    ):
        """
        Calculates pixels of interest mask from filtered image, and returns both the labeled mask and their bounding
        rectangles.
        :param filtered: The filtered frame
=        :return: regions of interest, mask frame
        """

        frame_height, frame_width = filtered.shape
        # get frames change
        if prev_filtered is not None:
            # we need a lot of precision because the values are squared.  Float32 should work.
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
                frame_number,
            )

            # want the real mass calculated from before the dilation
            region.mass = np.sum(region.subimage(thresh))

            # Add padding to region and change coordinates from edgeless image -> full image
            region.x += edge - padding
            region.y += edge - padding
            region.width += padding * 2
            region.height += padding * 2

            old_region = region.copy()
            region.crop(self.crop_rectangle)
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

    def filter_tracks(self):

        for track in self.tracks:
            track.trim()

        track_stats = [(track.get_stats(), track) for track in self.tracks]
        track_stats.sort(reverse=True, key=lambda record: record[0].score)

        if self.config.verbose:
            for stats, track in track_stats:
                start_s, end_s = self.clip.start_and_end_in_secs(track)
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
            "{} {}".format("Number of tracks before filtering", len(self.tracks))
        )

        for stats, track in track_stats:
            # discard any tracks that overlap too often with other tracks.  This normally means we are tracking the
            # tail of an animal.
            if not self.filter_track(track, stats):
                good_tracks.append(track)

        self.tracks = good_tracks

        self.print_if_verbose(
            "{} {}".format("Number of 'good' tracks", len(self.tracks))
        )
        # apply max_tracks filter
        # note, we take the n best tracks.
        if self.max_tracks is not None and self.max_tracks < len(self.tracks):
            logging.warning(
                " -using only {0} tracks out of {1}".format(
                    self.max_tracks, len(self.tracks)
                )
            )
            self.filtered_tracks.extend(
                [("Too many tracks", track) for track in self.tracks[self.max_tracks :]]
            )
            self.tracks = self.tracks[: self.max_tracks]

    def filter_track(self, track, stats):
        # discard any tracks that are less min_duration
        # these are probably glitches anyway, or don't contain enough information.
        if len(track) < self.config.min_duration_secs * 9:
            self.print_if_verbose("Track filtered. Too short, {}".format(len(track)))
            self.filtered_tracks.append(("Track filtered.  Too much overlap", track))
            return True

        # discard tracks that do not move enough
        if stats.max_offset < self.config.track_min_offset:
            self.print_if_verbose("Track filtered.  Didn't move")
            self.filtered_tracks.append(("Track filtered.  Didn't move", track))

            return True

        # discard tracks that do not have enough delta within the window (i.e. pixels that change a lot)
        if stats.delta_std < self.config.track_min_delta:
            self.print_if_verbose("Track filtered.  Too static")
            self.filtered_tracks.append(("Track filtered.  Too static", track))
            return True

        # discard tracks that do not have enough enough average mass.
        if stats.average_mass < self.config.track_min_mass:
            self.print_if_verbose(
                "Track filtered.  Mass too small ({})".format(stats.average_mass)
            )
            self.filtered_tracks.append(("Track filtered.  Mass too small", track))

            return True

        highest_ratio = 0
        for other in self.tracks:
            if track == other:
                continue
            highest_ratio = max(track.get_overlap_ratio(other), highest_ratio)

        if highest_ratio > self.config.track_overlap_ratio:
            self.print_if_verbose(
                "Track filtered.  Too much overlap {}".format(highest_ratio)
            )
            self.filtered_tracks.append(("Track filtered.  Too much overlap", track))
            return True

        return False

    def print_if_verbose(self, info_string):
        if self.config.verbose:
            logging.info(info_string)


class ClipStats:
    """ Stores background analysis statistics. """

    def __init__(self):
        self.mean_background_value = 0
        self.max_temp = None
        self.min_temp = None
        self.mean_temp = None
        self.frame_stats_min = []
        self.frame_stats_max = []
        self.frame_stats_median = []
        self.frame_stats_mean = []
        self.filtered_deviation = None
        self.filtered_sum = 0
        self.temp_thresh = 0
        self.threshold = None
        self.average_delta = None
        self.is_static_background = None

    def add_frame(self, thermal, filtered):
        f_median = np.median(thermal)
        f_max = np.max(thermal)
        f_min = np.min(thermal)
        f_mean = np.mean(thermal)
        self.max_temp = null_safe_compare(self.max_temp, f_max, max)
        self.min_temp = null_safe_compare(self.min_temp, f_min, min)

        self.frame_stats_min.append(f_min)
        self.frame_stats_max.append(f_max)
        self.frame_stats_median.append(f_median)
        self.frame_stats_mean.append(f_mean)
        self.filtered_sum += np.sum(np.abs(filtered))

    def completed(self, num_frames, height, width):
        if num_frames == 0:
            return
        total = num_frames * height * width
        self.filtered_deviation = self.filtered_sum / float(total)
        self.mean_temp = (height * width * sum(self.frame_stats_mean)) / float(total)


def null_safe_compare(a, b, cmp):
    if a is None:
        return b
    elif b:
        return cmp(a, b)
    else:
        return None
