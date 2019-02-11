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

import pytz
import datetime
import os

import numpy as np
import cv2
import h5py
import scipy.ndimage
import logging

from ml_tools.tools import get_image_subsection

from track.track import Track
from track.region import Region
from track.framebuffer import FrameBuffer

from cptv import CPTVReader


class TrackExtractor:
    """ Extracts tracks from a stream of frames. """

    def __init__(self, trackconfig):

        self.config = trackconfig

        # start time of video
        self.video_start_time = None
        # name of source file
        self.source_file = None
        # cptv reader
        self.reader = None
        # dictionary containing various statistics about the clip / tracking process.
        self.stats = {}
        # used to calculate optical flow
        self.opt_flow = None
        # how much hotter target must be from background to trigger a region of interest.
        self.threshold = 0
        # the current frame number
        self.frame_on = 0

        # per frame temperature statistics for thermal channel
        self.frame_stats_min = []
        self.frame_stats_max = []
        self.frame_stats_median = []
        self.frame_stats_mean = []

        # reason qwhy clip was rejected, or none if clip was accepted
        self.reject_reason = None

        self.max_tracks = trackconfig.max_tracks

        # a list of currently active tracks
        self.active_tracks = []
        # a list of all tracks
        self.tracks = []
        # list of regions for each frame
        self.region_history = []

        # if enabled will force the background subtraction algorithm off.
        self.disable_background_subtraction = False
        # rejects any videos that have non static backgrounds
        self.reject_non_static_clips = False

        # this buffers store the entire video in memory and are required for fast track exporting
        self.frame_buffer = FrameBuffer()

        # the previous filtered frame
        self._prev_filtered = None

        # accumulates frame changes for FM_DELTA algorithm
        self.accumulator = None

    def load(self, filename):
        """
        Loads a cptv file, and prepares for track extraction.
        """
        self.source_file = filename
        self.reader = CPTVReader(open(filename, 'rb'))
        local_tz = pytz.timezone('Pacific/Auckland')
        self.video_start_time = self.reader.timestamp.astimezone(local_tz)
        self.stats.update(self.get_video_stats())

    def extract_tracks(self):
        """
        Extracts tracks from given source.  Setting self.tracks to a list of good tracks within the clip
        :param source_file: filename of cptv file to process
        :returns: True if clip was successfully processed, false otherwise
        """

        assert self.reader, "Must call load before extracting tracks."

        self.reject_reason = None

        # we need to load the entire video so we can analyse the background.
        frames = [frame.pix for frame in self.reader]
        self.frame_buffer.thermal = frames

        # first we get the background.  This requires reading the entire source into memory.
        background, background_stats = self.analyse_background(frames)
        is_static_background = background_stats.background_deviation < self.config.static_background_threshold

        self.stats['threshold'] = background_stats.threshold
        self.stats['average_background_delta'] = background_stats.background_deviation
        self.stats['average_delta'] = background_stats.average_delta
        self.stats['mean_temp'] = background_stats.mean_temp
        self.stats['max_temp'] = background_stats.max_temp
        self.stats['min_temp'] = background_stats.min_temp
        self.stats['is_static'] = is_static_background

        self.threshold = background_stats.threshold

        # if the clip is moving then remove the estimated background and just use a threshold.
        if not is_static_background or self.disable_background_subtraction:
            background = None

        if len(frames) <= 9:
            self.reject_reason = "Clip too short {} frames".format(len(frames))
            return False

        if self.reject_non_static_clips and not is_static_background:
            self.reject_reason = "Non static background deviation={:.1f}".format(
                background_stats.background_deviation)
            return False

        # don't process clips that are too hot.
        if self.config.max_mean_temperature_threshold and background_stats.mean_temp > self.config.max_mean_temperature_threshold:
            self.reject_reason = "Mean temp too high {}".format(
                background_stats.mean_temp)
            return False

        # don't process clips with too large of a temperature difference
        if self.config.max_temperature_range_threshold and (background_stats.max_temp - background_stats.min_temp > self.config.max_temperature_range_threshold):
            self.reject_reason = "Temp delta too high {}".format(
                background_stats.max_temp - background_stats.min_temp)
            return False

        # reset the track ID so we start at 1
        Track._track_id = 1
        self.tracks = []
        self.active_tracks = []
        self.region_history = []

        # create optical flow
        self.opt_flow = cv2.createOptFlow_DualTVL1()
        self.opt_flow.setUseInitialFlow(True)
        if not self.config.high_quality_optical_flow:
            # see https://stackoverflow.com/questions/19309567/speeding-up-optical-flow-createoptflow-dualtvl1
            self.opt_flow.setTau(1 / 4)
            self.opt_flow.setScalesNumber(3)
            self.opt_flow.setWarpingsNumber(3)
            self.opt_flow.setScaleStep(0.5)

        # process each frame
        self.frame_on = 0
        for frame in frames:
            self.track_next_frame(frame, background)

        # filter out tracks that do not move, or look like noise
        self.filter_tracks()

        # apply smoothing if required
        if self.config.track_smoothing and len(frames) > 0:
            frame_height, frame_width = frames[0].shape
            for track in self.tracks:
                track.smooth(Rectangle(0, 0, frame_width, frame_height))

        return True

    def get_filtered(self, thermal, background=None):
        """
        Calculates filtered frame from thermal
        :param thermal: the thermal frame
        :param background: (optional) used for background subtraction
        :return: the filtered frame
        """

        thermal = np.float32(thermal)

        if background is None:
            filtered = thermal - np.median(thermal) - 40
            filtered[filtered < 0] = 0
        else:
            background = np.float32(background)
            filtered = thermal - background
            filtered[filtered < 0] = 0
            filtered = filtered - np.median(filtered)
            filtered[filtered < 0] = 0

        return filtered

    def track_next_frame(self, thermal, background=None):
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
        :param background: (optional) Background image, a numpy array of shape (height, width) and type uint16
            If specified background subtraction algorithm will be used.
        """

        thermal = np.float32(thermal)
        filtered = self.get_filtered(thermal, background)

        regions, mask = self.get_regions_of_interest(
            filtered, self._prev_filtered)

        # save frame stats
        self.frame_stats_min.append(np.min(thermal))
        self.frame_stats_max.append(np.max(thermal))
        self.frame_stats_median.append(np.median(thermal))
        self.frame_stats_mean.append(np.mean(thermal))

        # save history
        self.frame_buffer.filtered.append(np.float32(filtered))
        self.frame_buffer.mask.append(np.float32(mask))

        self.region_history.append(regions)

        self.apply_matchings(regions)
        # do we need this
        self._prev_filtered = filtered.copy()
        self.frame_on += 1

    def get_track_channels(self, track: Track, frame_number):
        """
        Gets frame channels for track at given frame number.  If frame number outside of track's lifespan an exception
        is thrown.  Requires the frame_buffer to be filled.
        :param track: the track to get frames for.
        :param frame_number: the frame number where 0 is the first frame of the track.
        :return: numpy array of size [channels, height, width] where channels are thermal, filtered, u, v, mask
        """

        if frame_number < 0 or frame_number >= len(track):
            raise ValueError("Frame {} is out of bounds for track with {} frames".format(
                frame_number, len(track))
            )

        bounds = track.bounds_history[frame_number]
        tracker_frame = track.start_frame + frame_number

        if tracker_frame < 0 or tracker_frame >= len(self.frame_buffer.thermal):
            raise ValueError("Track frame is out of bounds.  Frame {} was expected to be between [0-{}]".format(
                tracker_frame, len(self.frame_buffer.thermal)-1))

        thermal = get_image_subsection(
            self.frame_buffer.thermal[tracker_frame], bounds)
        filtered = get_image_subsection(
            self.frame_buffer.filtered[tracker_frame], bounds)
        flow = get_image_subsection(
            self.frame_buffer.flow[tracker_frame], bounds)
        mask = get_image_subsection(
            self.frame_buffer.mask[tracker_frame], bounds)

        # make sure only our pixels are included in the mask.
        mask[mask != bounds.id] = 0
        mask[mask > 0] = 1

        # stack together into a numpy array.
        # by using int16 we loose a little precision on the filtered frames, but not much (only 1 bit)
        frame = np.int16(
            np.stack((thermal, filtered, flow[:, :, 0], flow[:, :, 1], mask), axis=0))

        return frame

    def apply_matchings(self, regions):
        """
        Work out the best matchings between tracks and regions of interest for the current frame.
        Create any new tracks required.
        """
        scores = []
        for track in self.active_tracks:
            for region in regions:
                distance, size_change = track.get_track_region_score(region)

                # we give larger tracks more freedom to find a match as they might move quite a bit.
                max_distance = np.clip(7 * (track.mass ** 0.5), 30, 95)
                size_change = np.clip(track.mass, 50, 500)

                if distance > max_distance:
                    continue
                if size_change > size_change:
                    continue
                scores.append((distance, track, region))

        # apply matchings greedily.  Low score is best.
        matched_tracks = set()
        used_regions = set()
        new_tracks = set()

        scores.sort(key=lambda record: record[0])
        results = []

        for (score, track, region) in scores:
            # don't match a track twice
            if track in matched_tracks or region in used_regions:
                continue
            track.add_frame(region)
            used_regions.add(region)
            matched_tracks.add(track)
            results.append((track, score))

        # create new tracks for any unmatched regions
        for region in regions:
            if region in used_regions:
                continue
            # make sure we don't overlap with existing tracks.  This can happen if a tail gets tracked as a new object
            overlaps = [track.bounds.overlap_area(
                region) for track in self.active_tracks]
            if len(overlaps) > 0 and max(overlaps) > (region.area * 0.25):
                continue
            track = Track()
            track.add_frame(region)
            track.start_time = self.video_start_time + \
                datetime.timedelta(seconds=self.frame_on / 9.0)
            track.start_frame = self.frame_on
            new_tracks.add(track)
            self.active_tracks.append(track)
            self.tracks.append(track)

        # check if any tracks did not find a matched region
        for track in [track for track in self.active_tracks if track not in matched_tracks and track not in new_tracks]:
            # we lost this track.  start a count down, and if we don't get it back soon remove it
            track.frames_since_target_seen += 1
            track.add_blank_frame()

        # remove any tracks that have not seen their target in a while
        self.active_tracks = [
            track for track in self.active_tracks if track.frames_since_target_seen < self.config.remove_track_after_frames]

    def filter_tracks(self):

        for track in self.tracks:
            track.trim()

        track_stats = [(track.get_stats(), track) for track in self.tracks]
        track_stats.sort(reverse=True, key=lambda record: record[0].score)

        if self.config.verbose:
            for stats, track in track_stats:
                print(" - track duration:{:.1f}sec offset:{:.1f}px delta:{:.1f} mass:{:.1f}px".format(
                    stats.duration, stats.max_offset, stats.delta_std, stats.average_mass
                ))

        # find how much each track overlaps with other tracks

        track_overlap_ratio = {}

        for track in self.tracks:
            highest_ratio = 0
            for other in self.tracks:
                if track == other:
                    continue
                highest_ratio = max(
                    track.get_overlap_ratio(other), highest_ratio)
            track_overlap_ratio[track] = highest_ratio

        # filter out tracks that probably are just noise.
        good_tracks = []
        self.print_if_verbose("{} {}".format(
            "Number of tracks before filtering", len(self.tracks)))

        for stats, track in track_stats:
            # discard any tracks that overlap too often with other tracks.  This normally means we are tracking the
            # tail of an animal.
            if track_overlap_ratio[track] > self.config.track_smoothing:
                self.print_if_verbose("Track filtered.  Too much overlap")
                continue

            # discard any tracks that are less min_dirration
            # these are probably glitches anyway, or don't contain enough information.
            if stats.duration < self.config.min_duration_secs:
                self.print_if_verbose("Track filtered. Too short")
                continue

            # discard tracks that do not move enough
            if stats.max_offset < self.config.track_min_offset:
                self.print_if_verbose("Track filtered.  Didn't move")
                continue

            # discard tracks that do not have enough delta within the window (i.e. pixels that change a lot)
            if stats.delta_std < self.config.track_min_delta:
                self.print_if_verbose("Track filtered.  Too static")
                continue

            # discard tracks that do not have enough enough average mass.
            if stats.average_mass < self.config.track_min_mass:
                self.print_if_verbose("Track filtered.  Too small (mass)")
                continue

            good_tracks.append(track)

        self.tracks = good_tracks

        self.print_if_verbose("{} {}".format(
            "Number of 'good' tracks", len(self.tracks)))
        # apply max_tracks filter
        # note, we take the n best tracks.
        if self.max_tracks is not None and self.max_tracks < len(self.tracks):
            logging.warning(
                " -using only {0} tracks out of {1}".format(self.max_tracks, len(self.tracks)))
            self.tracks = self.tracks[:self.max_tracks]

    def get_regions_of_interest(self, filtered, prev_filtered=None):
        """
        Calculates pixels of interest mask from filtered image, and returns both the labeled mask and their bounding
        rectangles.
        :param filtered: The filtered frame
        :param prev_filtered: The previous filtered frame, required for pixel deltas to be calculated
        :return: regions of interest, mask frame
        """

        frame_height, frame_width = filtered.shape

        # get frames change
        if prev_filtered is not None:
            # we need a lot of precision because the values are squared.  Float32 should work.
            delta_frame = np.abs(np.float32(filtered) -
                                 np.float32(prev_filtered))
        else:
            delta_frame = None

        edge = self.config.edge_pixels
        if edge > 0:
            edgeless_filtered = filtered[edge:frame_height - edge, edge:frame_width - edge]
        else:
            edge = 0
            edgeless_filtered = filtered

        thresh = np.uint8(apply_threshold(edgeless_filtered, threshold=self.threshold))

        # applies erosion
        erosion = 0

        if erosion:
            # this removes small slivers
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations=erosion)

        labels, small_mask, stats, _ = cv2.connectedComponentsWithStats(
            thresh)

        mask = np.zeros(filtered.shape)
        mask[edge:frame_height - edge, edge:frame_width - edge] = small_mask

        # we enlarge the rects a bit, partly because we eroded them previously, and partly because we want some context.
        padding = self.config.frame_padding

        # find regions of interest
        regions = []
        for i in range(1, labels):

            region = Region(
                stats[i, 0] - padding + edge,
                stats[i, 1] - padding + edge,
                stats[i, 2] + padding * 2,
                stats[i, 3] + padding * 2,
                stats[i, 4], 0, i, self.frame_on
            )

            old_region = region.copy()
            if self.config.cropped_regions_strategy == "all":
                # in this case we just clip the region to the edges of the screen
                region.left = max(region.left, edge)
                region.top = max(region.top, edge)
                region.right = min(region.right, frame_width - 1 - edge)
                region.bottom = min(region.bottom, frame_height - 1 - edge)
            elif self.config.cropped_regions_strategy == "cautious":
                # keep cropped regions if they have a reasonable size
                region.left = max(region.left, edge)
                region.top = max(region.top, edge)
                region.right = min(region.right, frame_width - 1 - edge)
                region.bottom = min(region.bottom, frame_height - 1 - edge)

                crop_width_fraction = (
                    old_region.width - region.width) / old_region.width
                crop_height_fraction = (
                    old_region.height - region.height) / old_region.height

                if crop_width_fraction > 0.25 or crop_height_fraction > 0.25:
                    continue

            elif self.config.cropped_regions_strategy == "none":
                # all regions that touch the side of the screen are removed
                if (region.left < edge) or (region.right > frame_width - edge) or (region.top < edge) or (region.bottom > frame_height - edge):
                    continue
            else:
                raise ValueError(
                    "Invalid mode for CROPPED_REGIONS_STRATEGY, expected ['all','cautious','none'] but found {}".format(
                        self.config.cropped_regions_strategy))
            region.was_cropped = str(old_region) != str(region)

            if delta_frame is not None:
                region_difference = np.float32(
                    get_image_subsection(delta_frame, region))
                region.pixel_variance = np.var(region_difference)

            # filter out regions that are probably just noise
            if region.pixel_variance < 2.0 and region.mass < 8:
                continue

            regions.append(region)

        return regions, mask

    def print_if_verbose(self, info_string):
        if self.config.verbose:
            print(info_string)

    def get_video_stats(self):
        """
        Extracts useful statics from video clip.
        :returns: a dictionary containing the video statistics.
        """
        local_tz = pytz.timezone('Pacific/Auckland')
        result = {}
        result['date_time'] = self.video_start_time.astimezone(local_tz)
        result['is_night'] = self.video_start_time.astimezone(
            local_tz).time().hour >= 21 or self.video_start_time.astimezone(local_tz).time().hour <= 4

        return result

    def analyse_background(self, frames_list):
        """
        Runs through all provided frames and estimates the background, consuming all the source frames.
        :param frames_list: a list of numpy array frames
        :return: background, background_stats
        """

        # note: unfortunately this must be done before any other processing, which breaks the streaming architecture
        # for this reason we must return all the frames so they can be reused

        frames = np.float32(frames_list)
        background = np.percentile(frames, q=10, axis=0)
        filtered = np.float32([self.get_filtered(frame, background)
                               for frame in frames_list])

        delta = np.asarray(frames[1:], dtype=np.float32) - \
            np.asarray(frames[:-1], dtype=np.float32)
        average_delta = float(np.mean(np.abs(delta)))

        # take half the max filtered value as a threshold
        threshold = float(np.percentile(np.reshape(
            filtered, [-1]), q=self.config.threshold_percentile) / 2)

        # cap the threshold to something reasonable
        if threshold < self.config.min_threshold:
            threshold = self.config.min_threshold
        if threshold > self.config.max_threshold:
            threshold = self.config.max_threshold

        background_stats = BackgroundAnalysis()
        background_stats.threshold = float(threshold)
        background_stats.average_delta = float(average_delta)
        background_stats.min_temp = float(np.min(frames))
        background_stats.max_temp = float(np.max(frames))
        background_stats.mean_temp = float(np.mean(frames))
        background_stats.background_deviation = float(
            np.mean(np.abs(filtered)))

        return background, background_stats


def apply_threshold(frame, threshold):
    """
    Creates a binary mask out of an image by applying a threshold.
    Any pixels more than the threshold are set 1, all others are set to 0.
    A blur is also applied as a filtering step
    """
    thresh = cv2.GaussianBlur(np.float32(frame), (5, 5), 0) - threshold
    thresh[thresh < 0] = 0
    thresh[thresh > 0] = 1
    return thresh


class BackgroundAnalysis:
    """ Stores background analysis statistics. """

    def __init__(self):
        self.threshold = None
        self.average_delta = None
        self.max_temp = None
        self.min_temp = None
        self.mean_temp = None
        self.background_deviation = None
