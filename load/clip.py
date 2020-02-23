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


from ml_tools.tools import Rectangle
from track.framebuffer import FrameBuffer
from track.track import Track


class Clip:
    PREVIEW = "preview"
    FRAMES_PER_SECOND = 9
    local_tz = pytz.timezone("Pacific/Auckland")
    VERSION = 7
    CLIP_ID = 1

    def __init__(self, trackconfig, sourcefile, background=None, calc_stats=True):
        self._id = Clip.CLIP_ID
        Clip.CLIP_ID += 1
        Track._track_id = 1
        self.disable_background_subtraction = False
        self.frame_on = 0
        self.ffc_affected = False
        self.crop_rectangle = None
        self.num_preview_frames = 0
        self.preview_frames = []
        self.region_history = []
        self.active_tracks = set()
        self.tracks = []
        self.filtered_tracks = []
        self.from_metadata = False
        self.video_start_time = None
        self.location = None
        self.frame_buffer = None
        self.device = None
        self.background = None
        self.background_calculated = False
        self.res_x = None
        self.res_y = None
        self.background_frames = 0
        self.background_is_preview = trackconfig.background_calc == Clip.PREVIEW
        self.config = trackconfig
        self.frames_per_second = Clip.FRAMES_PER_SECOND

        self.calc_stats = calc_stats
        self.source_file = sourcefile
        self.stats = ClipStats()
        self.threshold = trackconfig.delta_thresh
        self.stats.threshold = self.threshold

        self.temp_thresh = self.config.temp_thresh

        if background is not None:
            self.background = background
            self._set_from_background()

    def _set_from_background(self):
        self.stats.mean_background_value = np.average(self.background)
        self.set_temp_thresh()
        self.background_calculated = True

    def on_preview(self):
        return not self.background_calculated

    def calculate_preview_from_frame(self, frame, ffc_affected=False):
        self.preview_frames.append((frame, ffc_affected))
        if ffc_affected:
            return
        if self.background is None:
            self.background = frame
        else:
            self.background = np.minimum(self.background, frame)
        if self.background_frames == (self.num_preview_frames - 1):
            self._set_from_background()
        self.background_frames += 1

    def background_from_frames(self, raw_frames):
        number_frames = self.num_preview_frames
        if not number_frames < len(raw_frames):
            logging.error("Video consists entirely of preview")
            number_frames = len(raw_frames)
        frames = [np.float32(frame.pix) for frame in raw_frames[0:number_frames]]
        self.background = np.min(frames, axis=0)
        self.background = np.int32(np.rint(self.background))

        self._set_from_background()

    def background_from_whole_clip(self, frames):
        """
        Runs through all provided frames and estimates the background, consuming all the source frames.
        :param frames_list: a list of numpy array frames
        :return: background
        """

        # note: unfortunately this must be done before any other processing, which breaks the streaming architecture
        # for this reason we must return all the frames so they can be reused

        # [][] array

        self.background = np.percentile(frames, q=10, axis=0)
        self._set_from_background()

    def _add_active_track(self, track):
        self.active_tracks.add(track)
        self.tracks.append(track)

    def get_id(self):
        return str(self._id)

    def set_temp_thresh(self):
        if self.config.dynamic_thresh:
            self.stats.temp_thresh = min(
                self.config.temp_thresh, self.stats.mean_background_value
            )
            self.temp_thresh = self.stats.temp_thresh
        else:
            self.temp_thresh = self.config.temp_thresh

    def set_video_stats(self, video_start_time):
        """
        Extracts useful statics from video clip.
        """
        self.video_start_time = video_start_time
        self.stats.date_time = video_start_time.astimezone(Clip.local_tz)
        self.stats.is_night = (
            video_start_time.astimezone(Clip.local_tz).time().hour >= 2
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
        self.tracks = set(tracks)

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
            logging.info("track start frame setting from {}".format(track.start_frame))

        if not track.end_s:
            track.end_s = (track.end_frame + 1) / self.frames_per_second

        return (track.start_s, track.end_s)

    def set_frame_buffer(self, high_quality_flow, cache_to_disk, use_flow, keep_frames):
        self.frame_buffer = FrameBuffer(
            self.source_file, high_quality_flow, cache_to_disk, use_flow, keep_frames
        )

    def set_res(self, res_x, res_y):
        self.res_x = res_x
        self.res_y = res_y
        self._set_crop_rectangle()

    def _set_crop_rectangle(self):

        edge = self.config.edge_pixels
        self.crop_rectangle = Rectangle(
            edge, edge, self.res_x - 2 * edge, self.res_y - 2 * edge
        )

    def start_and_end_time_absolute(self, start_s=0, end_s=None):
        if not end_s:
            end_s = len(self.frame_buffer.frames) / self.frames_per_second
        return (
            self.video_start_time + datetime.timedelta(seconds=start_s),
            self.video_start_time + datetime.timedelta(seconds=end_s),
        )

    def print_if_verbose(self, info_string):
        if self.config.verbose:
            logging.info(info_string)

    def add_frame(self, thermal, filtered, mask, ffc_affected=False):
        self.frame_buffer.add_frame(
            thermal, filtered, mask, self.frame_on, ffc_affected
        )
        if self.calc_stats:
            self.stats.add_frame(thermal, filtered)


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
        f_mean = np.nanmean(thermal)
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


def null_safe_compare(a, b, cmp_fn):
    if a is None:
        return b
    elif b is not None:
        return cmp_fn(a, b)
    else:
        return None
