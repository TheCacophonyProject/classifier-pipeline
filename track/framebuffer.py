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

import attr
import cv2
import numpy as np
from ml_tools.framecache import FrameCache
from track.track import TrackChannels
from ml_tools.tools import get_optical_flow_function, get_clipped_flow


@attr.s(slots=True)
class Frame:

    thermal = attr.ib()
    filtered = attr.ib()
    mask = attr.ib()
    frame_number = attr.ib()
    flow = attr.ib(default=None)
    flow_clipped = attr.ib(default=False)
    scaled_thermal = attr.ib(default=None)
    ffc_affected = attr.ib(default=False)

    @classmethod
    def from_array(
        cls, frame_arr, frame_number, flow_clipped=False, ffc_affected=False
    ):
        flow_h = frame_arr[TrackChannels.flow_h][:, :, np.newaxis]
        flow_v = frame_arr[TrackChannels.flow_v][:, :, np.newaxis]
        flow = np.concatenate((flow_h, flow_v), axis=2)
        return cls(
            frame_arr[TrackChannels.thermal],
            frame_arr[TrackChannels.filtered],
            frame_arr[TrackChannels.mask],
            frame_number,
            flow=flow,
            flow_clipped=flow_clipped,
            ffc_affected=ffc_affected,
        )

    def as_array(self, split_flow=True):
        if split_flow:
            return np.asarray(
                [
                    self.thermal,
                    self.filtered,
                    self.flow[:, :, 0]
                    if self.flow is not None
                    else np.zeros(self.filtered.shape),
                    self.flow[:, :, 1]
                    if self.flow is not None
                    else np.zeros(self.filtered.shape),
                    self.mask,
                ]
            )

        return np.asarray([self.thermal, self.filtered, self.flow, self.mask])

    def generate_optical_flow(self, opt_flow, prev_frame, flow_threshold=40):
        """
        Generate optical flow from thermal frames
        :param opt_flow: An optical flow algorithm
        """
        height, width = self.thermal.shape
        flow = np.zeros([height, width, 2], dtype=np.float32)
        threshold = np.median(self.thermal) + flow_threshold
        scaled_thermal = np.uint8(np.clip(self.thermal - threshold, 0, 255))
        if prev_frame is not None:
            # for some reason openCV spins up lots of threads for this which really slows things down, so we
            # cap the threads to 2
            cv2.setNumThreads(2)
            flow = opt_flow.calc(prev_frame.scaled_thermal, scaled_thermal, flow)
        self.scaled_thermal = scaled_thermal
        self.flow = flow
        if prev_frame:
            prev_frame.scaled_thermal = None

    def clip_flow(self):
        self.flow = get_clipped_flow(self.flow)
        self.flow_clipped = True

    def get_flow_split(self, clip_flow=False):
        if self.flow is not None:
            if self.clip_flow and not self.flow_clipped:
                flow_c = get_clipped_flow(self.flow)
                return flow_c[:, :, 0], flow_c[:, :, 1]

            else:
                return self.flow_h, self.flow_v
        return None, None

    def crop_by_region(self, region):
        # make a new frame cropped by region
        thermal = region.subimage(self.thermal)
        filtered = region.subimage(self.filtered)
        mask = region.subimage(self.mask)
        flow = None
        if self.flow is not None:
            flow = region.subimage(self.flow)
        frame = Frame(
            thermal,
            filtered,
            mask,
            self.frame_number,
            flow_clipped=self.flow_clipped,
            ffc_affected=self.ffc_affected,
        )
        frame.flow = flow
        return frame

    @property
    def flow_h(self):
        return self.flow[:, :, 0]

    @property
    def flow_v(self):
        return self.flow[:, :, 1]


class FrameBuffer:
    """ Stores entire clip in memory, required for some operations such as track exporting. """

    def __init__(
        self, cptv_name, high_quality_flow, cache_to_disk, calc_flow, keep_frames
    ):
        self.cache = FrameCache(cptv_name) if cache_to_disk else None
        self.opt_flow = None
        self.high_quality_flow = high_quality_flow
        self.frames = None
        self.prev_frame = None
        self.calc_flow = calc_flow
        self.keep_frames = keep_frames
        self.current_frame = 0
        if cache_to_disk or calc_flow:
            self.set_optical_flow()
        self.reset()

    def set_optical_flow(self):
        if self.opt_flow is None:
            self.opt_flow = get_optical_flow_function(self.high_quality_flow)

    def add_frame(self, thermal, filtered, mask, frame_number, ffc_affected=False):
        frame = Frame(thermal, filtered, mask, frame_number, ffc_affected=ffc_affected)
        if self.opt_flow:
            frame.generate_optical_flow(self.opt_flow, self.prev_frame)
        self.prev_frame = frame
        if self.keep_frames:
            if self.cache:
                self.cache.add_frame(frame)
            else:
                self.frames.append(frame)

    @property
    def has_flow(self):
        return self.cache or self.opt_flow

    def get_frame(self, frame_number):
        if self.prev_frame and self.prev_frame.frame_number == frame_number:
            return self.prev_frame
        elif self.cache:
            cache_frame, ffc_affected = self.cache.get_frame(frame_number)
            if cache_frame:
                return Frame.from_array(
                    cache_frame,
                    frame_number,
                    flow_clipped=True,
                    ffc_affected=ffc_affected,
                )
            return None
        if len(self.frames) > frame_number:
            return self.frames[frame_number]
        return None

    def close_cache(self):
        if self.cache:
            self.cache.close()

    def remove_cache(self):
        if self.cache:
            self.cache.delete()

    def get_last_frame(self):
        if self.cache and self.prev_frame:
            return self.prev_frame
        elif len(self.frames) > 0:
            return self.frames[-1]
        return None

    def get_last_filtered(self, region=None):

        if self.cache:
            if not self.prev_frame:
                return None
            prev = self.prev_frame.filtered
        else:
            if len(self.frames) > 0:
                prev = self.frames[-1].filtered
            else:
                return None

        if region:
            return region.subimage(prev)
        else:
            return prev

    def reset(self):
        """
        Empties buffer
        """
        self.frames = []

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        if self.cache:
            self.cache.open(mode="r")
        return self

    def __next__(self):
        frame = self.get_frame(self.current_frame)
        if frame is None:
            raise StopIteration

        self.current_frame += 1
        return frame
