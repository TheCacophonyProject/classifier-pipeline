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


import numpy as np
import cv2
import h5py
import numpy as np
from ml_tools.framecache import FrameCache


class Frame:
    def __init__(self, thermal, filtered, mask, frame_number):
        self.thermal = thermal
        self.filtered = filtered
        self.frame_number = frame_number
        self.mask = mask
        self.flow = None
        self.clipped_temp = None

    def generate_optical_flow(self, opt_flow, prev_frame, flow_threshold=40):
        """
        Generate optical flow from thermal frames
        :param opt_flow: An optical flow algorithm
        """
        height, width = self.thermal.shape
        flow = np.zeros([height, width, 2], dtype=np.float32)

        prev = prev_frame.clipped_temp if prev_frame else None
        prev_flow = prev_frame.flow if prev_frame else None

        frame = self.thermal
        threshold = np.median(frame) + flow_threshold
        current = np.uint8(np.clip(frame - threshold, 0, 255))

        if prev is not None:
            # for some reason openCV spins up lots of threads for this which really slows things down, so we
            # cap the threads to 2
            cv2.setNumThreads(2)
            flow = opt_flow.calc(prev, current, prev_flow)

        self.clipped_temp = current
        self.flow = flow


class FrameBuffer:
    """ Stores entire clip in memory, required for some operations such as track exporting. """

    def __init__(self, cptv_name, opt_flow, cache_to_disk):
        self.cache = FrameCache(cptv_name) if cache_to_disk else None
        self.opt_flow = opt_flow
        self.frames = None
        self.thermal = None
        self.filtered = None
        self.delta = None
        self.mask = None
        self.flow = None
        self.frame_number = 0
        self.prev_frame = None
        self.reset()

    def add_frame(self, thermal, filtered, mask):
        if self.cache:
            frame = Frame(thermal, filtered, mask, self.frame_number)
            frame.generate_optical_flow(self.opt_flow, self.prev_frame)
            self.cache.add_frame(frame)
            self.prev_frame = frame
            self.frame_number += 1
        else:
            self.filtered.append(filtered)
            self.mask.append(mask)

    @property
    def has_flow(self):
        return self.cache or self.flow

    def generate_optical_flow(self, opt_flow, flow_threshold=40):
        """
        Generate optical flow from thermal frames
        :param opt_flow: An optical flow algorithm
        """
        self.flow = []

        height, width = self.filtered[0].shape
        flow = np.zeros([height, width, 2], dtype=np.float32)

        current = None
        for frame in self.thermal:
            frame = np.float32(frame)
            # strong filtering helps with the optical flow.
            threshold = np.median(frame) + flow_threshold
            next = np.uint8(np.clip(frame - threshold, 0, 255))

            if current is not None:
                # for some reason openCV spins up lots of threads for this which really slows things down, so we
                # cap the threads to 2
                cv2.setNumThreads(2)
                flow = opt_flow.calc(current, next, flow)

            current = next

            # scale up the motion vectors so that we get some additional precision
            # but also make sure they fit within an int16
            scaled_flow = np.clip(flow * 256, -16000, 16000)
            self.flow.append(scaled_flow)

    def get_frame(self, frame_number):
        if self.cache:
            return self.cache.get_frame(frame_number)

        thermal = self.thermal[frame_number]
        filtered = self.filtered[frame_number]
        mask = self.mask[frame_number]
        if self.flow:
            flow = self.flow[frame_number]
            return [thermal, filtered, flow[:, :, 0], flow[:, :, 1], mask]
        else:
            return [thermal, filtered, None, None, mask]

    def close_cache(self):
        if self.cache:
            self.cache.close()

    def remove_cache(self):
        if self.cache:
            self.cache.delete()

    def reset(self):
        """
        Empties buffer
        """
        self.thermal = []
        self.filtered = []
        self.delta = []
        self.mask = []
        self.flow = []
        self.frames = []

    def __len__(self):
        return len(self.thermal)
