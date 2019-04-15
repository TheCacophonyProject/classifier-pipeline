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


class FrameBuffer:
    """ Stores entire clip in memory, required for some operations such as track exporting. """

    def __init__(self):
        self.thermal = None
        self.filtered = None
        self.delta = None
        self.mask = None
        self.flow = None
        self.reset()

    @property
    def has_flow(self):
        return self.flow is not None and len(self.flow) != 0

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

    def reset(self):
        """
        Empties buffer
        """
        self.thermal = []
        self.filtered = []
        self.delta = []
        self.mask = []
        self.flow = []

    def __len__(self):
        return len(self.thermal)
