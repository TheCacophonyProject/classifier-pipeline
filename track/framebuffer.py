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
        # all arrays should be np.float32
        self.thermal = None
        self.filtered = None
        self.delta = None
        self.mask = None
        self.flow = None
        self.reset()

    @property
    def has_flow(self):
        return self.flow is not None and len(self.flow) != 0

    def get_previous_filtered(self, region=None, frame_number=None):
        if frame_number:
            previous = frame_number - 1
            if previous < 0:
                return None
        else:
            previous = -2

        if region:
            return region.subimage(self.filtered[previous])
        else:
            return self.filtered[previous]

    def get_frame_channels(self, region, frame_number):
        """
        Gets frame channels for track at given frame number.  If frame number outside of track's lifespan an exception
        is thrown.  Requires the frame_buffer to be filled.
        :param track: the track to get frames for.
        :param frame_number: the frame number where 0 is the first frame of the track.
        :return: numpy array of size [channels, height, width] where channels are thermal, filtered, u, v, mask
        """

        if frame_number < 0 or frame_number >= len(self.thermal):
            raise ValueError(
                "Frame {} is out of bounds for track with {} frames".format(
                    frame_number, len(self.thermal)
                )
            )

        thermal = region.subimage(self.thermal[frame_number])
        filtered = region.subimage(self.filtered[frame_number])
        flow = region.subimage(self.flow[frame_number])
        mask = region.subimage(self.mask[frame_number])

        # make sure only our pixels are included in the mask.
        mask[mask != region.id] = 0
        mask[mask > 0] = 1

        # stack together into a numpy array.
        # by using int16 we loose a little precision on the filtered frames, but not much (only 1 bit)
        frame = np.int16(
            np.stack((thermal, filtered, flow[:, :, 0], flow[:, :, 1], mask), axis=0)
        )

        return frame

    def add_frame(self, filtered, mask):
        self.filtered.append(filtered)
        self.mask.append(mask)

    def generate_optical_flow(self, opt_flow, flow_threshold=40):
        """
        Generate optical flow from thermal frames
        :param opt_flow: An optical flow algorithm
        """

        self.flow = []

        height, width = self.thermal[0].shape
        flow = np.zeros([height, width, 2], dtype=np.float32)
        # for some reason openCV spins up lots of threads for this which really slows things down, so we
        # cap the threads to 2
        cv2.setNumThreads(2)

        current = None
        for frame in self.thermal:
            # strong filtering helps with the optical flow.
            threshold = np.median(frame) + flow_threshold
            next = np.uint8(np.clip(frame - threshold, 0, 255))

            if current is not None:
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
