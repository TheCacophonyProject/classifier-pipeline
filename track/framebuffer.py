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
import pickle
import h5py
from multiprocessing import Lock
import numpy as np

HDF5_LOCK = Lock()


class HDF5Manager:
    """ Class to handle locking of HDF5 files. """

    def __init__(self, db, mode="r"):
        self.mode = mode
        self.f = None
        self.db = db

    def __enter__(self):
        # note: we might not have to lock when in read only mode?
        # this could improve performance
        HDF5_LOCK.acquire()
        self.f = h5py.File(self.db, self.mode)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()
        HDF5_LOCK.release()


class Frame:
    def __init__(self, thermal, filtered, mask):
        self.thermal = thermal
        self.filtered = filtered
        self.mask = mask
        self.flow = None
        self.flow_temp = None

    def generate_optical_flow(self, opt_flow, last_frame, flow_threshold=40):
        """
        Generate optical flow from thermal frames
        :param opt_flow: An optical flow algorithm
        """

        height, width = self.thermal.shape
        flow = np.zeros([height, width, 2], dtype=np.float32)

        prev = last_frame.flow_temp if last_frame else None
        prev_flow = last_frame.flow if last_frame else None

        frame = self.thermal
        threshold = np.median(frame) + flow_threshold
        current = np.uint8(np.clip(frame - threshold, 0, 255))

        if prev is not None:
            # for some reason openCV spins up lots of threads for this which really slows things down, so we
            # cap the threads to 2
            cv2.setNumThreads(2)
            flow = opt_flow.calc(prev, current, prev_flow)

        self.flow_temp = current
        self.flow = flow


class FrameBuffer:
    """ Stores entire clip in memory, required for some operations such as track exporting. """

    def __init__(self):
        self.opt_flow = cv2.createOptFlow_DualTVL1()
        self.opt_flow.setUseInitialFlow(True)
        self.frames = None
        self.thermal = None
        self.filtered = None
        self.delta = None
        self.mask = None
        self.flow = None

        f = h5py.File("save.h5py", "w")
        f.create_group("frames")
        f.close()
        self.number = 0
        self.prev_frame = None
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

    def add_frame(self, thermal, filtered, mask):
        frame = Frame(thermal, filtered, mask)
        frame.generate_optical_flow(self.opt_flow, self.prev_frame)
        with HDF5Manager("save.h5py", "a") as f:
            frames = f["frames"]
            channels = 5
            height, width = frame.thermal.shape

            # using a chunk size of 1 for channels has the advantage that we can quickly load just one channel
            chunks = (1, height, width)

            dims = (5, height, width)
            frame_node = frames.create_dataset(
                str(self.number), dims, chunks=chunks, dtype=np.float16
            )

            frame_val = (
                np.float16(frame.thermal),
                np.float16(frame.filtered),
                np.float16(frame.mask),
                np.float16(frame.flow[:, :, 0]),
                np.float16(frame.flow[:, :, 1]),
            )
            frame_node[:, :, :] = frame_val

        self.prev_frame = frame
        self.number += 1

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
