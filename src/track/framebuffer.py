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

from ml_tools.framecache import FrameCache
from ml_tools.frame import Frame
from ml_tools.tools import get_optical_flow_function

import time
import logging


class FrameBuffer:
    """Stores entire clip in memory, required for some operations such as track exporting."""

    def __init__(
        self,
        cptv_name,
        high_quality_flow,
        cache_to_disk,
        calc_flow,
        keep_frames,
        max_frames=None,
    ):
        self.cache = FrameCache(cptv_name) if cache_to_disk else None
        self.opt_flow = None
        self.high_quality_flow = high_quality_flow
        self.frames = None
        self.prev_frame = None
        self.calc_flow = calc_flow
        self.max_frames = max_frames
        self.keep_frames = True if max_frames and max_frames > 0 else keep_frames
        self.current_frame_i = 0
        self.current_frame = None
        if calc_flow:
            self.set_optical_flow()
        self.reset()

    def set_optical_flow(self):
        if self.opt_flow is None:
            self.opt_flow = get_optical_flow_function(self.high_quality_flow)

    def add_frame(self, thermal, filtered, mask, frame_number, ffc_affected=False):
        self.prev_frame = self.current_frame
        frame = Frame(thermal, filtered, mask, frame_number, ffc_affected=ffc_affected)
        self.current_frame = frame

        if self.opt_flow:
            frame.generate_optical_flow(self.opt_flow, self.prev_frame)
        if self.keep_frames:
            if self.cache:
                self.cache.add_frame(frame)
            else:
                if self.max_frames and len(self.frames) == self.max_frames:
                    del self.frames[0]
                self.frames.append(frame)
        return frame

    @property
    def has_flow(self):
        return self.cache or self.opt_flow

    def get_frame(self, frame_number):
        if self.prev_frame and self.prev_frame.frame_number == frame_number:
            return self.prev_frame
        elif self.current_frame and self.current_frame.frame_number == frame_number:
            return self.current_frame
        elif self.cache:
            return self.cache.get_frame(frame_number)
        if self.current_frame is None:
            return None
        if frame_number > self.current_frame.frame_number:
            return None
        # this supports max frames etc
        frame_ago = self.current_frame.frame_number - frame_number
        frame = self.get_frame_ago(frame_ago)
        assert frame == None or frame.frame_number == frame_number
        return frame

    def close_cache(self):
        if self.cache:
            self.cache.close()

    def remove_cache(self):
        if self.cache:
            self.cache.delete()

    def get_frame_ago(self, n=1):
        if n == 0:
            return self.current_frame
        elif n == 1:
            return self.get_last_frame()
        if len(self.frames) > n:
            return self.frames[-(n + 1)]
        return None

    def get_last_frame(self):
        if self.prev_frame:
            return self.prev_frame
        elif len(self.frames) > 0:
            return self.frames[-1]
        return None

    def get_last_x(self, x=25):
        if self.cache and self.prev_frame:
            return self.prev_frame
        elif len(self.frames) > 0:
            return self.frames[-x:]
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
        frame = self.get_frame(self.current_frame_i)
        if frame is None:
            raise StopIteration

        self.current_frame_i += 1
        return frame
