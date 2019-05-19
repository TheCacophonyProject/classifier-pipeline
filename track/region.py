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

from ml_tools.tools import Rectangle
import cv2
import numpy as np


class Region(Rectangle):
    """ Region is a rectangle extended to support mass. """

    def __init__(
        self,
        topleft_x,
        topleft_y,
        width,
        height,
        mass=0,
        pixel_variance=0,
        id=0,
        frame_index=0,
        was_cropped=False,
    ):
        super().__init__(topleft_x, topleft_y, width, height)
        # number of active pixels in region
        self.mass = mass
        # how much pixels in this region have changed since last frame
        self.pixel_variance = pixel_variance
        # an identifier for this region
        self.id = id
        # frame index from clip
        self.frame_index = frame_index
        # if this region was cropped or not
        self.was_cropped = was_cropped

    def calculate_mass(self, filtered, threshold):
        thresh = blur_and_return_as_mask(filtered, threshold=threshold)
        self.mass = np.sum(thresh)

    def calculate_variance(self, filtered, prev_filtered):
        # print("filtered {} prev_filtered {}".format(filtered, prev_filtered))
        if prev_filtered is None:
            return
        delta_frame = np.abs(np.float32(filtered) - np.float32(prev_filtered))
        self.pixel_variance = np.var(delta_frame)

    def copy(self):
        return Region(
            self.x,
            self.y,
            self.width,
            self.height,
            self.mass,
            self.pixel_variance,
            self.id,
            self.frame_index,
            self.was_cropped,
        )


def blur_and_return_as_mask(frame, threshold):
    """
    Creates a binary mask out of an image by applying a threshold.
    Any pixels more than the threshold are set 1, all others are set to 0.
    A blur is also applied as a filtering step
    """
    thresh = cv2.GaussianBlur(np.float32(frame), (5, 5), 0) - threshold
    thresh[thresh < 0] = 0
    thresh[thresh > 0] = 1
    return thresh


def region_from_json(region_bounds, frame_number):
    width = region_bounds[2] - region_bounds[0]
    height = region_bounds[3] - region_bounds[1]
    bounds = Region(
        region_bounds[0], region_bounds[1], width, height, frame_index=frame_number
    )
    return bounds
