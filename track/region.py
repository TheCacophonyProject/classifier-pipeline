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

import ml_tools.tools as tools
from ml_tools.tools import Rectangle


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
        frame_number=0,
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
        self.frame_number = frame_number
        # if this region was cropped or not
        self.was_cropped = was_cropped
        self.is_along_border = False

    @classmethod
    def region_from_array(cls, region_bounds, frame_number=0):
        width = region_bounds[2] - region_bounds[0]
        height = region_bounds[3] - region_bounds[1]
        return cls(
            region_bounds[0], region_bounds[1], width, height, frame_number=frame_number
        )

    def calculate_mass(self, filtered, threshold):
        """ 
            calculates mass on this frame for this region 
            filtered is assumed to be cropped to the region
        """
        height, width = filtered.shape
        assert (
            width == self.width and height == self.height
        ), "calculating variance on incorrectly sized filtered"

        self.mass = tools.calculate_mass(filtered, threshold)

    def calculate_variance(self, filtered, prev_filtered):
        """ 
            calculates variance on this frame for this region 
            filtered is assumed to be cropped to the region
        """
        height, width = filtered.shape
        assert (
            width == self.width and height == self.height
        ), "calculating variance on incorrectly sized filtered"

        self.pixel_variance = tools.calculate_variance(filtered, prev_filtered)

    def set_is_along_border(self, bounds):
        self.is_along_border = (
            self.was_cropped
            or self.x == bounds.x
            or self.y == bounds.y
            or self.right == bounds.width
            or self.bottom == bounds.height
        )

    def copy(self):
        return Region(
            self.x,
            self.y,
            self.width,
            self.height,
            self.mass,
            self.pixel_variance,
            self.id,
            self.frame_number,
            self.was_cropped,
        )
