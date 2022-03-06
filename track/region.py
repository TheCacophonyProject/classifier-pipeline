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
import attr

import numpy as np


@attr.s(eq=False)
class Region(Rectangle):
    """Region is a rectangle extended to support mass."""

    mass = attr.ib(default=0)
    # how much pixels in this region have changed since last frame
    frame_number = attr.ib(default=0)
    pixel_variance = attr.ib(default=0)
    id = attr.ib(default=0)

    # if this region was cropped or not
    was_cropped = attr.ib(default=False)
    blank = attr.ib(default=False)
    is_along_border = attr.ib(default=False)

    @staticmethod
    def from_ltwh(left, top, width, height):
        """Construct a rectangle from left, top, right, bottom co-ords."""
        return Region(left, top, width=width, height=height)

    @classmethod
    def region_from_array(cls, region_bounds, frame_number=0):
        width = region_bounds[2] - region_bounds[0]
        height = region_bounds[3] - region_bounds[1]
        return cls(
            region_bounds[0],
            region_bounds[1],
            width,
            height,
            frame_number=np.uint16(frame_number),
        )

    @classmethod
    def region_from_json(cls, region_json):
        return cls(
            region_json["x"],
            region_json["y"],
            region_json["width"],
            region_json["height"],
            frame_number=region_json["frame_number"],
            mass=region_json.get("mass", 0),
            blank=region_json.get("blank", False),
            pixel_variance=region_json.get("pixel_variance", 0),
        )

    @staticmethod
    def from_ltrb(left, top, right, bottom):
        """Construct a rectangle from left, top, right, bottom co-ords."""
        return Region(left, top, width=right - left, height=bottom - top)

    def has_moved(self, region):
        """Determines if the region has shifted horizontally or veritcally
        Not just increased in width/height
        """
        return (self.x != region.x and self.right != region.right) or (
            self.y != region.y and self.bottom != region.bottom
        )

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
            self.frame_number,
            self.pixel_variance,
            self.id,
            self.was_cropped,
            self.blank,
            self.is_along_border,
        )

    def average_distance(self, other):
        """Calculates the distance between 2 regions by using the distance between
        (top, left), mid points and (bottom,right) of each region
        """
        distances = []

        expected_x = int(other.x)
        expected_y = int(other.y)
        distance = tools.eucl_distance((expected_x, expected_y), (self.x, self.y))
        distances.append(distance)
        # print("distance between", (expected_x, expected_y), (self.x, self.y), distance)

        expected_x = int(other.mid_x)
        expected_y = int(other.mid_y)
        distance = tools.eucl_distance(
            (expected_x, expected_y), (self.mid_x, self.mid_y)
        )
        distances.append(distance)
        # print(
        #     "distance between",
        #     (expected_x, expected_y),
        #     (self.mid_x, self.mid_y),
        #     distance,
        # )

        distance = tools.eucl_distance(
            (
                other.right,
                other.bottom,
            ),
            (self.right, self.bottom),
        )
        expected_x = int(other.x)
        expected_y = int(other.y)
        distance = tools.eucl_distance((expected_x, expected_y), (self.x, self.y))
        distances.append(distance)
        # print(
        #     "right bottom distance",
        #     (
        #         other.right,
        #         other.bottom,
        #     ),
        #     (self.right, self.bottom),
        #     distance,
        # )
        return distances
        # total_distance += distance
        #
        # total_distance /= 3.0
        # return total_distance
