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
import logging
import numpy as np


@attr.s(eq=False, slots=True)
class Region(Rectangle):
    """Region is a rectangle extended to support mass."""

    centroid = attr.ib()
    mass = attr.ib(default=0)
    # how much pixels in this region have changed since last frame
    frame_number = attr.ib(default=0)
    pixel_variance = attr.ib(default=0)
    id = attr.ib(default=0)

    # if this region was cropped or not
    was_cropped = attr.ib(default=False)
    blank = attr.ib(default=False)
    is_along_border = attr.ib(default=False)
    in_trap = attr.ib(default=False)

    def rescale(self, factor):
        self.x = int(self.x * factor)
        self.y = int(self.y * factor)

        self.width = int(self.width * factor)
        self.height = int(self.height * factor)
        self.mass = self.mass * (factor**2)

    @staticmethod
    def from_ltwh(left, top, width, height):
        """Construct a rectangle from left, top, right, bottom co-ords."""
        return Region(left, top, width=width, height=height, centroid=None)

    def to_array(self):
        """Return rectangle as left, top, right, bottom co-ords."""
        return np.uint16(
            [
                self.left,
                self.top,
                self.right,
                self.bottom,
                self.frame_number,
                self.mass,
                1 if self.blank else 0,
            ]
        )

    @classmethod
    def region_from_array(cls, region_bounds):
        width = int(region_bounds[2]) - region_bounds[0]
        height = int(region_bounds[3]) - region_bounds[1]
        height = np.uint8(max(height, 0))
        width = np.uint8(max(width, 0))
        frame_number = None
        if len(region_bounds) > 4:
            frame_number = region_bounds[4]
        mass = 0
        if len(region_bounds) > 5:
            mass = region_bounds[5]
        blank = False
        if len(region_bounds) > 6:
            blank = region_bounds[6] == 1
        centroid = [
            int(region_bounds[0] + width / 2),
            int(region_bounds[1] + height / 2),
        ]
        return cls(
            region_bounds[0],
            region_bounds[1],
            width,
            height,
            frame_number=np.uint16(frame_number) if frame_number is not None else None,
            mass=mass,
            blank=blank,
            centroid=centroid,
        )

    @classmethod
    def region_from_json(cls, region_json):
        frame = region_json.get("frame_number")
        if frame is None:
            frame = region_json.get("frameNumber")
        if frame is None:
            frame = region_json.get("order")
        if "centroid" in region_json:
            centroid = region_json["centroid"]
        else:
            centroid = [
                int(region_json["x"] + region_json["width"] / 2),
                int(region_json["y"] + region_json["height"] / 2),
            ]
        return cls(
            region_json["x"],
            region_json["y"],
            region_json["width"],
            region_json["height"],
            frame_number=frame,
            mass=region_json.get("mass", 0),
            blank=region_json.get("blank", False),
            pixel_variance=region_json.get("pixel_variance", 0),
            centroid=centroid,
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

    def set_is_along_border(self, bounds, edge=0):
        self.is_along_border = (
            self.was_cropped
            or self.x <= bounds.x + edge
            or self.y <= bounds.y + edge
            or self.right >= bounds.width - edge
            or self.bottom >= bounds.height - edge
        )

    def copy(self):
        return Region(
            self.x,
            self.y,
            self.width,
            self.height,
            self.centroid,
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
        distance = tools.eucl_distance_sq((expected_x, expected_y), (self.x, self.y))
        distances.append(distance)

        expected_x = int(other.mid_x)
        expected_y = int(other.mid_y)
        distance = tools.eucl_distance_sq(
            (expected_x, expected_y), (self.mid_x, self.mid_y)
        )
        distances.append(distance)

        distance = tools.eucl_distance_sq(
            (
                other.right,
                other.bottom,
            ),
            (self.right, self.bottom),
        )
        # expected_x = int(other.right)
        # expected_y = int(other.bottom)
        # distance = tools.eucl_distance_sq((expected_x, expected_y), (self.x, self.y))
        distances.append(distance)

        return distances

    def on_height_edge(self, crop_region):
        if self.top == crop_region.top or self.bottom == crop_region.bottom:
            return True
        return False

    def on_width_edge(self, crop_region):
        if self.left == crop_region.left or self.right == crop_region.right:
            return True
        return False
