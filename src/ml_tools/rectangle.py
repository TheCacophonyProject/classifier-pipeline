import attr
import numpy as np
import math


@attr.s(eq=False)
class Rectangle:
    """Defines a rectangle by the topleft point and width / height."""

    x = attr.ib()
    y = attr.ib()
    width = attr.ib()
    height = attr.ib()

    @staticmethod
    def from_ltrb(left, top, right, bottom):
        """Construct a rectangle from left, top, right, bottom co-ords."""
        return Rectangle(left, top, width=right - left, height=bottom - top)

    def to_ltrb(self):
        """Return rectangle as left, top, right, bottom co-ords."""
        return [self.left, self.top, self.right, self.bottom]

    def to_ltwh(self):
        """Return rectangle as left, top, right, bottom co-ords."""
        return [self.left, self.top, self.width, self.height]

    def copy(self):
        return Rectangle(self.x, self.y, self.width, self.height)

    @property
    def elongation(self):
        return max(self.width, self.height) / min(self.width, self.height)

    # (self.mid_x, self.mid_y)

    @property
    def mid(self):
        return (self.mid_x, self.mid_y)

    @property
    def mid_x(self):
        return self.x + self.width / 2

    @property
    def mid_y(self):
        return self.y + self.height / 2

    @property
    def left(self):
        return self.x

    @property
    def top(self):
        return self.y

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    @left.setter
    def left(self, value):
        old_right = self.right
        self.x = value
        self.right = old_right

    @top.setter
    def top(self, value):
        old_bottom = self.bottom
        self.y = value
        self.bottom = old_bottom

    @right.setter
    def right(self, value):
        self.width = value - self.x

    @bottom.setter
    def bottom(self, value):
        self.height = value - self.y

    def overlap_area(self, other):
        """Compute the area overlap between this rectangle and another."""
        x_overlap = max(0, min(self.right, other.right) - max(self.left, other.left))
        y_overlap = max(0, min(self.bottom, other.bottom) - max(self.top, other.top))
        return x_overlap * y_overlap

    def crop(self, bounds):
        """Crops this rectangle so that it fits within given bounds"""
        self.left = min(bounds.right, max(self.left, bounds.left))
        self.top = min(bounds.bottom, max(self.top, bounds.top))
        self.right = max(bounds.left, min(self.right, bounds.right))
        self.bottom = max(bounds.top, min(self.bottom, bounds.bottom))

    def subimage(self, image):
        """Returns a subsection of the original image bounded by this rectangle
        :param image mumpy array of dims [height, width]
        """
        return image[
            self.top : self.top + self.height, self.left : self.left + self.width
        ]

    # enlarge rectangle such equal pixels are added to width and height  with respect to the crop rectangle
    def enlarge_even(self, width_enlarge, height_enlarge, crop):

        self.left -= width_enlarge
        self.right += width_enlarge
        self.top -= height_enlarge
        self.bottom += height_enlarge
        left_adjust = crop.left - self.left
        left_adjust = max(0, left_adjust)
        left_adjust = min(left_adjust, crop.width)

        right_adjust = 0
        right_adjust = self.right - crop.right
        right_adjust = max(0, right_adjust)
        right_adjust = min(right_adjust, crop.width)
        width_adjust = max(left_adjust, right_adjust)

        self.left += width_adjust
        self.right -= width_adjust

        bottom_adjust = self.bottom - crop.bottom
        bottom_adjust = max(0, bottom_adjust)
        bottom_adjust = min(bottom_adjust, crop.height)

        top_adjust = crop.top - self.top
        top_adjust = max(0, top_adjust)
        top_adjust = min(top_adjust, crop.height)

        height_adjust = max(bottom_adjust, top_adjust)
        self.top += height_adjust
        self.bottom -= height_adjust

    def enlarge(self, border, max=None):
        """Enlarges this by border amount in each dimension such that it fits
        within the boundaries of max"""
        self.left -= border
        self.right += border
        self.top -= border
        self.bottom += border
        if max:
            self.crop(max)

    def contains(self, x, y):
        """Is this point contained in the rectangle"""
        return self.left <= x and self.right >= x and self.top >= y and self.bottom <= y

    @property
    def area(self):
        return int(self.width) * self.height

    def __repr__(self):
        return "(x{0},y{1},x2{2},y2{3})".format(
            self.left, self.top, self.right, self.bottom
        )

    def __str__(self):
        return "<(x{0},y{1})-h{2}xw{3}>".format(self.x, self.y, self.height, self.width)

    def meta_dictionary(self):
        # Return object as dictionary without is_along_border,was_cropped and id for saving to json
        region_info = attr.asdict(
            self,
            filter=lambda attr, value: attr.name
            not in ["is_along_border", "was_cropped", "id", "centroid"],
        )
        # region_info["centroid"][0] = round(region_info["centroid"][0], 1)
        # region_info["centroid"][1] = round(region_info["centroid"][1], 1)
        if region_info["pixel_variance"] is not None:
            region_info["pixel_variance"] = round(region_info["pixel_variance"], 2)
        else:
            region_info["pixel_variance"] = 0
        return region_info

    # enlarge a region such that the aspect ration of final_dim will be maintained
    # when it is resized to (final_dim,final_dim) and add extra pixels so that rotation augments
    # dont get empty pixels
    def enlarge_for_rotation(self, crop_rectangle, final_dim=32, extra_needed=13):
        scale_percent = (final_dim / np.array([self.width, self.height])).min()

        extra_pixels = extra_needed / scale_percent
        height_enlarge = math.ceil(extra_pixels / 2)

        width_enlarge = math.ceil(extra_pixels / 2)

        adjusted_height = self.height + extra_pixels
        adjusted_width = self.width + extra_pixels
        if self.width > self.height:
            diff = adjusted_width - adjusted_height
            height_enlarge = math.ceil((extra_pixels + diff) / 2)
        else:
            diff = adjusted_height - adjusted_width
            width_enlarge = math.ceil((extra_pixels + diff) / 2)

        self.enlarge_even(width_enlarge, height_enlarge, crop=crop_rectangle)
