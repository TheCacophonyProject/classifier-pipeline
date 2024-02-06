"""
Helper functions for classification of the tracks extracted from CPTV videos
"""

import attr
import os.path
import numpy as np
import random
import pickle
import math
import matplotlib.pyplot as plt
import logging
from sklearn import metrics
import json
import dateutil
import binascii
import datetime
import glob
import cv2
import enum
import timezonefinder
from matplotlib.colors import LinearSegmentedColormap
import subprocess
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path


EPISON = 1e-5

LOCAL_RESOURCES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
GLOBAL_RESOURCES = "/usr/lib/classifier-pipeline/resources"


class FrameTypes(enum.Enum):
    """Types of frames"""

    thermal_tiled = 0
    filtered_tiled = 1
    flow_tiled = 2
    overlay = 3
    flow_rgb = 4
    thermal = 5
    filtered = 6

    @staticmethod
    def is_valid(name):
        return name in FrameTypes.__members__.keys()


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
    def mid(self):
        return (self.mid_x, self.mid_y)

    @property
    def mid_x(self):
        return self.x + self.width / 2

    def calculate_mass(self, filtered, threshold):
        """
        calculates mass on this frame for this region
        filtered is assumed to be cropped to the region
        """
        height, width = filtered.shape
        assert (
            width == self.width and height == self.height
        ), "calculating variance on incorrectly sized filtered"

        self.mass = calculate_mass(filtered, threshold)

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

    def enlarge(self, border, max=None):
        """Enlarges this by border amount in each dimension such that it fits
        within the boundaries of max"""
        self.left -= border
        self.right += border
        self.top -= border
        self.bottom += border
        if max:
            self.crop(max)

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


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return list(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, Rectangle):
            return obj.meta_dictionary()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def load_colourmap(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def convert_heat_to_img(frame, colormap=None, temp_min=None, temp_max=None):
    """
    Converts a frame in float32 format to a PIL image in in uint8 format.
    :param frame: the numpy frame contining heat values to convert
    :param colormap: an optional colormap to use, if none is provided then tracker.colormap is used.
    :return: a pillow Image containing a colorised heatmap
    """
    # normalise
    if colormap is None:
        colormap = _load_colourmap(None)
    if temp_min is None:
        temp_min = np.amin(frame)
    if temp_max is None:
        temp_max = np.amax(frame)
    frame = np.float32(frame)
    frame = (frame - temp_min) / (temp_max - temp_min)
    colorized = np.uint8(255.0 * colormap(frame))
    img = Image.fromarray(colorized[:, :, :3])  # ignore alpha
    return img


def load_clip_metadata(filename):
    """
    Loads a metadata file for a clip.
    :param filename: full path and filename to meta file
    :return: returns the stats file
    """
    with open(filename, "r") as t:
        # add in some metadata stats
        meta = json.load(t)
    if meta.get("recordingDateTime"):
        meta["recordingDateTime"] = dateutil.parser.parse(meta["recordingDateTime"])
    if meta.get("tracks") is None and meta.get("Tracks"):
        meta["tracks"] = meta["Tracks"]
    return meta


def clear_session():
    import tensorflow as tf

    tf.keras.backend.clear_session()


def calculate_mass(filtered, threshold):
    """Calculates mass of filtered frame with threshold applied"""
    if filtered.size == 0:
        return 0
    _, mass = blur_and_return_as_mask(filtered, threshold=threshold)
    return np.uint16(mass)


def calculate_variance(filtered, prev_filtered):
    """Calculates variance of filtered frame with previous frame"""
    if filtered.size == 0:
        return 0
    if prev_filtered is None:
        return
    delta_frame = np.abs(filtered - prev_filtered)
    return np.var(delta_frame)


def blur_and_return_as_mask(frame, threshold):
    """
    Creates a binary mask out of an image by applying a threshold.
    Any pixels more than the threshold are set 1, all others are set to 0.
    A blur is also applied as a filtering step
    """
    thresh = cv2.GaussianBlur(frame, (5, 5), 0)
    thresh[thresh - threshold < 0] = 0
    values = thresh[thresh > 0]
    mass = len(values)
    values = 1
    return thresh, mass


def get_optical_flow_function(high_quality=False):
    opt_flow = cv2.optflow.createOptFlow_DualTVL1()
    opt_flow.setUseInitialFlow(True)
    if not high_quality:
        # see https://stackoverflow.com/questions/19309567/speeding-up-optical-flow-createoptflow-dualtvl1
        opt_flow.setTau(1 / 4)
        opt_flow.setScalesNumber(3)
        opt_flow.setWarpingsNumber(3)
        opt_flow.setScaleStep(0.5)
    return opt_flow


def frame_to_jpg(
    frame, filename, colourmap_file=None, f_min=None, f_max=None, img_fmt="PNG"
):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    colourmap = _load_colourmap(colourmap_file)
    if f_min is None:
        f_min = np.amin(frame)
    if f_max is None:
        f_max = np.amax(frame)
    img = convert_heat_to_img(frame, colourmap, f_min, f_max)
    img.save(filename, img_fmt)


def _load_colourmap(colourmap_path):
    if colourmap_path is None or not os.path.exists(colourmap_path):
        colourmap_path = resource_path("colourmap.dat")
    return load_colourmap(colourmap_path)


def resource_path(name):
    for base in [LOCAL_RESOURCES, GLOBAL_RESOURCES]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise OSError("unable to locate {} resource".format(name))


def add_heat_number(img, frame, scale):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(resource_path("Ubuntu-R.ttf"), 8)
    for y, row in enumerate(frame):
        if y % 4 == 0:
            min_v = np.amin(row)
            min_i = np.where(row == min_v)[0][0]
            max_v = np.amax(row)
            max_i = np.where(row == max_v)[0][0]
            # print("min is", min_v, "max is", max_v)
            draw.text((min_i * scale, y * scale), str(int(min_v)), (0, 0, 0), font=font)
            draw.text((max_i * scale, y * scale), str(int(max_v)), (0, 0, 0), font=font)
            # print("drawing the max at row", y, max, max_i * scale, y * scale)


gzip_compression = {"compression": "gzip"}


def eucl_distance_sq(first, second):
    first_sq = first[0] - second[0]
    first_sq = first_sq * first_sq
    second_sq = first[1] - second[1]
    second_sq = second_sq * second_sq

    return first_sq + second_sq


def get_clipped_flow(flow):
    return np.clip(flow * 256, -16000, 16000)


def saveclassify_image(data, filename):
    # saves image channels side by side, expected data to be values in the range of 0->1
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    r = Image.fromarray(np.uint8(data[:, :, 0]))
    g = Image.fromarray(np.uint8(data[:, :, 1]))
    b = Image.fromarray(np.uint8(data[:, :, 2]))
    concat = np.concatenate((r, g, b), axis=1)  # horizontally
    img = Image.fromarray(np.uint8(concat))
    img.save(filename + ".png")


def get_timezone_str(lat, lng):
    tf = timezonefinder.TimezoneFinder()
    timezone_str = tf.certain_timezone_at(lat=lat, lng=lng)

    if timezone_str is None:
        timezone_str = "Pacific/Auckland"
    return timezone_str


def saveclassify_rgb(data, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    r = Image.fromarray(np.uint8(data))
    r.save(filename + ".png")


def purge(dir, pattern):
    for f in glob.glob(os.path.join(dir, pattern)):
        os.remove(f)
