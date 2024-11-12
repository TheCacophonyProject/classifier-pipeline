"""
Helper functions for classification of the tracks extracted from CPTV videos
"""

import os.path
import numpy as np
import pickle
import json
import datetime
import glob
import cv2
import enum
import timezonefinder
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from ml_tools.rectangle import Rectangle
from dateutil import parser
from enum import Enum

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
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, Enum):
            return str(obj.name)
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
        meta["recordingDateTime"] = parser.parse(meta["recordingDateTime"])
    if meta.get("tracks") is None and meta.get("Tracks"):
        meta["tracks"] = meta["Tracks"]
    return meta


def clear_session():
    import tensorflow as tf

    tf.keras.backend.clear_session()


def calculate_variance(filtered, prev_filtered):
    """Calculates variance of filtered frame with previous frame"""
    if filtered.size == 0:
        return 0
    if prev_filtered is None:
        return
    delta_frame = np.abs(filtered - prev_filtered)
    return np.var(delta_frame)


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
    _, _, channels = data.shape

    if channels == 1:
        g = r
    else:
        g = Image.fromarray(np.uint8(data[:, :, 1]))

    if channels == 2:
        b = r
    else:
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
