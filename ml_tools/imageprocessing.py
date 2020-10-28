import cv2
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from ml_tools.tools import eucl_distance
from track.track import TrackChannels


def movement_images(
    frames,
    regions,
    dim,
    require_movement=False,
):
    """Return 2 images describing the movement, one has dots representing
    the centre of mass, the other is a collage of all frames
    """
    channel = TrackChannels.filtered

    i = 0
    dots = np.zeros(dim)
    overlay = np.zeros(dim)

    prev = None
    prev_overlay = None
    line_colour = 60
    dot_colour = 120

    img = Image.fromarray(np.uint8(dots))

    d = ImageDraw.Draw(img)
    # draw movment lines and draw frame overlay
    center_distance = 0
    min_distance = 2
    for i, frame in enumerate(frames):
        region = regions[i]

        x = int(region.mid_x)
        y = int(region.mid_y)

        # writing dot image
        if prev is not None:
            d.line(prev + (x, y), fill=line_colour, width=1)
        prev = (x, y)

        # writing overlay image
        if require_movement and prev_overlay:
            center_distance = eucl_distance(
                prev_overlay,
                (
                    x,
                    y,
                ),
            )

        if (
            prev_overlay is None or center_distance > min_distance
        ) or not require_movement:
            frame = frame[channel]
            subimage = region.subimage(overlay)
            subimage[:, :] += np.float32(frame)
            center_distance = 0
            min_distance = pow(region.width / 2.0, 2)
            prev_overlay = (x, y)

    # then draw dots to dot image so they go over the top
    for i, frame in enumerate(frames):
        region = regions[i]
        x = int(region.mid_x)
        y = int(region.mid_y)
        d.point([(x, y)], fill=dot_colour)

    return np.array(img), overlay


def square_clip(data, frames_per_row, tile_dim, type=None):
    # lay each frame out side by side in rows
    new_frame = np.zeros((frames_per_row * tile_dim[0], frames_per_row * tile_dim[1]))

    i = 0
    success = False
    for x in range(frames_per_row):
        for y in range(frames_per_row):
            if i >= len(data):
                frame = data[-1]
            else:
                frame = data[i]
            frame, norm_success = normalize(frame)
            if not norm_success:
                continue
            success = True
            new_frame[
                x * tile_dim[0] : (x + 1) * tile_dim[0],
                y * tile_dim[1] : (y + 1) * tile_dim[1],
            ] = np.float32(frame)
            i += 1

    return new_frame, success


def normalize(data, min=None, max=None, new_max=1):
    if max is None:
        max = np.amax(data)
    if min is None:
        min = np.amin(data)
    if max == min:
        if max == 0:
            return np.zeros((data.shape)), False
        return data / max, False
    data -= min
    data = data / (max - min) * new_max
    return data, True


def save_image_channels(data, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    r = Image.fromarray(np.uint8(data[:, :, 0] * 255))
    g = Image.fromarray(np.uint8(data[:, :, 1] * 255))
    b = Image.fromarray(np.uint8(data[:, :, 2] * 255))
    concat = np.concatenate((r, g, b), axis=1)
    img = Image.fromarray(np.uint8(concat))
    img.save(filename + ".png")


def resize_cv(image, dim, interpolation=cv2.INTER_LINEAR, extra_h=0, extra_v=0):
    return cv2.resize(
        image,
        dsize=(dim[0] + extra_h, dim[1] + extra_v),
        interpolation=interpolation,
    )


def detect_objects(image, otsus=True, threshold=0, kernel=(5, 5)):
    image = np.uint8(image)
    # filtered = cv2.fastNlMeansDenoising(filtered, None)
    image = cv2.GaussianBlur(image, kernel, 0)
    flags = cv2.THRESH_BINARY
    if otsus:
        flags += cv2.THRESH_OTSU
    _, image = cv2.threshold(image, threshold, 255, flags)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(image)
    return components, small_mask, stats
