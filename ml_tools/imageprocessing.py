import cv2
import numpy as np

from pathlib import Path
from PIL import Image
from scipy import ndimage
from PIL import Image


def resize_and_pad(
    frame,
    resize_dim,
    new_dim,
    region,
    crop_region,
    keep_edge=False,
    pad=None,
    interpolation=cv2.INTER_LINEAR,
    extra_h=0,
    extra_v=0,
):
    if pad is None:
        pad = np.min(frame)
    else:
        pad = 0
    resized = np.full(new_dim, pad, dtype=frame.dtype)
    offset_x = 0
    offset_y = 0
    frame_resized = resize_cv(frame, resize_dim, interpolation=interpolation)
    frame_height, frame_width = frame_resized.shape[:2]
    offset_x = (new_dim[1] - frame_width) // 2
    offset_y = (new_dim[0] - frame_height) // 2
    if keep_edge:
        if region.left == crop_region.left:
            offset_x = 0

        elif region.right == crop_region.right:
            offset_x = new_dim[1] - frame_width

        if region.top == crop_region.top:
            offset_y = 0

        elif region.bottom == crop_region.bottom:
            offset_y = new_dim[0] - frame_height
    if len(resized.shape) == 3:
        resized[
            offset_y : offset_y + frame_height, offset_x : offset_x + frame_width, :
        ] = frame_resized
    else:
        resized[
            offset_y : offset_y + frame_height,
            offset_x : offset_x + frame_width,
        ] = frame_resized

    return resized


def rotate(image, degrees, mode="nearest", order=1):
    return ndimage.rotate(image, degrees, reshape=False, mode=mode, order=order)


def resize_cv(image, dim, interpolation=cv2.INTER_LINEAR, extra_h=0, extra_v=0):

    return cv2.resize(
        np.float32(image),
        dsize=(dim[0] + extra_h, dim[1] + extra_v),
        interpolation=interpolation,
    )


def square_clip(data, frames_per_row, tile_dim):
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

            frame, stats = normalize(frame)
            if not stats[0]:
                continue
            success = True
            new_frame[
                x * tile_dim[0] : (x + 1) * tile_dim[0],
                y * tile_dim[1] : (y + 1) * tile_dim[1],
            ] = np.float32(frame)
            i += 1

    return new_frame, success


def square_clip_flow(data_flow, frames_per_row, tile_dim, use_rgb=False):

    if use_rgb:
        new_frame = np.zeros(
            (frames_per_row * tile_dim[0], frames_per_row * tile_dim[1], 3)
        )
    else:
        new_frame = np.zeros(
            (frames_per_row * tile_dim[0], frames_per_row * tile_dim[1])
        )

    i = 0
    success = False
    hsv = np.zeros((tile_dim[0], tile_dim[1], 3), dtype=np.float32)
    hsv[..., 1] = 255
    for x in range(frames_per_row):
        for y in range(frames_per_row):
            if i >= len(data_flow):
                flow = data_flow[-1]
            else:
                flow = data_flow[i]
            flow_h = flow[:, :, 0]
            flow_v = flow[:, :, 1]

            mag, ang = cv2.cartToPolar(flow_h, flow_v)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            if use_rgb:
                flow_magnitude = rgb
            else:
                flow_magnitude = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            frame, norm_success = normalize(flow_magnitude)

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
    """
    Normalize an array so that the values range from 0 -> new_max
    Returns normalized array, stats tuple (Success, min used, max used)
    """
    if data.size == 0:
        return np.zeros((data.shape)), (False, None, None)
    if max is None:
        max = np.amax(data)
    if min is None:
        min = np.amin(data)
    if max == min:
        if max == 0:
            return np.zeros((data.shape)), (False, max, min)
        data = data / max
        return data, (True, max, min)
    data = new_max * (data - min) / (max - min)
    return data, (True, max, min)


def save_image_channels(data, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    r = Image.fromarray(np.uint8(data[:, :, 0] * 255))
    g = Image.fromarray(np.uint8(data[:, :, 1] * 255))
    b = Image.fromarray(np.uint8(data[:, :, 2] * 255))
    concat = np.concatenate((r, g, b), axis=1)
    img = Image.fromarray(np.uint8(concat))
    img.save(filename + ".png")


index = 0


def theshold_saliency(image, otsus=False, threshold=100, kernel=(15, 15)):
    image = np.uint8(image)
    image = cv2.fastNlMeansDenoising(np.uint8(image), None)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # image = cv2.GaussianBlur(image, kernel, 0)
    flags = cv2.THRESH_BINARY
    if otsus:
        flags += cv2.THRESH_OTSU

    _, image = cv2.threshold(image, threshold, 255, flags)

    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(image)
    #
    # import matplotlib.pyplot as plt
    #
    # imgplot = plt.imshow(image)
    # plt.show()

    return components, small_mask, stats


def detect_objects_ir(image, otsus=True, threshold=0, kernel=(15, 15)):

    image = np.uint8(image)
    image = cv2.Canny(image, 100, 200)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(image)
    return components, small_mask, stats


def detect_objects(image, otsus=True, threshold=0, kernel=(15, 15)):
    global index
    index += 1
    image = np.uint8(image)
    image = cv2.GaussianBlur(image, kernel, 0)
    flags = cv2.THRESH_BINARY
    if otsus:
        flags += cv2.THRESH_OTSU
    _, image = cv2.threshold(image, threshold, 255, flags)
    image = cv2.dilate(image, kernel, iterations=1)

    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # import matplotlib.pyplot as plt
    #
    # imgplot = plt.imshow(image)
    # plt.savefig(f"0 below{kernel[0]}-dilate{index}.png")
    # plt.clf()
    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(image)
    return components, small_mask, stats


def clear_frame(frame):
    filtered = frame.filtered
    thermal = frame.thermal
    if len(filtered) == 0 or len(thermal) == 0:
        return False
    thermal_deviation = np.amax(thermal) != np.amin(thermal)
    filtered_deviation = np.amax(filtered) != np.amin(filtered)
    if not thermal_deviation or not filtered_deviation:
        return False

    return True
