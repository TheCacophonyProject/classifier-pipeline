import cv2
import numpy as np

from pathlib import Path
from PIL import Image
from scipy import ndimage
from PIL import Image
import logging


def adapt_hist(image):
    from skimage import exposure

    return exposure.equalize_adapthist(
        image, kernel_size=(image.shape[0] // 2, image.shape[1] // 2), clip_limit=0.008
    )


def resize_and_pad(
    frame,
    new_dim,
    region,
    crop_region,
    keep_edge=False,
    pad=None,
    interpolation=cv2.INTER_LINEAR,
    extra_h=0,
    extra_v=0,
    edge_offset=(0, 0, 0, 0),
    original_region=None,
):
    scale_percent = (new_dim[:2] / np.array(frame.shape[:2])).min()
    width = round(frame.shape[1] * scale_percent)
    height = round(frame.shape[0] * scale_percent)
    width = max(width, 1)
    height = max(height, 1)

    width = min(width, new_dim[1])
    height = min(height, new_dim[0])

    if len(frame.shape) == 3:
        resize_dim = (width, height, frame.shape[2])
    else:
        resize_dim = (width, height)
    if pad is None:
        pad = np.min(frame)
    if original_region is None:
        original_region = region
    resized = np.full(new_dim, pad, dtype=frame.dtype)
    offset_x = 0
    offset_y = 0
    frame_resized = resize_cv(frame, resize_dim, interpolation=interpolation)
    frame_height, frame_width = frame_resized.shape[:2]
    offset_x = (new_dim[1] - frame_width) // 2
    offset_y = (new_dim[0] - frame_height) // 2
    if keep_edge and crop_region is not None:
        if original_region.left <= crop_region.left:
            offset_x = min(edge_offset[0], new_dim[1] - frame_width)
        elif original_region.right >= crop_region.right:
            offset_x = (new_dim[1] - edge_offset[2]) - frame_width
            offset_x = max(offset_x, 0)
        if original_region.top <= crop_region.top:
            offset_y = min(edge_offset[1], new_dim[0] - frame_height)

        elif original_region.bottom >= crop_region.bottom:
            offset_y = new_dim[0] - frame_height - edge_offset[3]
            offset_y = max(offset_y, 0)

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


def square_clip(data, frames_per_row, tile_dim, frame_samples=None, pad_with=None):
    # lay each frame out side by side in rows
    n_tiles = frames_per_row * frames_per_row

    if frame_samples is not None:
        idx = np.asarray(frame_samples[:n_tiles])
        frames = np.float32(np.asarray(data)[idx])
    else:
        frames = np.array(data)
    if len(frames) < n_tiles:
        if pad_with is None:
            pad_with = 0
            logging.warning(
                "Since there are less than %s frames padding with default of 0 since pad_with was None",
                n_tiles,
            )
        pad = np.full(
            (n_tiles - len(frames), tile_dim[0], tile_dim[1]),
            pad_with,
            dtype=frames.dtype,
        )
        frames = np.concatenate([frames, pad], axis=0)
    grid = frames.reshape(frames_per_row, frames_per_row, tile_dim[0], tile_dim[1])
    new_frame = grid.transpose(0, 2, 1, 3).reshape(
        frames_per_row * tile_dim[0], frames_per_row * tile_dim[1]
    )
    return new_frame


def normalize(data, min=None, max=None, new_max=1):
    """
    Normalize an array so that the values range from 0 -> new_max
    Returns normalized array, stats tuple (Success, min used, max used)
    """
    if data.size == 0:
        return None, (False, None, None)
    if max is None:
        max = np.amax(data)
    if min is None:
        min = np.amin(data)
    if max == min:
        if max == 0:
            return None, (False, max, min)
        data = data / max
        return data, (True, max, min)

    data = new_max * (np.float32(data) - min) / (max - min)
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


def detect_objects_ir(image, otsus=False, threshold=100, kernel=(15, 15)):
    image = np.uint8(image)
    # image = cv2.fastNlMeansDenoising(np.uint8(image), None)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # image = cv2.GaussianBlur(image, kernel, 0)
    flags = cv2.THRESH_BINARY
    if otsus:
        flags += cv2.THRESH_OTSU

    _, image = cv2.threshold(image, threshold, 255, flags)

    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(image)
    return components, small_mask, stats


def detect_objects_both(
    salicencyMap, backsub, threshold=30, kernel=(15, 15), otsus=False
):
    if salicencyMap is not None:
        salicencyMap = np.uint8(salicencyMap)
        # image = cv2.fastNlMeansDenoising(np.uint8(image), None)

        salicencyMap = cv2.morphologyEx(salicencyMap, cv2.MORPH_OPEN, kernel)

        flags = cv2.THRESH_BINARY
        if otsus:
            flags += cv2.THRESH_OTSU

        _, salicencyMap = cv2.threshold(salicencyMap, threshold, 255, flags)

    backsub = np.uint8(backsub)
    backsub = cv2.GaussianBlur(backsub, kernel, 0)
    flags = cv2.THRESH_BINARY
    if otsus:
        flags += cv2.THRESH_OTSU
    _, backsub = cv2.threshold(backsub, threshold, 255, flags)
    # cv2.imshow("theshold", image)
    backsub = cv2.dilate(backsub, kernel, iterations=1)

    backsub = cv2.morphologyEx(backsub, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("backsub.png", np.uint8(backsub))
    both = backsub
    if salicencyMap is not None:
        # cv2.imshow("salicencyMap.png", np.uint8(salicencyMap))
        both = backsub | salicencyMap
        # cv2.imshow("both.png", np.uint8(both))

    # cv2.waitKey(10)

    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(both)
    return components, small_mask, stats


def detect_objects(image, otsus=False, threshold=30, kernel=(15, 15)):
    image = np.uint8(image)
    image = cv2.GaussianBlur(image, kernel, 0)
    flags = cv2.THRESH_BINARY
    if otsus:
        flags += cv2.THRESH_OTSU
    _, image = cv2.threshold(image, threshold, 255, flags)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return cv2.connectedComponentsWithStats(image)


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


def hist_diff(region, background, thermal, normalize_images=False):
    track_back = region.subimage(background).copy()
    track_thermal = region.subimage(thermal).copy()
    if normalize_images:
        track_back, _ = normalize(track_back, new_max=255)
        track_thermal, _ = normalize(track_thermal, new_max=255)
        track_back = np.float32(track_back)
        track_thermal = np.float32(track_thermal)
    h_bins = 60
    # s_bins = 60
    histSize = [h_bins]

    hist_base = cv2.calcHist(
        [track_back],
        None,
        None,
        histSize,
        [0, 255],
        accumulate=False,
    )
    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_track = cv2.calcHist(
        [track_thermal],
        None,
        None,
        histSize,
        [0, 255],
        accumulate=False,
    )

    cv2.normalize(
        hist_track,
        hist_track,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
    )
    compared_v = cv2.compareHist(hist_track, hist_base, 0)
    return compared_v
