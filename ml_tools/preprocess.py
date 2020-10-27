import cv2
import numpy as np
import random

from ml_tools import tools
from track.track import TrackChannels
from ml_tools import imageprocessing


# size to scale each frame to when loaded.
FRAME_SIZE = 48

MIN_SIZE = 4


def preprocess_segment(
    frames,
    reference_level=None,
    frame_velocity=None,
    augment=False,
    encode_frame_offsets_in_flow=False,
    default_inset=2,
    filter_to_delta=True,
):
    """
    Preprocesses the raw track data, scaling it to correct size, and adjusting to standard levels
    :param frames: a list of np array of shape [C, H, W]
    :param reference_level: thermal reference level for each frame in data
    :param frame_velocity: velocity (x,y) for each frame.
    :param augment: if true applies a slightly random crop / scale
    :param default_inset: the default number of pixels to inset when no augmentation is applied.
    :param filter_to_delta: If true change filterted channel to be the delta of thermal frames.
    """

    # -------------------------------------------
    # first we scale to the standard size

    # adjusting the corners makes the algorithm robust to tracking differences.
    top_offset = random.randint(0, 5) if augment else default_inset
    bottom_offset = random.randint(0, 5) if augment else default_inset
    left_offset = random.randint(0, 5) if augment else default_inset
    right_offset = random.randint(0, 5) if augment else default_inset

    scaled_frames = []

    for frame in frames:
        channels, frame_height, frame_width = frame.shape

        if frame_height < MIN_SIZE or frame_width < MIN_SIZE:
            return

        frame_bounds = tools.Rectangle(0, 0, frame_width, frame_height)

        # set up a cropping frame
        crop_region = tools.Rectangle.from_ltrb(
            left_offset,
            top_offset,
            frame_width - right_offset,
            frame_height - bottom_offset,
        )

        # if the frame is too small we make it a little larger
        while crop_region.width < MIN_SIZE:
            crop_region.left -= 1
            crop_region.right += 1
            crop_region.crop(frame_bounds)
        while crop_region.height < MIN_SIZE:
            crop_region.top -= 1
            crop_region.bottom += 1
            crop_region.crop(frame_bounds)

        cropped_frame = frame[
            :,
            crop_region.top : crop_region.bottom,
            crop_region.left : crop_region.right,
        ]

        scaled_frame = [
            cv2.resize(
                np.float32(cropped_frame[channel]),
                dsize=(FRAME_SIZE, FRAME_SIZE),
                interpolation=cv2.INTER_LINEAR
                if channel != TrackChannels.mask
                else cv2.INTER_NEAREST,
            )
            for channel in range(channels)
        ]

        scaled_frames.append(scaled_frame)

    # convert back into [F,C,H,W] array.
    data = np.float32(scaled_frames)

    if reference_level:
        # -------------------------------------------
        # next adjust temperature and flow levels
        # get reference level for thermal channel
        assert len(data) == len(
            reference_level
        ), "Reference level shape and data shape not match."

        # reference thermal levels to the reference level
        data[:, 0, :, :] -= np.float32(reference_level)[:, np.newaxis, np.newaxis]

    # map optical flow down to right level,
    # we pre-multiplied by 256 to fit into a 16bit int
    data[:, 2 : 3 + 1, :, :] *= 1.0 / 256.0

    # write frame motion into center of frame
    if encode_frame_offsets_in_flow:
        F, C, H, W = data.shape
        for x in range(-2, 2 + 1):
            for y in range(-2, 2 + 1):
                data[:, 2 : 3 + 1, H // 2 + y, W // 2 + x] = frame_velocity[:, :]

    # set filtered track to delta frames
    if filter_to_delta:
        reference = np.clip(data[:, 0], 20, 999)
        data[0, 1] = 0
        data[1:, 1] = reference[1:] - reference[:-1]

    # -------------------------------------------
    # finally apply and additional augmentation

    if augment:
        if random.random() <= 0.75:
            # we will adjust contrast and levels, but only within these bounds.
            # that is a bright input may have brightness reduced, but not increased.
            LEVEL_OFFSET = 4

            # apply level and contrast shift
            level_adjust = random.normalvariate(0, LEVEL_OFFSET)
            contrast_adjust = tools.random_log(0.9, (1 / 0.9))

            data[:, 0] *= contrast_adjust
            data[:, 0] += level_adjust
            data[:, 1] *= contrast_adjust

        if random.random() <= 0.50:
            # when we flip the frame remember to flip the horizontal velocity as well
            data = np.flip(data, axis=3)
            data[:, 2] = -data[:, 2]
    np.clip(data[:, 0, :, :], a_min=0, a_max=None, out=data[:, 0, :, :])

    return data


def preprocess_frame(
    data, output_dim, use_thermal=True, augment=False, preprocess_fn=None
):
    if use_thermal:
        channel = TrackChannels.thermal
    else:
        channel = TrackChannels.filtered
    data = data[channel]

    max = np.amax(data)
    min = np.amin(data)
    if max == min:
        return None

    data -= min
    data = data / (max - min)
    np.clip(data, a_min=0, a_max=None, out=data)

    data = data[np.newaxis, :]
    data = np.transpose(data, (1, 2, 0))
    data = np.repeat(data, output_dim[2], axis=2)
    data = imageprocessing.resize_cv(data, output_dim)

    # preprocess expects values in range 0-255
    if preprocess_fn:
        data = data * 255
        data = preprocess_fn(data)
    return data


def preprocess_movement(
    data,
    segment,
    frames_per_row,
    regions,
    channel,
    preprocess_fn=None,
    augment=False,
    use_dots=True,
    reference_level=None,
):
    segment = preprocess_segment(
        segment,
        reference_level=reference_level,
        augment=augment,
        filter_to_delta=False,
        default_inset=0,
    )

    segment = segment[:, channel]
    # as long as one frame it's fine
    square, success = imageprocessing.square_clip(
        segment, frames_per_row, (FRAME_SIZE, FRAME_SIZE), type
    )
    if not success:
        return None
    dots, overlay = imageprocessing.movement_images(
        data,
        regions,
        dim=square.shape,
        require_movement=True,
    )
    overlay, success = imageprocessing.normalize(overlay, min=0)
    if not success:
        return None

    data = np.empty((square.shape[0], square.shape[1], 3))
    data[:, :, 0] = square
    if use_dots:
        dots = dots / 255
        data[:, :, 1] = dots  # dots
    else:
        data[:, :, 1] = np.zeros(dots.shape)
    data[:, :, 2] = overlay  # overlay
    if preprocess_fn:
        for i, frame in enumerate(data):
            frame = frame * 255
            data[i] = preprocess_fn(frame)
    return data
