import numpy as np
import random
from ml_tools import tools
from track.track import TrackChannels
import logging
from ml_tools import imageprocessing

import tensorflow as tf

# size to scale each frame to when loaded.
FRAME_SIZE = 32

MIN_SIZE = 4
EDGE = 1

res_x = 120
res_y = 160


def convert(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def augement_frame(frame, dim):
    frame = imageprocessing.resize_cv(
        frame,
        dim,
        extra_h=random.randint(0, int(FRAME_SIZE * 0.05)),
        extra_v=random.randint(0, int(FRAME_SIZE * 0.05)),
    )

    image = convert(frame)
    image = tf.image.random_crop(image, size=[dim[0], dim[1], 3])
    if random.random() > 0.50:
        image = tf.image.flip_left_right(image)

    if random.random() > 0.20:
        image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)
    return image.numpy()


def preprocess_segment(
    frames,
    reference_level=None,
    frame_velocity=None,
    augment=False,
    default_inset=0,
    keep_edge=False,
):
    """
    Preprocesses the raw track data, scaling it to correct size, and adjusting to standard levels
    :param frames: a list of Frames
    :param reference_level: thermal reference level for each frame in data
    :param frame_velocity: velocity (x,y) for each frame.
    :param augment: if true applies a slightly random crop / scale
    :param default_inset: the default number of pixels to inset when no augmentation is applied.
    """
    if reference_level is not None:
        # -------------------------------------------
        # next adjust temperature and flow levels
        # get reference level for thermal channel
        assert len(frames) == len(
            reference_level
        ), "Reference level shape and data shape not match."

    crop_rectangle = tools.Rectangle(EDGE, EDGE, res_x - 2 * EDGE, res_y - 2 * EDGE)

    # -------------------------------------------
    # first we scale to the standard size
    data = []
    flip = False
    if augment:
        contrast_adjust = None
        level_adjust = None
        if random.random() <= 0.75:
            # we will adjust contrast and levels, but only within these bounds.
            # that is a bright input may have brightness reduced, but not increased.
            LEVEL_OFFSET = 4

            # apply level and contrast shift
            level_adjust = float(random.normalvariate(0, LEVEL_OFFSET))
            contrast_adjust = float(tools.random_log(0.9, (1 / 0.9)))
        if random.random() <= 0.50:
            flip = True
    for i, frame in enumerate(frames):
        frame.float_arrays()
        frame_height, frame_width = frame.thermal.shape
        # adjusting the corners makes the algorithm robust to tracking differences.
        # gp changed to 0,1 maybe should be a percent of the frame size
        max_height_offset = int(np.clip(frame_height * 0.1, 1, 2))
        max_width_offset = int(np.clip(frame_width * 0.1, 1, 2))

        top_offset = random.randint(0, max_height_offset) if augment else default_inset
        bottom_offset = (
            random.randint(0, max_height_offset) if augment else default_inset
        )
        left_offset = random.randint(0, max_width_offset) if augment else default_inset
        right_offset = random.randint(0, max_width_offset) if augment else default_inset
        if frame_height < MIN_SIZE or frame_width < MIN_SIZE:
            continue

        frame_bounds = tools.Rectangle(0, 0, frame_width, frame_height)
        # rotate then crop
        if augment and random.random() <= 0.75:

            degrees = random.randint(0, 40) - 20
            frame.rotate(degrees)

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
        frame.crop_by_region(crop_region, out=frame)
        # if frame.mask is not None:
        #     assert np.all(np.mod(frame.mask, 1) == 0), "Mask isn't integer"

        try:
            frame.resize_with_aspect(
                (FRAME_SIZE, FRAME_SIZE), crop_rectangle, keep_edge=keep_edge
            )
        except Exception as e:
            logging.error("Error resizing frame %s exception %s", frame, e)
            continue
        if reference_level is not None:
            frame.thermal -= reference_level[i]
            np.clip(frame.thermal, a_min=0, a_max=None, out=frame.thermal)
        # dont think we need to do this
        # map optical flow down to right level,
        # we pre-multiplied by 256 to fit into a 16bit int
        # data[:, 2 : 3 + 1, :, :] *= 1.0 / 256.0
        if frame.flow is not None and frame.flow_clipped:
            frame.flow *= 1.0 / 256.0
            frame.flow_clipped = False
        frame.normalize()

        if augment:
            if level_adjust is not None:
                frame.brightness_adjust(level_adjust)
            if contrast_adjust is not None:
                frame.contrast_adjust(contrast_adjust)
            if flip:
                frame.flip()
        data.append(frame)

    return data, flip


def preprocess_frame(
    frame,
    augment,
    thermal_median,
    velocity,
    output_dim,
    preprocess_fn=None,
    sample=None,
):
    processed_frame, flipped = preprocess_segment(
        [frame],
        [thermal_median],
        [velocity],
        augment=augment,
        default_inset=0,
    )
    if len(processed_frame) == 0:
        return
    processed_frame = processed_frame[0]
    thermal = processed_frame.get_channel(TrackChannels.thermal)
    filtered = processed_frame.get_channel(TrackChannels.filtered)
    thermal, stats = imageprocessing.normalize(thermal, min=0)
    if not stats[0]:
        return None
    filtered, stats = imageprocessing.normalize(filtered, min=0)
    if not stats[0]:
        return None
    np.clip(filtered, a_min=0, a_max=None, out=filtered)
    np.clip(filtered, a_min=0, a_max=None, out=filtered)

    data = np.empty((*thermal.shape, 3))
    data[:, :, 0] = thermal
    data[:, :, 1] = filtered
    data[:, :, 2] = thermal
    # tools.saveclassify_image(data, f"test-{frame.frame_number}")
    # tools.saveclassify_image(
    #     data,
    #     f"samples/{sample.label}-{sample.clip_id}-{sample.track_id}",
    # )

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
    use_dots=False,
    reference_level=None,
    sample=None,
    overlay=None,
    type=None,
    keep_edge=False,
):
    segment, flipped = preprocess_segment(
        segment,
        reference_level=reference_level,
        augment=augment,
        default_inset=0,
        keep_edge=keep_edge,
    )
    if type == 2:
        # just use flow
        flow_segment = [frame.get_channel(TrackChannels.flow) for frame in segment]
        flow_square, success = imageprocessing.square_clip_flow(
            flow_segment, frames_per_row, (FRAME_SIZE, FRAME_SIZE), use_rgb=True
        )
        if success is None:
            return None
        data = flow_square
        # tools.saveclassify_rgb(
        #     data,
        #     f"samples/{type}-{sample.track.camera}-{sample.track.label}-{sample.track.clip_id}-{sample.track.track_id}-{flipped}",
        # )
        # print("RGB FLOW")
    else:
        thermal_segment = [frame.get_channel(channel) for frame in segment]
        # as long as one frame it's fine
        square, success = imageprocessing.square_clip(
            thermal_segment, frames_per_row, (FRAME_SIZE, FRAME_SIZE), type
        )
        if not success:
            return None
        filtered_segment = [
            frame.get_channel(TrackChannels.filtered) for frame in segment
        ]
        filtered_square, success = imageprocessing.square_clip(
            filtered_segment, frames_per_row, (FRAME_SIZE, FRAME_SIZE), type
        )
        # if overlay is None:
        #     dots, overlay = imageprocessing.movement_images(
        #         data,
        #         regions,
        #         dim=square.shape,
        #         require_movement=True,
        #     )
        #     overlay, stats = imageprocessing.normalize(overlay, min=0)
        #     if not stats[0]:
        #         return None
        # else:
        #     overlay, stats = imageprocessing.normalize(overlay)
        #     if stats[0] == False:
        #         return None
        #     overlay_full_size = np.zeros(square.shape)
        #     overlay_full_size[: overlay.shape[0], : overlay.shape[1]] = overlay
        #     overlay = overlay_full_size
        # if flipped:
        #     overlay = np.flip(overlay, axis=1)
        # dots = np.flip(dots, axis=1)

        data = np.empty((square.shape[0], square.shape[1], 3))

        if type == 0:
            data[:, :, 0] = filtered_square
            red = "filtered"
        else:
            data[:, :, 0] = square
            red = "thermal"
        green = "filtered"
        data[:, :, 1] = filtered_square
        # data[:, :, 2] = np.zeros(filtered_square.shape)
        if type == 3:
            flow_segment = [frame.get_channel(TrackChannels.flow) for frame in segment]
            flow_square, success = imageprocessing.square_clip_flow(
                flow_segment, frames_per_row, (FRAME_SIZE, FRAME_SIZE), use_rgb=False
            )
            if success is None:
                return None
            data[:, :, 2] = flow_square
            blue = "flow"
        else:
            data[:, :, 2] = filtered_square
            blue = "filtered"
        # print("{}-{}-{}".format(red, green, blue))
        # # for debugging
        #
        tools.saveclassify_image(
            data,
            f"samples/{type}{sample.label}-{sample.clip_id}-{sample.track_id}-{flipped}",
        )
    if preprocess_fn:
        data = data * 255
        data = preprocess_fn(data)
    return data
