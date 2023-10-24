import numpy as np
import random
from ml_tools import tools
from ml_tools.frame import TrackChannels
import logging
from ml_tools import imageprocessing
from track.region import Region
import cv2
from ml_tools.tools import FrameTypes

# size to scale each frame to when loaded.

MIN_SIZE = 4
EDGE = 1

res_x = 120
res_y = 160


# this is from tf source code same as preprocess_input
def preprocess_fn(x):
    x /= 127.5
    x -= 1.0
    return x


def convert(image):
    import tensorflow as tf

    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def augement_frame(frame, frame_size, dim):
    frame = imageprocessing.resize_cv(
        frame,
        dim,
        extra_h=random.randint(0, int(frame_size * 0.05)),
        extra_v=random.randint(0, int(frame_size * 0.05)),
    )

    image = convert(frame)
    import tensorflow as tf

    image = tf.image.random_crop(image, size=[dim[0], dim[1], 3])
    if random.random() > 0.50:
        image = tf.image.flip_left_right(image)

    if random.random() > 0.20:
        image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)
    return image.numpy()


def preprocess_frame(
    frame,
    out_dim,
    region,
    background=None,
    crop_rectangle=None,
    calculate_filtered=True,
):
    median = np.median(frame.thermal)
    cropped_frame = frame.crop_by_region(region, only_thermal=True)
    cropped_frame.thermal = np.float32(cropped_frame.thermal)
    if calculate_filtered:
        if background is None:
            logging.warning(
                "Not calculating filtered frame as no background was supplied"
            )
        else:
            cropped_frame.filtered = cropped_frame.thermal - region.subimage(background)
    cropped_frame.resize_with_aspect(
        out_dim,
        crop_rectangle,
        True,
    )
    cropped_frame.thermal -= median
    np.clip(cropped_frame.thermal, 0, None, out=cropped_frame.thermal)
    cropped_frame.normalize()
    return cropped_frame


#
#
# def preprocess_frame(
#     frame,
#     frame_size,
#     thermal_median,
#     velocity,
#     output_dim,
#     preprocess_fn=None,
#     sample=None,
# ):
#     processed_frame, flipped = preprocess_segment(
#         [frame],
#         frame_size,
#         reference_level=[thermal_median],
#         augment=augment,
#         default_inset=0,
#     )
#     if len(processed_frame) == 0:
#         return
#     processed_frame = processed_frame[0]
#     thermal = processed_frame.get_channel(TrackChannels.thermal)
#     filtered = processed_frame.get_channel(TrackChannels.filtered)
#     thermal, stats = imageprocessing.normalize(thermal, min=0)
#     if not stats[0]:
#         return None
#     filtered, stats = imageprocessing.normalize(filtered, min=0)
#     if not stats[0]:
#         return None
#
#     data = np.empty((*thermal.shape, 3))
#     data[:, :, 0] = thermal
#     data[:, :, 1] = filtered
#     data[:, :, 2] = filtered
#     # for testing
#     # tools.saveclassify_image(
#     #     data,
#     #     f"samples/{sample.label}-{sample.clip_id}-{sample.track_id}",
#     # )
#
#     # preprocess expects values in range 0-255
#     if preprocess_fn:
#         data = data * 255
#         data = preprocess_fn(data)
#     return data


index = 0


def preprocess_single_frame(
    frame,
    frame_size,
    region=None,
    preprocess_fn=None,
    save_info="",
):
    preprocessed_frame = preprocess_frame(
        frame,
        frame_size,
        region,
    )

    image = np.stack(
        (
            preprocessed_frame.thermal,
            preprocessed_frame.thermal,
            preprocessed_frame.thermal,
        ),
        axis=2,
    )
    if preprocess_fn:
        image = preprocess_fn(image)
    return image


# index = 0


#
def preprocess_movement(
    preprocess_frames,
    frames_per_row,
    frame_size,
    red_type,
    green_type,
    blue_type,
    preprocess_fn=None,
    sample=None,
):
    frame_types = {}
    channel_types = set([green_type, blue_type, red_type])
    for type in channel_types:
        if type == FrameTypes.thermal_tiled:
            channel = TrackChannels.thermal
        elif type == FrameTypes.filtered_tiled:
            channel = TrackChannels.filtered
        else:
            raise Exception("Cannot process type %s for movement", type)
        channel_segment = [frame.get_channel(channel) for frame in preprocess_frames]
        channel_data, success = imageprocessing.square_clip(
            channel_segment,
            frames_per_row,
            (frame_size, frame_size),
            normalize=False,
        )
        # already done normalization

        if not success:
            return None

        frame_types[type] = channel_data

    data = np.stack(
        (frame_types[red_type], frame_types[green_type], frame_types[blue_type]), axis=2
    )
    #
    # # # # # for testing
    # global index
    # index += 1
    # tools.saveclassify_image(
    #     data,
    #     f"samples/{index}",
    # )

    if preprocess_fn:
        data = preprocess_fn(data)
    return np.float32(data)
