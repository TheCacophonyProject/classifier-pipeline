import numpy as np
import random
from ml_tools import tools
from ml_tools.frame import TrackChannels
import logging
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
    from ml_tools.imageprocessing import resize_cv
    frame = resize_cv(
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



def preprocess_frame_v2(
    frame,
    out_dim,
    region,
    crop_rectangle=None,
    median=None,
):
    from ml_tools.imageprocessing import adapt_hist,normalize
    THERMAL_MIN_KV =  27315 
    THERMAL_MAX_KV = 31515 #42 celcius
    # enlarge to
    enlarged_region = region.copy()

    if region.width > out_dim or region.height >out_dim:

        enlarged_region.enlarge_for_rotation(final_dim = out_dim,extra_needed = 0)
    else:
        enlarged_region.enlarge_to(out_dim)

    # enlarged_region.enlarge_to(out_dim)
    # logging.info("original region %s now %s",region,enlarged_region)
    cropped_frame = frame.crop_by_region_with_padding(enlarged_region,crop_rectangle,out_dim)
    cropped_frame.float_arrays()
    # logging.info("Frame has been cropped  was %s now is %s",frame.thermal.shape,cropped_frame.thermal.shape)
    cropped_frame.thermal_norm = cropped_frame.thermal.copy()
    cropped_frame.thermal_norm -= median
    if np.median(cropped_frame.thermal_norm) >= 0:
        np.clip(
            cropped_frame.thermal_norm, a_min=0, a_max=None, out=cropped_frame.mask
        )
    cropped_frame.thermal_norm, _ = normalize(
        cropped_frame.thermal_norm,
    )
    cropped_frame.thermal_norm =adapt_hist(cropped_frame.thermal_norm)*255
    

    cropped_frame.filtered, _ = normalize(
        cropped_frame.filtered,
    )
    cropped_frame.filtered =adapt_hist(cropped_frame.filtered) * 255
    np.clip(cropped_frame.thermal, THERMAL_MIN_KV, THERMAL_MAX_KV,out = cropped_frame.thermal)

    cropped_frame.thermal, _ = normalize(
        cropped_frame.thermal,
        min=THERMAL_MIN_KV,
        max=THERMAL_MAX_KV,
        new_max=255
    )

    if cropped_frame.region.width > out_dim or cropped_frame.region.height >out_dim:

        # downsize
        cropped_frame.resize_with_aspect(
            (out_dim,out_dim),
            crop_rectangle,
            keep_edge=False,
            original_region=region,
            interpolation = cv2.INTER_AREA
        )
    cropped_frame.preprocessed = True
    return cropped_frame

def preprocess_frame(
    frame,
    out_dim,
    region,
    background=None,
    crop_rectangle=None,
    calculate_filtered=True,
    filtered_norm_limits=None,
    thermal_norm_limits=None,
    cropped=False,
    sub_median=True,
    median=None,
    clip_thermals_at_zero=True,
):
    from imageprocessing import normalize
    if sub_median and median is None:
        median = np.median(frame.thermal)
    if not cropped:
        cropped_frame = frame.crop_by_region(region, only_thermal=calculate_filtered)
    else:
        cropped_frame = frame
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
    if sub_median:
        cropped_frame.thermal -= median
    if thermal_norm_limits is None and clip_thermals_at_zero:
        np.clip(cropped_frame.thermal, 0, None, out=cropped_frame.thermal)

    if filtered_norm_limits is not None:
        cropped_frame.filtered, stats = normalize(
            cropped_frame.filtered,
            min=filtered_norm_limits[0],
            max=filtered_norm_limits[1],
            new_max=255,
        )
        if frame.thermal is not None:
            thermal_min = None
            thermal_max = None
            if thermal_norm_limits is not None:
                thermal_min, thermal_max = thermal_norm_limits
            cropped_frame.thermal, _ = normalize(
                cropped_frame.thermal, min=thermal_min, max=thermal_max, new_max=255
            )
    else:
        cropped_frame.normalize()
    cropped_frame.preprocessed = True
    return cropped_frame


index = 0


def preprocess_single_frame(
    preprocessed_frame,
    channels,
    preprocess_fn=None,
    save_info="",
):
    data = []
    for channel in channels:
        if isinstance(channel, str):
            channel = TrackChannels[channel]
        data.append(preprocessed_frame.get_channel(channel))

    image = np.stack(
        data,
        axis=2,
    )
    # global index
    # index += 1
    # tools.saveclassify_image(
    #     image,
    #     f"samples/{save_info}-{index}",
    # )

    if preprocess_fn:
        image = preprocess_fn(image)
    return image


index = 0


#
def preprocess_movement(
    preprocess_frames,
    frames_per_row,
    frame_size,
    channels,
    preprocess_fn=None,
    sample=None,
    seed = None
):
    from ml_tools.imageprocessing import square_clip
    frame_types = {}
    data = []
    frame_samples = list(np.arange(len(preprocess_frames)))
    if len(preprocess_frames) < frames_per_row * 5:
        rng = np.random.default_rng(seed)
        extra_samples = rng.choice(frame_samples, frames_per_row * 5 - len(preprocess_frames))

        frame_samples.extend(extra_samples)
        frame_samples.sort()
    for channel in channels:
        if isinstance(channel, str):
            channel = TrackChannels[channel]
        if channel in frame_types:
            data.append(frame_types[channel])
            continue
        channel_segment = [frame.get_channel(channel) for frame in preprocess_frames]
  
        channel_data, success = square_clip(
            channel_segment,
            frames_per_row,
            (frame_size, frame_size),
            frame_samples,
            normalize=False,
        )
        # already done normalization

        if not success:
            return None
        data.append(channel_data)
        frame_types[channel] = channel_data
    data = np.stack(data, axis=2)
    #
    # # # # # # for testing
    # global index
    # index += 1
    # tools.saveclassify_image(
    #     data,
    #     f"samples/{sample}-{index}",
    # )

    if preprocess_fn:
        data = preprocess_fn(data)

    return np.float32(data)
