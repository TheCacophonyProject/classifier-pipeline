import numpy as np
import random
from ml_tools import tools
from ml_tools.frame import TrackChannels
import logging
from track.region import Region
from ml_tools.rectangle import Rectangle
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
    crop_rectangle,
    original_dim=None,  # represent if we are taking extra padding
):
    import math
    from ml_tools.imageprocessing import adapt_hist, normalize, apply_fair_clahe
    from ml_tools.thermalwriter import THERMAL_MAX_KV, THERMAL_MIN_KV, MeanData
    from ml_tools.frame import repeat_border

    if original_dim is None:
        original_dim = out_dim
    median = np.median(frame.thermal)
    # enlarge to
    enlarged_region = region.copy()
    resize_times = None
    if region.width > out_dim or region.height > out_dim:

        enlarged_region.enlarge_for_rotation(final_dim=out_dim, extra_needed=0)
    else:
        # we are going to resize up to 4 times.
        MAX_RESIZE = 4
        resize_dim = max(region.width, region.height)
        resize_times = min(MAX_RESIZE, original_dim / resize_dim)
        if resize_times < 1:
            enlarged_region.enlarge_to(out_dim)
            # logging.info("Resizing %s enlarging to %s region %s  enlarged %s",resize_times,resize_dim ,region,enlarged_region)

            resize_times = None
        else:

            extra_padding = (out_dim - (resize_dim * resize_times)) / resize_times
            if extra_padding > 0:
                # ceil this and maybe crop it out later
                extra_padding = math.ceil(extra_padding)
                enlarged_region.enlarge_to(resize_dim + extra_padding)
            # logging.info("Resizing %s enlarging to %s region %s  enlarged %s",resize_times,resize_dim + extra_padding,region,enlarged_region)
        # logging.info("Resizing %s max %s ",resize_dim,out_dim / MAX_RESIZE)
    pad_top = 0
    pad_left = 0
    if (
        enlarged_region.width < crop_rectangle.width
        and crop_rectangle.left > enlarged_region.left
    ):
        pad_left = crop_rectangle.left - enlarged_region.left
    if (
        enlarged_region.height < crop_rectangle.height
        and crop_rectangle.top > enlarged_region.top
    ):
        pad_top = crop_rectangle.top - enlarged_region.top
    new_width = max(out_dim, min(crop_rectangle.width, enlarged_region.width))
    new_height = max(
        out_dim,
        min(crop_rectangle.height, enlarged_region.height),
    )
    content_region = enlarged_region.copy()
    content_region.crop(crop_rectangle)
    cropped_frame = frame.crop_and_copy_as_float(content_region, make_copy=True)
    cropped_frame.thermal_norm = cropped_frame.thermal.copy()
    cropped_frame.thermal_norm -= median
    if np.median(cropped_frame.thermal_norm) >= 0:
        np.clip(
            cropped_frame.thermal_norm,
            a_min=0,
            a_max=None,
            out=cropped_frame.thermal_norm,
        )
    cropped_frame.thermal_norm, stats = normalize(
        cropped_frame.thermal_norm,
    )
    if not stats[0]:
        return None

    if np.median(cropped_frame.filtered) >= 0:
        np.clip(
            cropped_frame.filtered,
            a_min=0,
            a_max=None,
            out=cropped_frame.filtered,
        )
    cropped_frame.filtered, stats = normalize(
        cropped_frame.filtered,
    )

    if not stats[0]:
        return None

    np.clip(
        cropped_frame.thermal, THERMAL_MIN_KV, THERMAL_MAX_KV, out=cropped_frame.thermal
    )

    cropped_frame.thermal, _ = normalize(
        cropped_frame.thermal, min=THERMAL_MIN_KV, max=THERMAL_MAX_KV
    )

    # apply paddings
    needs_pad = (
        pad_top > 0
        or pad_left > 0
        or content_region.width < new_width
        or content_region.height < new_height
    )
    needs_resize = (
        cropped_frame.region.width > out_dim or cropped_frame.region.height > out_dim
    )
    if needs_resize:
        # resize the real content first, then pad - padding
        # the full (new_height, new_width) canvas before
        # resizing would resize the border padding along with
        # the real content, so shrink the canvas size and
        # offsets by the same scale the content is resized by
        scale = out_dim / max(new_width, new_height)
        scaled_w = max(1, round(content_region.width * scale))
        scaled_h = max(1, round(content_region.height * scale))

        cropped_frame.resize(
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_AREA,
        )
        pad_top = max(0, min(round(pad_top * scale), out_dim - scaled_h))
        pad_left = max(0, min(round(pad_left * scale), out_dim - scaled_w))
        new_height = out_dim
        new_width = out_dim
        # the resize may already have landed exactly on
        # (resize_dim, resize_dim) with no offset (eg. a
        # square content_region with no pad_top/pad_left),
        # in which case there's nothing left to pad
        needs_pad = (
            pad_top > 0 or pad_left > 0 or scaled_w < new_width or scaled_h < new_height
        )
    elif resize_times is not None:
        scaled_w = max(1, round(content_region.width * resize_times))
        scaled_h = max(1, round(content_region.height * resize_times))
        # t_norm = cropped_frame.thermal_norm.copy()
        # logging.info("pre %s",t_norm.shape)
        # t_norm = np.uint8(t_norm*255)
        # cv2.imshow("f",t_norm)
        # cv2.waitKey()
        cropped_frame.resize(
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_CUBIC,
        )
        # logging.info("resized %s",cropped_frame.thermal_norm.shape)

        # inter cubic can cause values out of range
        np.clip(cropped_frame.filtered, 0, 1, out=cropped_frame.filtered)
        np.clip(cropped_frame.thermal_norm, 0, 1, out=cropped_frame.thermal_norm)
        np.clip(cropped_frame.thermal, 0, 1, out=cropped_frame.thermal)
        # t_norm = cropped_frame.thermal_norm.copy()
        # t_norm = np.uint8(t_norm*255)
        # cv2.imshow("f",t_norm)
        # cv2.waitKey()
        # centre crop cropped_frame to a max of out_dim by out_dim, anything over this will be
        # from the ceil of the extra padding
        if scaled_w > out_dim or scaled_h > out_dim:
            crop_w = min(scaled_w, out_dim)
            crop_h = min(scaled_h, out_dim)
            centre_crop = Rectangle(
                (scaled_w - crop_w) // 2, (scaled_h - crop_h) // 2, crop_w, crop_h
            )
            cropped_frame.crop_by_region(centre_crop, out=cropped_frame)

        # logging.info("%s Up sizing %s to %s resize times %s %s" ,region,content_region,(scaled_w, scaled_h), resize_times ,cropped_frame.thermal.shape)
        # scale pad by the actual height/width scale applied to content_region,
        # not resize_times, since the centre crop above may have shrunk
        # the content below what resize_times alone would give
        scaled_h, scaled_w = cropped_frame.thermal.shape[:2]
        height_scale = scaled_h / content_region.height
        width_scale = scaled_w / content_region.width
        pad_top = max(0, min(round(pad_top * height_scale), out_dim - scaled_h))
        pad_left = max(0, min(round(pad_left * width_scale), out_dim - scaled_w))
        new_height = out_dim
        new_width = out_dim
        needs_pad = (
            pad_top > 0 or pad_left > 0 or scaled_w < new_width or scaled_h < new_height
        )

    # CLAHE after resize
    cropped_frame.filtered = apply_fair_clahe(
        cropped_frame.filtered, resize_times if resize_times else 1
    )

    cropped_frame.thermal_norm = apply_fair_clahe(
        cropped_frame.thermal_norm, resize_times if resize_times else 1
    )

    # calculate averages of background
    thermal_border = content_region.get_border(cropped_frame.thermal, 2, crop_rectangle)
    filtered_border = content_region.get_border(
        cropped_frame.filtered, 2, crop_rectangle
    )
    thermal_norm_border = content_region.get_border(
        cropped_frame.thermal_norm, 2, crop_rectangle
    )

    # this is the offset in the final image of our actual image
    h, w = cropped_frame.thermal.shape[:2]
    data_region = [pad_left, pad_top, w, h]
    mean_value = MeanData(
        thermal=np.mean(thermal_border),
        thermal_norm=np.mean(thermal_norm_border),
        filtered=np.mean(filtered_border),
        frames_used=len(thermal_border),
    )

    # logging.info("Mean border data is %s from filtered %s",mean_value, filtered_border)

    # i dont think this will happen
    if len(thermal_border) == 0:
        logging.info("NO thermal border so using 10% quartile")
        mean_value = MeanData(
            np.quantile(cropped_frame.thermal, 0.1),
            np.quantile(cropped_frame.filtered, 0.1),
            np.quantile(cropped_frame.thermal_norm, 0.1),
        )

    if needs_pad:
        # logging.info("Paddding %s %s %s",pad_top,pad_left,cropped_frame.thermal.shape)
        threshold = mean_value.filtered
        # pad frame by repeating the border pixels and ignoring animal content( pixels above threshold)
        repeat_border(
            cropped_frame,
            new_height,
            new_width,
            pad_top,
            pad_left,
            threshold,
            mean_value,
        )
    # t_norm = cropped_frame.thermal_norm.copy()
    # t_norm = np.uint8(t_norm*255)
    # cv2.imshow("f",t_norm)
    # cv2.waitKey()
    cropped_frame.preprocessed = True
    return cropped_frame, mean_value, data_region


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
    from ml_tools.imageprocessing import normalize

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
    seed=None,
    pad_with=None,
):
    from ml_tools.imageprocessing import square_clip

    frame_types = {}
    data = []
    frame_samples = None
    if pad_with is not None:
        frame_samples = list(np.arange(len(preprocess_frames)))
        if len(preprocess_frames) < frames_per_row * 5 and pad_with is None:
            rng = np.random.default_rng(seed)
            extra_samples = rng.choice(
                frame_samples, frames_per_row * 5 - len(preprocess_frames)
            )

            frame_samples.extend(extra_samples)
            frame_samples.sort()

    for channel in channels:
        if isinstance(channel, str):
            channel = TrackChannels[channel]
        if channel in frame_types:
            data.append(frame_types[channel])
            continue
        channel_segment = [frame.get_channel(channel) for frame in preprocess_frames]
        channel_data = square_clip(
            channel_segment,
            frames_per_row,
            (frame_size, frame_size),
            frame_samples,
            pad_with=pad_with,
        )

        data.append(channel_data)
        frame_types[channel] = channel_data
    data = np.stack(data, axis=2)
    #
    # # # # # # for testing
    # global index
    # index += 1
    # from ml_tools.tools import saveclassify_image

    # saveclassify_image(
    #     np.uint8(data),
    #     f"samples/{sample}-{index}",
    # )

    if preprocess_fn:
        data = preprocess_fn(data)
    return np.float32(data)
