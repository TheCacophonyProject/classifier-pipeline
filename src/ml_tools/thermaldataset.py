import math
import matplotlib.pyplot as plt

import tensorflow as tf
from functools import partial
import numpy as np
import time
from config.config import Config
import json
from ml_tools.logs import init_logging
import logging

from ml_tools.featurenorms import mean_v, std_v
from ml_tools.frame import TrackChannels
from pathlib import Path
from ml_tools.tools import saveclassify_image

# seed = 1341
# tf.random.set_seed(seed)
# np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64

insect = None
fp = None


# labels can be any subset of this, prevents new labels being trained on until we explicitly add them to here
def get_acceptable_labels(remapped_labels):
    # logging.warning("Need to add remapped labels into acceptable labels")

    accepted_labels = [
        "bird",
        "cat",
        "deer",
        "dog",
        "false-positive",
        "hedgehog",
        "human",
        "kiwi",
        "leporidae",
        "mustelid",
        "penguin",
        "possum",
        "rodent",
        "sheep",
        "vehicle",
        "wallaby",
        "weka",
        "chicken",
    ]
    for k, v in remapped_labels.items():
        if v in accepted_labels and k not in accepted_labels:
            accepted_labels.append(k)
    return accepted_labels


def get_excluded():
    return [
        "noise",
        "agouti",
        "animal",
        "goat",
        "lizard",
        "not identifiable",
        "other",
        "pest",
        "sealion",
        "bat",
        "mammal",
        "frog",
        "static",
        # added gp forretrain
        "wombat",
        "bandicoot",
        "horse",
        "otter",
        "pig",
        "cow",
        # "gray kangaroo",
        # "echidna",
        # "fox",
        # "deer",
        # "sheep",
        # "wombat",
    ]


def get_remapped(multi_label=False):
    land_bird = "bird"

    mappings = {
        "brushtail possum": "possum",
        "fox": "dog",
        "echidna": "hedgehog",
        "kangaroo": "wallaby",
        "grey kangaroo": "wallaby",
        "sambar deer": "deer",
        "mouse": "rodent",
        "rat": "rodent",
        "rain": "false-positive",
        "water": "false-positive",
        "insect": "false-positive",
        "allbirds": "bird",
        "black swan": land_bird,
        "brown quail": land_bird,
        "california quail": land_bird,
        "duck": land_bird,
        "pheasant": land_bird,
        "pukeko": land_bird,
        "quail": land_bird,
    }
    if not multi_label:
        mappings["chicken"] = "bird"
        mappings["weka"] = "bird"
    return mappings


def get_extra_mappings(labels):
    land_birds = ["chicken", "weka"]
    if "bird" not in labels:
        logging.info("Extra mappings none")
        return None
    bird_index = labels.index("bird")
    values = []
    keys = []
    for l in land_birds:
        if l in labels:
            l_i = labels.index(l)
            keys.append(l_i)
            values.append(bird_index)
    if len(keys) == 0:
        return None
    extra_label_map = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values),
        ),
        default_value=tf.constant(-1),
        name="extra_label_map",
    )
    for key, value in zip(keys, values):
        logging.info("Extra label mapping is %s to %s ", labels[key], labels[value])
    return extra_label_map


rotation_augmentation = None


def load_dataset(filenames, remap_lookup, labels, args):
    deterministic = args.get("deterministic", False)
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = (
        deterministic  # disable order, increase speed
    )
    dataset = tf.data.TFRecordDataset(
        filenames, compression_type="GZIP", num_parallel_reads=4
    )
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order

    image_size = args["image_size"]
    augment = args.get("augment", False)
    preprocess_fn = args.get("preprocess_fn")
    include_features = args.get("include_features", False)
    only_features = args.get("only_features", False)
    one_hot = args.get("one_hot", True)
    pads = args["pads"]
    logging.info("Using mean paddings %s", pads)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    extra_label_map = None
    if args.get("multi_label"):
        extra_label_map = get_extra_mappings(labels)
        logging.info("Using multi label")

    rows = 5

    # Ensure the division happens in float space, then compute the mosaic size
    mosaic_size = tf.cast(image_size[0], tf.float32) / tf.cast(rows, tf.float32)
    mosaic_larger_size = tf.math.floor(mosaic_size * 1.41)

    mosaic_size = tf.cast(mosaic_size, dtype=tf.int32)
    mosaic_larger_size = tf.cast(mosaic_larger_size, dtype=tf.int32)
    # larger images are taken for rotating and random cropping augmentation
    # TODO could take small ones for datasets other than test
    difference = tf.math.subtract(mosaic_larger_size, mosaic_size)
    padding = tf.math.ceil(tf.cast(difference, dtype=tf.float32) / 2.0)

    # Cast everything to int32 at the end
    padding = tf.cast(padding, dtype=tf.int32)
    channels = args.get(
        "channels",
        [
            TrackChannels.thermal.name,
            TrackChannels.thermal_norm.name,
            TrackChannels.filtered.name,
        ],
    )
    mean_pad_values = []
    pads = pads.to_dict()
    for channel in channels:
        mean_pad_values.append(pads[channel])

    mean_pad_values = tf.constant(mean_pad_values, dtype=tf.float32)
    logging.info("Mean pad values become %s", mean_pad_values)
    global rotation_augmentation
    rotation_augmentation = RandomRotationPerChannelFill(
        # Tested at 0.5 and 0.1 seems to work best
        factor=0.1,
        # fill_values=mean_pad_values,
    )
    dataset = dataset.map(
        partial(
            read_tfrecord,
            image_size=image_size,
            remap_lookup=remap_lookup,
            num_labels=len(labels),
            mosaic_size=mosaic_size,
            mosaic_larger_size=mosaic_larger_size,
            padding=padding,
            mean_pad_values=mean_pad_values,
            augment=augment,
            preprocess_fn=preprocess_fn,
            include_features=include_features,
            only_features=only_features,
            one_hot=one_hot,
            extra_label_map=extra_label_map,
            include_track=args.get("include_track", False),
            num_frames=args.get("num_frames", 25),
            channels=args.get(
                "channels",
                [
                    TrackChannels.thermal.name,
                    TrackChannels.thermal_norm.name,
                    TrackChannels.filtered.name,
                ],
            ),
            repeat_frames=args.get("single_input", False),
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    if only_features:
        filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x))
    else:
        filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x["input_image"][0]))

    dataset = dataset.filter(filter_nan)

    # if features are missing they wil be 0 size
    if args.get("only_features"):
        filter_none = lambda x, y: tf.size(x) > 0
        dataset = dataset.filter(filter_none)
    elif args.get("include_features"):
        filter_none = lambda x, y: tf.size(x[1]) > 0
        dataset = dataset.filter(filter_none)

    rebalance = args.get("rebalance", False)
    # rebalance
    if rebalance:
        target = 2000
        logging.info("Rebalancing to target %s", target)
        keep_probs = [min(1.0, target / CLASS_TOTALS[lbl]) for lbl in labels]
        keep_probs = tf.constant(keep_probs)

        dataset = dataset.filter(
            lambda img, lbl: randomized_balance_filter(img, lbl, keep_probs)
        )
    elif args.get("downsize_fp", False):
        # down size false-positive class
        fp_target = int(CLASS_TOTALS["false-positive"] * 0.1856)
        logging.info("Downsizing false positives to %s", fp_target)
        keep_probs = [0.1856 if lbl == "false-positive" else 1.0 for lbl in labels]
        # keep_probs=keep_probs[:10]
        keep_probs = tf.constant(keep_probs)
        logging.info("Keep probs are %s", keep_probs)
        dataset = dataset.filter(
            lambda img, lbl: randomized_balance_filter(img, lbl, keep_probs)
        )

    if augment:
        dataset = prepare_cutmix_dataset(
            dataset, img_size=image_size[0], prob=args.get("cutmix_prob", 0.4)
        )
    else:
        # remove num_frames_used from y
        dataset = dataset.map(
            lambda x, y: (x, y["label"]), num_parallel_calls=tf.data.AUTOTUNE
        )
    return dataset


import tensorflow as tf

# 1. Define your original class totals to calculate downsampling rates
CLASS_TOTALS = {
    "bird": 107746,
    "false-positive": 93317,
    "rodent": 53460,
    "leporidae": 25454,
    "human": 17397,
    "possum": 8487,
    "chicken": 7988,
    "vehicle": 7150,
    "mustelid": 6483,
    "hedgehog": 5066,
    "kiwi": 4679,
    "cat": 3944,
    "dog": 1938,
    "penguin": 1903,
    "wallaby": 1410,
    "weka": 174,
    "deer": 293,
    "sheep": 709,
}
TARGET = 2000.0

# Pre-calculate keeping probabilities (e.g., bird = 0.0103, wallaby = 1.0)
KEEP_PROBS = [min(1.0, TARGET / CLASS_TOTALS[k]) for k in CLASS_TOTALS.keys()]
KEEP_PROBS_TF = tf.constant(KEEP_PROBS, dtype=tf.float32)


# 3. Randomized rejection filter (Handles multi-label logic)
def randomized_balance_filter(image, label, keep_probs):
    # Multiply the image's multi-label tags by their respective downsample probabilities
    # e.g., if it's a chicken: label=[1,0,0,...,1,...], probs=[0.01, ..., 0.12, ...]
    print(label)
    active_probs = label["label"] * keep_probs

    # Find the maximum probability among the labels present in this image.
    # This ensures a chicken or a rare bird isn't accidentally rejected by the bird tag!
    max_keep_prob = tf.reduce_max(active_probs)

    # Generate a random uniform number between 0 and 1
    random_roll = tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32)

    # Keep the image if the random roll falls under its maximum target probability
    return random_roll < max_keep_prob


class RandomRotationPerChannelFill(tf.keras.layers.Layer):
    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self._rotation = SequenceRotation(factor)

    def call(self, inputs, fill_values, training=False):
        # fill_values: [num_frames, channels] per-frame mean for each channel.
        # ImageProjectiveTransformV3 only takes a single scalar fill_value, so
        # shift by the mean, rotate with a CONSTANT 0 fill, then shift back -
        # areas rotated into view land exactly on the per-frame channel mean.
        fill = fill_values[:, tf.newaxis, tf.newaxis, :]
        rotated = self._rotation(inputs - fill, training=training)
        return rotated + fill


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomBrightness(0.2),  # better per frame or per sequence??
        tf.keras.layers.RandomContrast(0.5),
    ]
)

import math


class SequenceRotation(tf.keras.layers.Layer):
    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, inputs, training=None):
        # Expected unbatched input shape: (25, 32, 32, channels)
        if not training:
            return inputs

        # 1. Capture spatial dimensions dynamically
        num_frames = tf.shape(inputs)[0]
        height = tf.cast(tf.shape(inputs)[1], tf.float32)
        width = tf.cast(tf.shape(inputs)[2], tf.float32)

        # 2. Generate exactly ONE random angle for this single sequence
        # factor * 2pi determines the max rotation range
        max_angle = self.factor * 2.0 * math.pi
        angle = tf.random.uniform([], -max_angle, max_angle)

        # 3. Duplicate this single scalar angle across all 25 frames
        flat_angles = tf.repeat(angle, num_frames)

        # 4. Calculate the transformation matrix elements
        cos_theta = tf.cos(flat_angles)
        sin_theta = tf.sin(flat_angles)

        x_offset = (width - 1.0) / 2.0
        y_offset = (height - 1.0) / 2.0

        # Build the rotation matrix variables
        a0 = cos_theta
        a1 = -sin_theta
        a2 = x_offset - x_offset * cos_theta + y_offset * sin_theta
        b0 = sin_theta
        b1 = cos_theta
        b2 = y_offset - x_offset * sin_theta - y_offset * cos_theta

        # Stack into the required 8-element projective transform vector
        transforms = tf.stack(
            [a0, a1, a2, b0, b1, b2, tf.zeros_like(a0), tf.zeros_like(a0)], axis=1
        )

        # 5. Rotate all frames simultaneously using BILINEAR & a CONSTANT 0 fill.
        # The caller (RandomRotationPerChannelFill) shifts by the per-channel
        # mean beforehand so this 0 fill lands on the right value once shifted back.
        rotated_sequence = tf.raw_ops.ImageProjectiveTransformV3(
            images=inputs,  # Direct (25, 32, 32, channels) input
            transforms=transforms,
            output_shape=tf.cast([height, width], tf.int32),
            interpolation="BILINEAR",  # Perfect for smooth 8-bit thermal gradients
            fill_mode="CONSTANT",
            fill_value=0.0,
        )

        return rotated_sequence


# order the "image/means" feature is written in, see thermalwriter.py create_tf_example
MEANS_CHANNEL_ORDER = [
    TrackChannels.thermal.name,
    TrackChannels.filtered.name,
    TrackChannels.thermal_norm.name,
]


def read_tfrecord(
    example,
    image_size,
    remap_lookup,
    num_labels,
    mosaic_size,
    mosaic_larger_size,
    padding,
    mean_pad_values,
    augment=False,
    preprocess_fn=None,
    only_features=False,
    one_hot=True,
    include_features=False,
    extra_label_map=None,
    include_track=False,
    num_frames=25,
    channels=[
        TrackChannels.thermal.name,
        TrackChannels.thermal_norm.name,
        TrackChannels.filtered.name,
    ],
    repeat_frames=False,
):
    logging.info(
        "Read tf record with image %s lbls %s aug  %s  prepr %s only features %s one hot %s include fetures %s num frames %s mosaic_size %s mosaic_enalrged %s padding %s",
        image_size,
        num_labels,
        augment,
        preprocess_fn,
        only_features,
        one_hot,
        include_features,
        num_frames,
        mosaic_size,
        mosaic_larger_size,
        padding,
    )
    logging.info("Channels are %s", channels)
    load_images = not only_features
    tfrecord_format = {
        "image/class/label": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/num_frames": tf.io.FixedLenFeature((), tf.int64, 25),
        "image/frame_numbers": tf.io.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True
        ),
        "image/roi": tf.io.FixedLenFeature([], tf.string),
        "image/means": tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True
        ),
    }

    if load_images:
        if TrackChannels.filtered.name in channels:
            tfrecord_format["image/filtered_encoded"] = tf.io.FixedLenSequenceFeature(
                [], dtype=tf.float32, allow_missing=True
            )
        if TrackChannels.thermal_norm.name in channels:
            tfrecord_format["image/thermal_norm_encoded"] = (
                tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)
            )
        if TrackChannels.thermal.name in channels:
            tfrecord_format["image/thermal_raw_encoded"] = (
                tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)
            )
    if include_track:
        tfrecord_format["image/track_id"] = tf.io.FixedLenFeature((), tf.int64, -1)
        tfrecord_format["image/avg_mass"] = tf.io.FixedLenFeature((), tf.int64, -1)
    if include_features or only_features:
        tfrecord_format["image/features"] = tf.io.FixedLenSequenceFeature(
            [36 * 5 + 8], dtype=tf.float32, allow_missing=True
        )
    example = tf.io.parse_single_example(example, tfrecord_format)
    record_frames = example["image/num_frames"]
    frame_indices = example["image/frame_numbers"]
    means = example["image/means"]

    regions = tf.io.decode_raw(example["image/roi"], out_type=tf.uint8)
    regions = tf.reshape(regions, [record_frames, 4])
    # written as [thermal, filtered, thermal_norm] per frame, see thermalwriter.py
    means = tf.reshape(means, [record_frames, 3])
    mean_indices = [MEANS_CHANNEL_ORDER.index(c) for c in channels]
    frame_means = tf.gather(means, mean_indices, axis=1) * 255.0

    print("Regions are", regions)
    record_frames = tf.cast(record_frames, tf.int32)
    if load_images:
        if TrackChannels.thermal_norm.name in channels:
            thermalnorm = 255.0 * example["image/thermal_norm_encoded"]
            thermals = tf.reshape(
                thermalnorm, [record_frames, mosaic_larger_size, mosaic_larger_size, 1]
            )
        if TrackChannels.filtered.name in channels:
            filteredencoded = 255.0 * example["image/filtered_encoded"]
            filtered = tf.reshape(
                filteredencoded,
                [record_frames, mosaic_larger_size, mosaic_larger_size, 1],
            )
        if TrackChannels.thermal.name in channels:
            rawthermal = 255.0 * example["image/thermal_raw_encoded"]
            rawthermal = tf.reshape(
                rawthermal, [record_frames, mosaic_larger_size, mosaic_larger_size, 1]
            )

        rgb_image = None

        for type in channels:
            if type == TrackChannels.thermal_norm.name:
                image = thermals
            elif type == TrackChannels.filtered.name:
                image = filtered
            elif type == TrackChannels.thermal.name:
                image = rawthermal
            if rgb_image is None:
                rgb_image = image
            else:
                rgb_image = tf.concat((rgb_image, image), axis=3)
        # rotation augmentation before tiling
        if augment:
            logging.info("Augmenting")

            rgb_image = rotation_augmentation(rgb_image, frame_means, training=True)
            random_value = tf.random.uniform(
                shape=[], minval=0.0, maxval=1.0, dtype=tf.float32
            )

            if tf.greater(random_value, 0.5):
                rgb_image = tf.image.flip_left_right(rgb_image)

            rgb_image = tf.image.random_crop(
                rgb_image, size=[record_frames, mosaic_size, mosaic_size, 3]
            )
            rgb_image, frame_indices = mask_random_frames(
                rgb_image, frame_indices, record_frames
            )
            record_frames = tf.shape(frame_indices)[0]
        else:
            rgb_image = tf.image.crop_to_bounding_box(
                rgb_image, padding, padding, mosaic_size, mosaic_size
            )

        mask = get_frame_mask(record_frames, frame_indices)

        zero_pad = True
        # times = tf.concat([tf.cast(frame_indices,tf.float32), tf.fill([25 - record_frames], -1.0)], axis=0)

        # ',times)

        if num_frames > 1 and not repeat_frames:
            pad_size = num_frames - tf.shape(rgb_image)[0]
            ch_r = tf.pad(
                rgb_image[..., 0:1],
                [[0, pad_size], [0, 0], [0, 0], [0, 0]],
                constant_values=0,
                # mean_pad_values[0],
            )
            ch_g = tf.pad(
                rgb_image[..., 1:2],
                [[0, pad_size], [0, 0], [0, 0], [0, 0]],
                constant_values=0,
            )
            ch_b = tf.pad(
                rgb_image[..., 2:3],
                [[0, pad_size], [0, 0], [0, 0], [0, 0]],
                constant_values=0,
            )
            rgb_image = tf.concat([ch_r, ch_g, ch_b], axis=-1)
            rgb_image = tf.ensure_shape(
                rgb_image, [num_frames, mosaic_size, mosaic_size, 3]
            )

        elif num_frames > 1:
            logging.info("Repeating frames to make 25")
            # this repeats frames to make 25
            actual_frames = tf.shape(rgb_image)[0]
            repeat_indices = tf.random.shuffle(
                tf.tile(tf.range(actual_frames), [num_frames // actual_frames + 1])
            )[:num_frames]
            repeat_indices = tf.sort(repeat_indices)
            rgb_image = tf.gather(rgb_image, repeat_indices)
            rgb_image = tf.ensure_shape(
                rgb_image, [num_frames, mosaic_size, mosaic_size, 3]
            )
            record_frames = 25
        rgb_image = tile_images(rgb_image)

        rgb_image = tf.ensure_shape(rgb_image, [*image_size, 3])

    label = tf.cast(example["image/class/label"], tf.int32)
    label = remap_lookup.lookup(label)
    if extra_label_map is not None:
        extra = extra_label_map.lookup(label)
        label = tf.stack([label, extra], axis=0)
    if one_hot:
        label = tf.one_hot(label, num_labels)
        if extra_label_map is not None:
            label = tf.reduce_max(label, axis=0)
    label = tf.cast(label, dtype=tf.float32)
    if include_track:

        track_id = tf.cast(example["image/track_id"], tf.int32)
        avg_mass = tf.cast(example["image/avg_mass"], tf.int32)
        label = (label, track_id, avg_mass)
    if not include_features and not only_features:
        return {"input_image": rgb_image, "input_mask": mask}, {
            "label": label,
            "num_frames": record_frames,
        }

    if include_features or only_features:
        # TODO this has not been updated to work with cut mix
        features = tf.squeeze(example["image/features"])
        if only_features:
            return features, label

        return (rgb_image, features), label
    # TODO this has not been updated to work with cut mix
    # if only_features:
    #     return tf.squeeze(example["image/features"])
    # elif include_features:
    #     return (rgb_image, tf.squeeze(example["image/features"]))
    # return rgb_image


def prepare_cutmix_dataset(dataset_original, img_size, prob):
    # 1. Create a second dataset and shuffle it to mix different images together
    dataset_shuffled = dataset_original.shuffle(buffer_size=4096)

    # 2. Zip them together so each element is ((img1, lbl1), (img2, lbl2))
    zipped_dataset = tf.data.Dataset.zip((dataset_original, dataset_shuffled))

    # 3. Map the CutMix function (passed via lambda to include parameters)
    cutmix_dataset = zipped_dataset.map(
        lambda d1, d2: video_mosaic_cutmix(d1, d2, img_size, prob),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return cutmix_dataset


def video_mosaic_cutmix(
    data1, data2, img_size, prob, grid_rows=5, grid_cols=5, alpha=0.3
):
    x1, y1 = data1
    image1 = x1["input_image"]
    mask1 = x1["input_mask"]
    label1 = y1["label"]
    logging.info("Chance of a cut mix on an input is %s and alpha %s", prob, alpha)

    def no_mix():
        return {"input_image": image1, "input_mask": mask1}, label1

    def mix():
        frames_used1 = y1["num_frames"]

        x2, y2 = data2
        image2 = x2["input_image"]

        label2 = y2["label"]
        mask2 = x2["input_mask"]
        frames_used2 = y2["num_frames"]
        num_frames = tf.math.minimum(frames_used1, frames_used2)
        # 1. Define dimensions of an individual sub-frame
        sub_h = img_size // grid_rows
        sub_w = img_size // grid_cols
        total_cells = grid_rows * grid_cols

        # 2. Sample how many sub-frames to replace (from Beta distribution)
        beta_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
        lam = beta_dist.sample([])

        # Convert lambda to a discrete number of blocks to swap (at least 1, at most num_frames-1)
        # Only swap within the real frames to avoid cutmixing empty/tiled cells
        num_blocks_to_swap = tf.cast(
            tf.math.round((1.0 - lam) * tf.cast(num_frames, tf.float32)), tf.int32
        )
        num_blocks_to_swap = tf.clip_by_value(num_blocks_to_swap, 1, num_frames - 1)

        # 3. Create a random binary mask for the grid cells
        # Only shuffle within the real frame indices to avoid empty cells
        real_indices = tf.random.shuffle(tf.range(num_frames))
        swap_indices = real_indices[:num_blocks_to_swap]

        all_indices = tf.range(total_cells)
        # Build a 1D boolean mask for the cells
        cell_mask_1d = tf.reduce_any(
            tf.equal(tf.expand_dims(all_indices, 0), tf.expand_dims(swap_indices, 1)),
            axis=0,
        )

        # Reshape the 1D mask back into the 2D grid shape (e.g., 2x2)
        grid_mask = tf.reshape(cell_mask_1d, [grid_rows, grid_cols])

        # 4. Upsample the grid mask to full image resolution using block repeats
        mask_expanded = tf.repeat(grid_mask, repeats=sub_h, axis=0)
        mask_expanded = tf.repeat(mask_expanded, repeats=sub_w, axis=1)

        # Add a channel dimension and cast to float
        mask_expanded = tf.cast(mask_expanded[:, :, tf.newaxis], tf.float32)

        # 5. Blend the two mosaic images using our clean grid-aligned mask
        mixed_image = image1 * (1.0 - mask_expanded) + image2 * mask_expanded

        # 6. Compute exact adjusted lambda based on how many cells were swapped
        actual_swap_ratio = tf.cast(num_blocks_to_swap, tf.float32) / tf.cast(
            total_cells, tf.float32
        )
        adjusted_lam = 1.0 - actual_swap_ratio

        # Blend the one-hot labels
        mixed_label = label1 * adjusted_lam + label2 * (1.0 - adjusted_lam)

        # Globally smooth the time steps to match the soft target label blend
        mixed_mask = mask1 * adjusted_lam + mask2 * (1.0 - adjusted_lam)
        # not sure how the mask will work with this
        return {"input_image": mixed_image, "input_mask": mask1}, mixed_label

    return tf.cond(tf.random.uniform([]) < prob, mix, no_mix)


def tile_images(images):
    s = tf.shape(images)
    t = tf.reshape(images, [5, 5, s[1], s[2], 3])
    t = tf.transpose(t, [0, 2, 1, 3, 4])
    return tf.reshape(t, [5 * s[1], 5 * s[2], 3])


from collections import Counter


# test stuff
def main():
    init_logging()
    logging.info("Loading %s", "classifier.yaml")
    config = Config.load_from_file("classifier.yaml")
    from .tfdataset import get_dataset, get_distribution, apply_label_mapping

    # file = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/training-meta.json"
    training_folder = Path(config.base_folder) / "training-data"
    meta_f = training_folder / "training-meta.json"
    with open(meta_f, "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])
    pads = meta.get("background_average")
    from ml_tools.thermalwriter import MeanData

    if pads is None:
        pads = MeanData()
    else:
        pads = MeanData(
            thermal=pads["thermal"],
            filtered=pads["filtered"],
            thermal_norm=pads["thermal_norm"],
            frames_used=1,
        )
        pads = pads * 255

    excluded_labels = get_excluded()
    # for l in labels:
    #     if l not in ["mustelid", "deer", "sheep"]:
    #         excluded_labels.append(l)

    include_track = False
    if "weka" not in labels:
        labels.append("weka")
    if "chicken" not in labels:
        labels.append("chicken")
    orig_labels = labels.copy()
    labels, tf_mappings = apply_label_mapping(
        labels, excluded_labels, get_remapped(True)
    )
    logging.info("Labels are now %s", labels)
    for k, v in tf_mappings.items():
        logging.info(
            "Original %s is mapped to %s",
            orig_labels[k],
            "Nothing" if v == -1 else labels[v],
        )

    labels, tf_mappings = apply_label_mapping(
        labels, excluded_labels, get_remapped(multi_label=True)
    )
    resampled_ds, epoch_size = get_dataset(
        load_dataset,
        training_folder / "validation",
        labels,
        batch_size=32,
        image_size=(160, 160),
        augment=True,
        shuffle=False,
        include_features=False,
        remapped_labels=get_remapped(multi_label=True),
        excluded_labels=excluded_labels,
        include_track=include_track,
        num_frames=25,
        deterministic=True,
        pads=pads,
        tf_mappings=tf_mappings,
        rebalance=False,
    )
    print("Epoch size is", epoch_size)
    # print(get_distribution(resampled_ds, len(labels), extra_meta=False))
    # return
    #
    save_dir = Path("./test-images")
    save_dir.mkdir(exist_ok=True)
    for e in range(1):
        batch_i = 0
        print("epoch", e)
        for x, y in resampled_ds:
            save_batch(x, y, labels, save_dir, tracks=include_track)
            # show_batch(x, y, labels, save=save_dir / f"{batch_i}.jpg", tracks=True)
            batch_i += 1
    # return


save_index = 0


def save_batch(image_batch, label_batch, labels, save_dir, tracks=False):
    global save_index
    masks = image_batch["input_mask"]
    image_batch = image_batch["input_image"]

    # for m in masks:
    # print("masks are ",m[0].shape,m.shape,m[1].shape)
    if tracks:
        track_batch = label_batch[1]
        label_batch = label_batch[0]
    for n, img in enumerate(image_batch):
        # print("Mask is ",masks[n],frame_indices[n])
        if tracks:
            file_title = (
                f"{labels[np.argmax(label_batch[n])]}-{track_batch[n]}-{save_index}.png"
            )
        else:
            file_title = f"{labels[np.argmax(label_batch[n])]}-{save_index}.png"
        save_index += 1
        file_name = save_dir / file_title
        saveclassify_image(img, file_name)
    #     [:,:,1]
    #     channels = img.shape[-1]
    #     repeat = 3 - channels
    #     while repeat > 0:
    #         img = np.concatenate((img, img[:, :, :1]), axis=2)
    #         repeat -= 1
    #     plt.imshow(img)
    #     if tracks:
    #         plt.title(f"{labels[np.argmax(label_batch[n])]}-{track_batch[n]}")
    #     else:
    #         plt.title(labels[np.argmax(label_batch[n])])

    #     plt.axis("off")
    # # return
    # if save:
    #     plt.savefig(save)
    # plt.show()


def show_batch(image_batch, label_batch, labels, save=None, tracks=False):
    plt.figure(figsize=(20, 20))
    print("images in batch", len(image_batch), len(label_batch))
    num_images = min(len(image_batch), 25)
    if tracks:
        track_batch = label_batch[1]
        label_batch = label_batch[0]
    for n in range(num_images):
        ax = plt.subplot(5, 5, n + 1)
        img = np.uint8(image_batch[n])[:, :, 1]
        channels = img.shape[-1]
        repeat = 3 - channels
        while repeat > 0:
            img = np.concatenate((img, img[:, :, :1]), axis=2)
            repeat -= 1
        plt.imshow(img)
        if tracks:
            plt.title(f"{labels[np.argmax(label_batch[n])]}-{track_batch[n]}")
        else:
            plt.title(labels[np.argmax(label_batch[n])])

        plt.axis("off")
    # return
    if save:
        plt.savefig(save)
    plt.show()


@tf.function
def mask_random_frames(rgb_image, frame_indices, record_frames, min_frames=15):
    """
    Drops a random number of frames from the end of rgb_image, removing up to
    (record_frames - min_frames) frames so that at least min_frames remain.
    Returns the matching frame_indices so the frame mask can be recomputed
    against the surviving frames.
    """
    max_drop = tf.maximum(record_frames - min_frames, 0)
    num_drop = tf.random.uniform(
        shape=[], minval=0, maxval=max_drop + 1, dtype=tf.int32
    )

    keep = record_frames - num_drop
    return rgb_image[:keep], frame_indices[:keep]


@tf.function
def get_frame_mask(num_valid, frame_indices, camera_fps=9):
    """
    Normalises frame intervals with a fixed 5-minute (300 seconds) ceiling
    using logarithmic compression. Camera speed is set to 9 FPS.
    """

    # Establish our fixed 5-minute logarithmic divider ceiling
    LOG_5MIN_CEILING = tf.math.log1p(300.0)  # log1p(300) equals roughly 5.7071

    # Elapsed seconds between consecutive frames at the given camera FPS
    indices = tf.cast(frame_indices, tf.float32)
    indices = indices - indices[0]
    seconds_delta = (indices[1:] - indices[:-1]) / tf.cast(camera_fps, tf.float32)
    # Apply log1p transformation to handle variation stably, then clip to [0.0, 1.0]
    normalised_delta = tf.minimum(tf.math.log1p(seconds_delta) / LOG_5MIN_CEILING, 1.0)
    # Use -1.0 as a strict geometric flag for empty padding slots
    mask_flat = tf.concat(
        [[0.0], normalised_delta, tf.fill([25 - num_valid], -1.0)], axis=0
    )

    mask = tf.reshape(mask_flat, (5, 5, 1))
    return mask


if __name__ == "__main__":
    main()
