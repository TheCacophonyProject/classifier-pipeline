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

tf.config.set_soft_device_placement(True)

# tf.config.run_functions_eagerly(True)
# seed = 1341
# tf.random.set_seed(seed)
# np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64

# Jitter dataset epoch-staging thresholds/probabilities (see jitter_dataset)
JITTER_MEDIUM_STAGE_EPOCH = tf.constant(5, dtype=tf.int32)
JITTER_HEAVY_STAGE_EPOCH = tf.constant(15, dtype=tf.int32)
JITTER_LIGHT_STAGE_PROB = tf.constant(0.1, dtype=tf.float32)
JITTER_MEDIUM_STAGE_PROB = tf.constant(0.25, dtype=tf.float32)
JITTER_HEAVY_STAGE_PROB = tf.constant(0.4, dtype=tf.float32)

# Tracks which jitter stage (0=light, 1=medium, 2=heavy) was last announced,
# so jitter_dataset only tf.prints on the call where the stage actually changes.
LAST_JITTER_STAGE = tf.Variable(
    -1, dtype=tf.int32, trainable=False, name="last_jitter_stage"
)

insect = None
fp = None
USE_VELOCITY = True


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


def load_excluded_tracks(filename="exclude.txt"):
    path = Path(filename)
    if not path.exists():
        logging.warning(
            "Exclude tracks file %s not found, not excluding any tracks", path
        )
        return None
    with open(path, "r") as f:
        track_ids = [int(line.strip()) for line in f if line.strip()]
    logging.info("Loaded %s excluded tracks from %s", len(track_ids), path)
    keys = tf.constant(track_ids, dtype=tf.int32)
    values = tf.ones_like(keys, dtype=tf.int32)

    return tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys=keys, values=values),
        default_value=False,
        name="excluded_tracks",
    )


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

    global rotation_augmentation
    rotation_augmentation = RandomRotationPerChannelFill(
        # Tested at 0.5 and 0.1 seems to work best
        factor=0.1,
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
            repeat_frames=False,
            # args.get("single_input", False),
            current_epoch=args.get("current_epoch"),
            use_velocity=USE_VELOCITY,
            single_input=args.get("single_input", False),
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    if only_features:
        filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x))
    else:
        filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x["input_image"][0]))

    dataset = dataset.filter(filter_nan)

    logging.info("Filtering tracks that have been considered misclassified by umap")
    excluded_tracks_table = load_excluded_tracks()
    if excluded_tracks_table is not None:
        filter_bad_tracks = lambda x, y: tf.math.logical_not(
            tf.cast(excluded_tracks_table.lookup(y["label"][1]), tf.bool)
        )
        dataset = dataset.filter(filter_bad_tracks)
    dataset = dataset.map(
        lambda x, y: (x, {"label": y["label"][0], "num_frames": y["num_frames"]}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
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
        if not args.get("single_input", False):
            logging.info("Doing jitter")
            dataset = prepare_jitter_dataset(
                dataset,
                current_epoch=args.get("current_epoch"),
            )
        else:
            logging.info("Doing cutmix")
            dataset = prepare_cutmix_dataset(
                dataset,
                img_size=image_size[0],
                prob=args.get("cutmix_prob", 0.4),
                current_epoch=args.get("current_epoch"),
            )
    else:
        # remove num_frames_used from y
        dataset = dataset.map(
            lambda x, y: (x, y["label"]), num_parallel_calls=tf.data.AUTOTUNE
        )

    # this might be slightly slower than doing this here instead of when reading the records
    RNN = False
    if not RNN:
        dataset = dataset.map(
            lambda x, y: (tile_input(x, USE_VELOCITY), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset


@tf.function
def tile_input(x, use_velocity):
    input_image = tile_images(x["input_image"])
    if use_velocity:
        mask = tf.reshape(x["input_mask"], (5, 5, 7))
    else:
        mask = tf.reshape(x["input_mask"], (5, 5, 2))

    return {"input_image": input_image, "input_mask": mask}


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
        rotated, angle = self._rotation(inputs - fill, training=training)
        return rotated + fill, angle


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
            return inputs, tf.constant(0.0)

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

        return rotated_sequence, angle


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
    current_epoch=None,
    use_velocity=False,
    single_input=False,
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
        "image/original_roi": tf.io.FixedLenFeature([], tf.string),
        "image/means": tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True
        ),
    }
    if use_velocity:
        tfrecord_format["image/centre_x"] = tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True
        )
        tfrecord_format["image/centre_y"] = tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True
        )

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
    centre_x = None
    centre_y = None
    if use_velocity:
        centre_x = example["image/centre_x"]
        centre_y = example["image/centre_y"]
    # centre_x = tf.reshape(centre_x, [record_frames])
    # centre_y = tf.reshape(centre_y, [record_frames])

    regions = tf.io.decode_raw(example["image/original_roi"], out_type=tf.uint8)
    regions = tf.reshape(regions, [record_frames, 4])
    # written as [thermal, filtered, thermal_norm] per frame, see thermalwriter.py
    means = tf.reshape(means, [record_frames, 3])
    mean_indices = [MEANS_CHANNEL_ORDER.index(c) for c in channels]
    frame_means = tf.gather(means, mean_indices, axis=1) * 255.0

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
        rotation_angle = 0.0
        if augment:
            logging.info("Augmenting")

            rgb_image, rotation_angle = rotation_augmentation(
                rgb_image, frame_means, training=True
            )
            random_value = tf.random.uniform(
                shape=[], minval=0.0, maxval=1.0, dtype=tf.float32
            )

            if tf.greater(random_value, 0.5):
                rgb_image = tf.image.flip_left_right(rgb_image)
                rotation_angle += math.pi

            rgb_image = tf.image.random_crop(
                rgb_image, size=[record_frames, mosaic_size, mosaic_size, 3]
            )

            logging.info("Applying random frame mask")
            # Stage the frame-masking probability by epoch, same boundaries as
            # the cutmix staging above: heavy early on, tapering to off, read
            # live off current_epoch so it tracks training progress. Applied
            # for both single and dual input - replicates real tracks that
            # naturally run shorter than 25 frames, so (unlike jitter) the
            # label isn't softened: nothing is being faked missing here.
            epoch = current_epoch.read_value()

            def heavy_mask_prob():
                return tf.constant(0.5, dtype=tf.float32)

            def medium_mask_prob():
                return tf.constant(0.25, dtype=tf.float32)

            def off_mask_prob():
                return tf.constant(0.0, dtype=tf.float32)

            mask_frames_prob = tf.case(
                [(epoch < 15, heavy_mask_prob), (epoch < 25, medium_mask_prob)],
                default=off_mask_prob,
            )

            if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < mask_frames_prob:
                rgb_image, frame_indices = mask_random_frames(
                    rgb_image, frame_indices, record_frames
                )
                record_frames = tf.shape(frame_indices)[0]
        else:
            rgb_image = tf.image.crop_to_bounding_box(
                rgb_image, padding, padding, mosaic_size, mosaic_size
            )

        # double the resolution of each frame (mosaic_size -> mosaic_size * 2)
        rgb_image = tf.image.resize(
            rgb_image, [mosaic_size * 2, mosaic_size * 2], method="bicubic"
        )
        rgb_image = tf.clip_by_value(rgb_image, 0.0, 255.0)

        mask = get_frame_mask_v2(
            record_frames,
            frame_indices,
            centre_x,
            centre_y,
            use_velocity,
            rotation_angle,
            regions,
        )

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
            rgb_image = tf.ensure_shape(rgb_image, [num_frames, 64, 64, 3])

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
                rgb_image, [num_frames, mosaic_size * 2, mosaic_size * 2, 3]
            )
            record_frames = 25

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


def prepare_jitter_dataset(dataset_original, current_epoch):
    # 1. Create a second dataset and shuffle it to mix different images together

    # current_epoch is read inside the mapped function (not here) so that each
    # call sees the *live* value of the variable as it's updated by
    # EpochTrackerCallback across epochs, rather than baking in a constant
    # captured at dataset-construction time.
    cutmix_dataset = dataset_original.map(
        lambda x, y: jitter_dataset(x, y, current_epoch),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return cutmix_dataset


def jitter_dataset(x, y, current_epoch):
    # Read the current epoch value from the global variable graph pointer.
    # Doing this here (inside the traced map function) rather than once when
    # the dataset is built means it re-reads the variable's live value on
    # every call, so the staging below tracks CURRENT_EPOCH as training
    # progresses.

    epoch = current_epoch.read_value()

    def light_stage():
        return JITTER_LIGHT_STAGE_PROB

    def medium_stage():
        return JITTER_MEDIUM_STAGE_PROB

    def heavy_stage():
        return JITTER_HEAVY_STAGE_PROB

    # Reconfigured epoch boundary gates to scale up as training progresses
    prob = tf.case(
        [
            (epoch < JITTER_MEDIUM_STAGE_EPOCH, light_stage),
            (epoch < JITTER_HEAVY_STAGE_EPOCH, medium_stage),
        ],
        default=heavy_stage,
    )

    stage_index = tf.case(
        [
            (epoch < JITTER_MEDIUM_STAGE_EPOCH, lambda: tf.constant(0, dtype=tf.int32)),
            (epoch < JITTER_HEAVY_STAGE_EPOCH, lambda: tf.constant(1, dtype=tf.int32)),
        ],
        default=lambda: tf.constant(2, dtype=tf.int32),
    )

    def announce_new_stage():
        stage_name = tf.gather(["light", "medium", "heavy"], stage_index)
        tf.print("jitter_dataset: entering", stage_name, "stage at epoch", epoch)
        LAST_JITTER_STAGE.assign(stage_index)
        return False

    tf.cond(
        tf.not_equal(stage_index, LAST_JITTER_STAGE),
        announce_new_stage,
        lambda: False,
    )
    image = x["input_image"]  # Shape: (num_frames, size, size, 3)
    mask = x[
        "input_mask"
    ]  # Shape: (num_frames, 4) if USE_VELOCITY else (num_frames, 2)
    frames_used = tf.cast(y["num_frames"], tf.int32)

    def no_jitter():
        return {"input_image": image, "input_mask": mask}, y["label"]

    def jitter():

        # mask a random number of frames up to prob * frames_used
        max_to_mask = tf.math.maximum(
            tf.cast(tf.math.floor(prob * tf.cast(frames_used, tf.float32)), tf.int32),
            1,
        )
        num_to_mask = tf.random.uniform(
            [], minval=1, maxval=max_to_mask + 1, dtype=tf.int32
        )

        keep_gate = tf.concat(
            [
                tf.zeros([num_to_mask], dtype=tf.float32),
                tf.ones([frames_used - num_to_mask], dtype=tf.float32),
            ],
            axis=0,
        )
        keep_gate = tf.random.shuffle(keep_gate)

        keep_gate = tf.concat(
            [
                keep_gate,
                tf.zeros([tf.shape(image)[0] - frames_used], dtype=tf.float32),
            ],
            axis=0,
        )
        # mask the input image by setting these frames to zero
        jittered_image = image * keep_gate[:, tf.newaxis, tf.newaxis, tf.newaxis]
        # mask the mask by setting these frames to zero, except channel 0
        # (the absolute time reconstructed in get_frame_mask_v2) which must
        # stay untouched - it's a cumulative timeline, not a per-frame
        # signal, so zeroing a dropped frame's entry would falsely reset
        # the clock back to the start mid-sequence.
        jittered_mask = tf.concat(
            [mask[:, :1], mask[:, 1:] * keep_gate[:, tf.newaxis]], axis=-1
        )

        f_used_float = tf.cast(frames_used, tf.float32)
        n_mask_float = tf.cast(num_to_mask, tf.float32)

        fraction_retained = (f_used_float - n_mask_float) / f_used_float
        # Soften the target label so the model isn't penalized for missing data you erased
        adjusted_label = y["label"] * fraction_retained

        return {
            "input_image": jittered_image,
            "input_mask": jittered_mask,
        }, adjusted_label

    is_active_stage = tf.greater(prob, 0.0)
    min_frames_required = tf.constant(5, dtype=tf.int32)
    has_enough_frames = tf.greater_equal(frames_used, min_frames_required)

    should_jitter = tf.logical_and(is_active_stage, tf.random.uniform([]) < prob)
    should_jitter = tf.logical_and(should_jitter, has_enough_frames)
    return tf.cond(should_jitter, jitter, no_jitter)


def prepare_cutmix_dataset(dataset_original, img_size, prob, current_epoch):
    # 1. Create a second dataset and shuffle it to mix different images together
    dataset_shuffled = dataset_original.shuffle(buffer_size=4096)

    # 2. Zip them together so each element is ((img1, lbl1), (img2, lbl2))
    zipped_dataset = tf.data.Dataset.zip((dataset_original, dataset_shuffled))

    # current_epoch is read inside the mapped function (not here) so that each
    # call sees the *live* value of the variable as it's updated by
    # EpochTrackerCallback across epochs, rather than baking in a constant
    # captured at dataset-construction time.
    cutmix_dataset = zipped_dataset.map(
        lambda d1, d2: video_sequential_cutmix(d1, d2, current_epoch),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return cutmix_dataset


def video_sequential_cutmix(data1, data2, current_epoch):
    # Read the current epoch value from the global variable graph pointer.
    # Doing this here (inside the traced map function) rather than once when
    # the dataset is built means it re-reads the variable's live value on
    # every call, so the staging below tracks CURRENT_EPOCH as training
    # progresses.
    epoch = current_epoch.read_value()

    def heavy_stage():
        return tf.constant(0.4, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32)

    def medium_stage():
        return tf.constant(0.4, dtype=tf.float32), tf.constant(0.3, dtype=tf.float32)

    def off_stage():
        return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

    # Graph-safe conditional selection based on epoch boundaries
    prob, alpha = tf.case(
        [(epoch < 16, heavy_stage), (epoch < 25, medium_stage)], default=off_stage
    )

    # # Sampled so it doesn't flood the logs - remove once you've confirmed
    # # the epoch value is actually advancing across epochs.
    # if tf.random.uniform([]) < 0.001:
    #     tf.print(
    #         "[cutmix] current_epoch=", epoch, " prob=", prob, " alpha=", alpha
    #     )

    x1, y1 = data1
    image1 = x1["input_image"]  # Shape: (25, 32, 32, 3)
    mask1 = x1["input_mask"]  # Shape: (25, 1)
    label1 = y1["label"]  # One-hot encoded class label array

    def no_mix():
        return {"input_image": image1, "input_mask": mask1}, label1

    def mix():
        x2, y2 = data2
        image2 = x2["input_image"]  # Shape: (25, 32, 32, 3)
        mask2 = x2["input_mask"]  # Shape: (25, 1)
        label2 = y2["label"]

        # Track the actual valid frame count for both video windows
        frames_used1 = y1["num_frames"]
        frames_used2 = y2["num_frames"]

        # Only swap within slots where both videos contain genuine, unpadded tracking data
        num_frames = tf.math.minimum(frames_used1, frames_used2)
        total_cells = 25  # Sequence length

        # 1. Sample how many frames to replace (from Beta distribution)
        beta_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
        lam = beta_dist.sample([])

        # Convert lambda to a discrete number of frames to swap (at least 1, at most num_frames-1)
        num_blocks_to_swap = tf.cast(
            tf.math.round((1.0 - lam) * tf.cast(num_frames, tf.float32)), tf.int32
        )
        half_frames1 = frames_used1 // 2
        num_blocks_to_swap = tf.where(
            num_blocks_to_swap > half_frames1,
            frames_used1 - num_blocks_to_swap,
            num_blocks_to_swap,
        )
        num_blocks_to_swap = tf.math.maximum(num_blocks_to_swap, 1)

        # num_blocks_to_swap = tf.clip_by_value(num_blocks_to_swap, 1, num_frames - 1)

        # 2. Create a random binary mask for the 25 chronological timesteps
        real_indices = tf.random.shuffle(tf.range(num_frames))
        swap_indices = real_indices[:num_blocks_to_swap]

        all_indices = tf.range(total_cells)
        # Build a 1D boolean tensor across the 25 steps: True means swap from video 2
        time_mask_1d = tf.reduce_any(
            tf.equal(tf.expand_dims(all_indices, 0), tf.expand_dims(swap_indices, 1)),
            axis=0,
        )

        # 3. Format the binary gate for 5D elementwise multiplication
        # Expand time_mask_1d from (25,) to (25, 1, 1, 1) to match (25, 32, 32, 3) image tensors
        image_gate = tf.cast(
            time_mask_1d[:, tf.newaxis, tf.newaxis, tf.newaxis], tf.float32
        )

        # Expand time_mask_1d from (25,) to (25, 1) to match the input_mask shape
        # mask_gate = tf.cast(time_mask_1d[:, tf.newaxis], tf.float32)

        # 4. Mix the 32x32 frames and the timestamps discretely using the gates
        mixed_image = image1 * (1.0 - image_gate) + image2 * image_gate
        # mixed_mask = mask1 * (1.0 - mask_gate) + mask2 * mask_gate

        # 5. Compute the final soft target label blend based on the absolute ratio of swapped frames
        actual_swap_ratio = tf.cast(num_blocks_to_swap, tf.float32) / tf.cast(
            total_cells, tf.float32
        )
        # adjusted_lam = 1.0 - actual_swap_ratio

        # Blend the one-hot vectors normally
        # mixed_label = label1 * adjusted_lam + label2 * (1.0 - adjusted_lam)

        total_active_frames = tf.cast(frames_used1, tf.float32)
        swap_count_float = tf.cast(num_blocks_to_swap, tf.float32)

        # Calculate precise percentage weights
        weight_2 = swap_count_float / total_active_frames
        weight_1 = 1.0 - weight_2
        # because we are doing multi label just saying both are present is better
        # mixed_label = tf.maximum(label1, label2)
        # 5. LABEL LOGIC: Only mix Image 2's labels if it has enough visual presence
        # (e.g., at least 4 tiles / 15% of the timeline) to be reasonably seen.
        presence_threshold = tf.cast(frames_used1, tf.float32) * 0.15
        soft_label_mixed = (label1 * weight_1) + (label2 * weight_2)

        # Enforce your structural noise threshold filter
        # If Video 2's visual footprint is under 15%, drop its influence completely
        presence_threshold = 0.15
        has_enough_presence = weight_2 >= presence_threshold

        mixed_label = tf.where(
            has_enough_presence,
            soft_label_mixed,  # Clean soft density target matching visual evidence
            label1,  # Pure base label if Video 2 is just an unreadable flash
        )

        return {"input_image": mixed_image, "input_mask": mask1}, mixed_label

    # Conditionally execute the mix or no_mix subgraph based on probability
    return tf.cond(tf.random.uniform([]) < prob, mix, no_mix)


# def prepare_cutmix_dataset(dataset_original, img_size, prob):
#     # 1. Create a second dataset and shuffle it to mix different images together
#     dataset_shuffled = dataset_original.shuffle(buffer_size=4096)

#     # 2. Zip them together so each element is ((img1, lbl1), (img2, lbl2))
#     zipped_dataset = tf.data.Dataset.zip((dataset_original, dataset_shuffled))

#     # 3. Map the CutMix function (passed via lambda to include parameters)
#     cutmix_dataset = zipped_dataset.map(
#         lambda d1, d2: video_mosaic_cutmix(d1, d2, img_size, prob),
#         num_parallel_calls=tf.data.AUTOTUNE,
#     )

#     return cutmix_dataset


# def video_mosaic_cutmix(
#     data1, data2, img_size, prob, grid_rows=5, grid_cols=5, alpha=0.3
# ):
#     x1, y1 = data1
#     image1 = x1["input_image"]
#     mask1 = x1["input_mask"]
#     label1 = y1["label"]
#     logging.info("Chance of a cut mix on an input is %s and alpha %s", prob, alpha)

#     def no_mix():
#         return {"input_image": image1, "input_mask": mask1}, label1

#     def mix():
#         frames_used1 = y1["num_frames"]

#         x2, y2 = data2
#         image2 = x2["input_image"]

#         label2 = y2["label"]
#         mask2 = x2["input_mask"]
#         frames_used2 = y2["num_frames"]
#         num_frames = tf.math.minimum(frames_used1, frames_used2)
#         # 1. Define dimensions of an individual sub-frame
#         sub_h = img_size // grid_rows
#         sub_w = img_size // grid_cols
#         total_cells = grid_rows * grid_cols

#         # 2. Sample how many sub-frames to replace (from Beta distribution)
#         beta_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
#         lam = beta_dist.sample([])

#         # Convert lambda to a discrete number of blocks to swap (at least 1, at most num_frames-1)
#         # Only swap within the real frames to avoid cutmixing empty/tiled cells
#         num_blocks_to_swap = tf.cast(
#             tf.math.round((1.0 - lam) * tf.cast(num_frames, tf.float32)), tf.int32
#         )
#         num_blocks_to_swap = tf.clip_by_value(num_blocks_to_swap, 1, num_frames - 1)

#         # 3. Create a random binary mask for the grid cells
#         # Only shuffle within the real frame indices to avoid empty cells
#         real_indices = tf.random.shuffle(tf.range(num_frames))
#         swap_indices = real_indices[:num_blocks_to_swap]

#         all_indices = tf.range(total_cells)
#         # Build a 1D boolean mask for the cells
#         cell_mask_1d = tf.reduce_any(
#             tf.equal(tf.expand_dims(all_indices, 0), tf.expand_dims(swap_indices, 1)),
#             axis=0,
#         )

#         # Reshape the 1D mask back into the 2D grid shape (e.g., 2x2)
#         grid_mask = tf.reshape(cell_mask_1d, [grid_rows, grid_cols])

#         # 4. Upsample the grid mask to full image resolution using block repeats
#         mask_expanded = tf.repeat(grid_mask, repeats=sub_h, axis=0)
#         mask_expanded = tf.repeat(mask_expanded, repeats=sub_w, axis=1)

#         # Add a channel dimension and cast to float
#         mask_expanded = tf.cast(mask_expanded[:, :, tf.newaxis], tf.float32)

#         # 5. Blend the two mosaic images using our clean grid-aligned mask
#         mixed_image = image1 * (1.0 - mask_expanded) + image2 * mask_expanded

#         # 6. Compute exact adjusted lambda based on how many cells were swapped
#         actual_swap_ratio = tf.cast(num_blocks_to_swap, tf.float32) / tf.cast(
#             total_cells, tf.float32
#         )
#         adjusted_lam = 1.0 - actual_swap_ratio

#         # Blend the one-hot labels
#         mixed_label = label1 * adjusted_lam + label2 * (1.0 - adjusted_lam)

#         # Globally smooth the time steps to match the soft target label blend
#         mixed_mask = mask1 * adjusted_lam + mask2 * (1.0 - adjusted_lam)
#         # not sure how the mask will work with this
#         return {"input_image": mixed_image, "input_mask": mask1}, mixed_label

#     return tf.cond(tf.random.uniform([]) < prob, mix, no_mix)


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

    include_track = True
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
        augment=False,
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
        current_epoch=tf.Variable(
            0, dtype=tf.int32, trainable=False, name="current_epoch"
        ),
        single_input=False,
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

        # roi = label_batch[-1]
        label_batch = label_batch[0]
    for n, img in enumerate(image_batch):
        # print("Mask is ", masks[n][0])
        # print(roi[n])
        # for row in masks[n]:
        #     for column in row:
        #         print(column)
        # # for i,mask in enumerate(masks[n]):
        # #     print(i, " mask is ", mask)
        # # 1/0
        # continue
        # ,frame_indices[n])
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
def mask_random_frames(rgb_image, frame_indices, record_frames):
    """
    Drops a random number of frames from the end of rgb_image, removing up to
    75% of record_frames so that at least 25% remain.
    Returns the matching frame_indices so the frame mask can be recomputed
    against the surviving frames.
    """
    min_frames = tf.cast(tf.cast(record_frames, tf.float32) * 0.25, tf.int32)

    def no_drop():
        return rgb_image, frame_indices

    def drop_frames():
        max_drop = tf.maximum(record_frames - min_frames, 0)
        num_drop = tf.random.uniform(
            shape=[], minval=1, maxval=max_drop + 1, dtype=tf.int32
        )
        keep = record_frames - num_drop
        return rgb_image[:keep], frame_indices[:keep]

    return tf.cond(min_frames > 0, drop_frames, no_drop)


@tf.function
def get_frame_mask_v2(
    num_valid, frame_indices, centre_x, centre_y, use_velocity, rotation_angle, regions
):
    """
    Normalises frame intervals uniformly against a maximum inter-frame distance of 9 frames.
    """
    # this comes from the random section logic where frames are selected at intervals of 4.32 frames apart
    #  allowing for a possible missed chunk  double this
    MAX_FRAME_DIST = 9
    # max possible centre_x/centre_y displacement between consecutive frames (mosaic tile width)

    indices = tf.cast(frame_indices, tf.float32)
    frame_delta = indices[1:] - indices[:-1]
    normalised_delta = tf.minimum(frame_delta / MAX_FRAME_DIST, 1.0)

    # Reconstruct an absolute timeline by cumulative-summing the
    # backward-aligned per-frame deltas (index i = delta from frame i-1 to
    # i), then normalise to [0, 1] against this track's own max. Done here,
    # against the real per-track frame numbers, rather than later against
    # (possibly jitter-dropped) frames, so the timeline always reflects
    # genuine elapsed time.
    backward_delta = tf.concat([[0.0], normalised_delta], axis=0)
    abs_time = tf.cumsum(backward_delta)
    max_time = tf.maximum(tf.reduce_max(abs_time), 1e-5)
    normalized_abs_time = abs_time / max_time
    time_mask = tf.concat([normalized_abs_time, tf.fill([25 - num_valid], 0.0)], axis=0)

    presence_mask = tf.concat(
        [tf.fill([num_valid], 1.0), tf.fill([25 - num_valid], 0.0)], axis=0
    )
    if use_velocity:
        centre_x = centre_x[:num_valid]
        x_delta = centre_x[1:] - centre_x[:-1]

        centre_y = centre_y[:num_valid]
        y_delta = centre_y[1:] - centre_y[:-1]

        def rotate_velocity():
            c = tf.cos(rotation_angle)
            s = tf.sin(rotation_angle)

            # Standard transformation layout for top-left (0,0) image coordinates
            rot_matrix = tf.stack([tf.stack([c, -s]), tf.stack([s, c])])

            vel_pairs = tf.stack([x_delta, y_delta], axis=-1)
            rotated_vel = tf.matmul(vel_pairs, rot_matrix)  # Removed transpose_b=True
            return rotated_vel[:, 0], rotated_vel[:, 1]

        def no_rotate():
            return x_delta, y_delta

        UNIFIED_MAX_DIST = 6.94

        # Apply spatial transformations natively
        rotated_vel_x, rotated_vel_y = tf.cond(
            tf.not_equal(rotation_angle, 0.0), rotate_velocity, no_rotate
        )

        # Apply isotropic normalisation and clipping bounds
        rotated_vel_x = tf.clip_by_value(rotated_vel_x / UNIFIED_MAX_DIST, -1.0, 1.0)
        rotated_vel_y = tf.clip_by_value(rotated_vel_y / UNIFIED_MAX_DIST, -1.0, 1.0)

        # Synchronise trailing pads to eliminate sequence lag
        padding_len = 25 - num_valid + 1
        x_mask = tf.concat(
            [rotated_vel_x, tf.zeros([padding_len], dtype=tf.float32)], axis=0
        )
        y_mask = tf.concat(
            [rotated_vel_y, tf.zeros([padding_len], dtype=tf.float32)], axis=0
        )

        MAX_WIDTH = 160
        MAX_HEIGHT = 120
        # regions are stored per-frame as [left, top, width, height] (ltwh)
        # against a 160x120 source frame
        valid_regions = tf.cast(regions[:num_valid], tf.float32)
        width_percent = valid_regions[:, 2] / MAX_WIDTH
        height_percent = valid_regions[:, 3] / MAX_HEIGHT

        width_mask = tf.concat(
            [width_percent, tf.zeros([25 - num_valid], dtype=tf.float32)], axis=0
        )
        height_mask = tf.concat(
            [height_percent, tf.zeros([25 - num_valid], dtype=tf.float32)], axis=0
        )

        # time_mask is backward-aligned (index i holds the delta from frame
        # i-1 to i) so it can be cumsum'd into an absolute timeline later.
        # x_mask/y_mask are forward-aligned (index i holds the delta from
        # frame i to i+1). dt_forward_mask reuses the same forward-aligned
        # normalised_delta so it lines up with the velocities at the same
        # index without needing a runtime shift.
        dt_forward_mask = tf.concat(
            [normalised_delta, tf.zeros([padding_len], dtype=tf.float32)], axis=0
        )

        mask = tf.stack(
            [
                time_mask,
                presence_mask,
                width_mask,
                height_mask,
                x_mask,
                y_mask,
                dt_forward_mask,
            ],
            axis=1,
        )
        # mask =  tf.stack(
        #     [
        #         tf.zeros_like(time_mask),
        #         presence_mask,
        #         tf.zeros_like(width_mask),
        #         tf.zeros_like(height_mask),
        #         tf.zeros_like(x_mask),
        #         tf.zeros_like(y_mask),
        #         tf.zeros_like(dt_forward_mask),
        #     ],
        #     axis=1,
        # )
        return mask
    else:
        return tf.stack([time_mask, presence_mask], axis=1)


@tf.function
def get_frame_mask(num_valid, frame_indices):
    """
    Normalises frame intervals uniformly against a maximum inter-frame distance of 9 frames.
    """
    # this comes from the random section logic where frames are selected at intervals of 4.32 frames apart
    #  allowing for a possible missed chunk  double this
    MAX_FRAME_DIST = 9

    indices = tf.cast(frame_indices, tf.float32)
    frame_delta = indices[1:] - indices[:-1]
    normalised_delta = tf.minimum(frame_delta / MAX_FRAME_DIST, 1.0)
    # Use -1.0 as a strict geometric flag for empty padding slots
    mask_flat = tf.concat(
        [[0.0], normalised_delta, tf.fill([25 - num_valid], -1.0)], axis=0
    )

    # mask = tf.reshape(mask_flat, (5, 5, 1))
    return mask_flat


if __name__ == "__main__":
    main()
