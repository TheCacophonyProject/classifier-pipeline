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

IMG_SIZE = 45


# labels can be any subset of this, prevents new labels being trained on until we explicitly add them to here
def get_acceptable_labels():
    logging.warning("Need to add remapped labels into acceptable labels")
    return None
    return [
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
    ]


def get_excluded():
    return [
        "animal",
        "goat",
        "lizard",
        "not identifiable",
        "other",
        "pest",
        "pig",
        "sealion",
        "bat",
        "mammal",
        "frog",
        "cow",
        "static",
        # added gp forretrain
        "wombat",
        "bandicoot",
        "horse",
        "otter",
        # "gray kangaroo",
        # "echidna",
        # "fox",
        # "deer",
        # "sheep",
        # "wombat",
    ]


def get_remapped(multi_label=False):
    land_bird = "land-bird" if multi_label else "bird"
    return {
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
        "chicken": land_bird,
    }


def get_extra_mappings(labels):
    land_birds = ["land-bird"]
    if "bird" not in labels:
        return None
    bird_index = labels.index("bird")
    values = []
    keys = []
    for l in land_birds:
        if l in labels:
            l_i = labels.index(l)
            keys.append(l_i)
            values.append(bird_index)
    extra_label_map = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values),
        ),
        default_value=tf.constant(-1),
        name="extra_label_map",
    )
    logging.info("Extra label mapping is %s to %s ", keys, values)
    return extra_label_map


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
    labeled = args.get("labeled", True)
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
    dataset = dataset.map(
        partial(
            read_tfrecord,
            image_size=image_size,
            remap_lookup=remap_lookup,
            num_labels=len(labels),
            labeled=labeled,
            augment=augment,
            preprocess_fn=preprocess_fn,
            include_features=include_features,
            only_features=only_features,
            one_hot=one_hot,
            extra_label_map=extra_label_map,
            include_track=args.get("include_track", False),
            num_frames=args.get("num_frames", 25),
            channels=args.get(
                "channels", [TrackChannels.thermal.name, TrackChannels.filtered.name]
            ),
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    if only_features:
        filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x))
    else:
        filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x[0]))

    dataset = dataset.filter(filter_nan)

    # if features are missing they wil be 0 size
    if args.get("only_features"):
        filter_none = lambda x, y: tf.size(x) > 0
        dataset = dataset.filter(filter_none)
    elif args.get("include_features"):
        filter_none = lambda x, y: tf.size(x[1]) > 0
        dataset = dataset.filter(filter_none)
    return dataset


rotation_augmentation = tf.keras.Sequential(
    [
        # Tested at 0.5 and 0.1 seems to work best
        tf.keras.layers.RandomRotation(0.1, fill_mode="nearest", fill_value=0),
    ]
)
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomBrightness(0.2),  # better per frame or per sequence??
        tf.keras.layers.RandomContrast(0.5),
    ]
)


def read_tfrecord(
    example,
    image_size,
    remap_lookup,
    num_labels,
    labeled,
    augment=False,
    preprocess_fn=None,
    only_features=False,
    one_hot=True,
    include_features=False,
    extra_label_map=None,
    include_track=False,
    num_frames=25,
    channels=[TrackChannels.thermal.name, TrackChannels.filtered.name],
):
    logging.info(
        "Read tf record with image %s lbls %s labeld %s aug  %s  prepr %s only features %s one hot %s include fetures %s num frames %s",
        image_size,
        num_labels,
        labeled,
        augment,
        preprocess_fn,
        only_features,
        one_hot,
        include_features,
        num_frames,
    )
    load_images = not only_features
    tfrecord_format = {
        "image/class/label": tf.io.FixedLenFeature((), tf.int64, -1),
    }
    if load_images:
        if TrackChannels.filtered.name in channels:
            tfrecord_format["image/filteredencoded"] = tf.io.FixedLenSequenceFeature(
                [num_frames * IMG_SIZE * IMG_SIZE], dtype=tf.float32, allow_missing=True
            )
        if TrackChannels.thermal.name in channels:
            tfrecord_format["image/thermalencoded"] = tf.io.FixedLenSequenceFeature(
                [num_frames * IMG_SIZE * IMG_SIZE], dtype=tf.float32, allow_missing=True
            )

    if include_track:
        tfrecord_format["image/track_id"] = tf.io.FixedLenFeature((), tf.int64, -1)
        tfrecord_format["image/avg_mass"] = tf.io.FixedLenFeature((), tf.int64, -1)
    if include_features or only_features:
        tfrecord_format["image/features"] = tf.io.FixedLenSequenceFeature(
            [36 * 5 + 8], dtype=tf.float32, allow_missing=True
        )
    example = tf.io.parse_single_example(example, tfrecord_format)
    if load_images:
        if TrackChannels.thermal.name in channels:
            thermalencoded = example["image/thermalencoded"]
            thermals = tf.reshape(thermalencoded, [num_frames, IMG_SIZE, IMG_SIZE, 1])
        if TrackChannels.filtered.name in channels:
            filteredencoded = example["image/filteredencoded"]
            filtered = tf.reshape(filteredencoded, [num_frames, IMG_SIZE, IMG_SIZE, 1])
        rgb_image = None
        for type in channels:
            if type == TrackChannels.thermal.name:
                image = thermals
            elif type == TrackChannels.filtered.name:
                image = filtered
            if rgb_image is None:
                rgb_image = image
            else:
                rgb_image = tf.concat((rgb_image, image), axis=3)
        # rotation augmentation before tiling
        if augment:
            logging.info("Augmenting")
            rgb_image = rotation_augmentation(rgb_image)
            random_value = tf.random.uniform(
                shape=[], minval=0.0, maxval=1.0, dtype=tf.float32
            )
            if tf.greater(random_value, 0.5):
                rgb_image = tf.image.flip_left_right(rgb_image)
        rgb_image = tf.ensure_shape(
            rgb_image, (num_frames, IMG_SIZE, IMG_SIZE, len(channels))
        )
        rgb_image = tf.image.crop_to_bounding_box(rgb_image, 7, 7, 32, 32)

        if num_frames > 1:
            rgb_image = tile_images(rgb_image)

        if augment:
            rgb_image = data_augmentation(rgb_image)
        if num_frames == 1:
            # remove the leading axis
            rgb_image = tf.squeeze(rgb_image)

        if preprocess_fn is not None:
            logging.info(
                "Preprocessing with %s.%s",
                preprocess_fn.__module__,
                preprocess_fn.__name__,
            )
            rgb_image = preprocess_fn(rgb_image)
    if labeled:
        label = tf.cast(example["image/class/label"], tf.int32)
        label = remap_lookup.lookup(label)
        if extra_label_map is not None:
            extra = extra_label_map.lookup(label)
            label = tf.stack([label, extra], axis=0)
        if one_hot:
            label = tf.one_hot(label, num_labels)
            if extra_label_map is not None:
                label = tf.reduce_max(label, axis=0)
        if include_track:

            track_id = tf.cast(example["image/track_id"], tf.int32)
            avg_mass = tf.cast(example["image/avg_mass"], tf.int32)
            label = (label, track_id, avg_mass)
        if include_features or only_features:
            features = tf.squeeze(example["image/features"])
            if only_features:
                return features, label
            return (rgb_image, features), label
        return rgb_image, label
    if only_features:
        return tf.squeeze(example["image/features"])
    elif include_features:
        return (rgb_image, tf.squeeze(example["image/features"]))
    return rgb_image


def tile_images(images):
    index = 0
    image = None
    for x in range(5):
        t_row = tf.concat(tf.unstack(images[index : index + 5]), axis=1)
        if image is None:
            image = t_row
        else:
            image = tf.concat([image, t_row], 0)

        index += 5
    return image


from collections import Counter


# test stuff
def main():
    init_logging()
    logging.info("Loading %s", "classifier.yaml")
    config = Config.load_from_file("classifier.yaml")
    from .tfdataset import get_dataset, get_distribution

    # file = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/training-meta.json"
    training_folder = Path(config.base_folder) / "training-data"
    meta_f = training_folder / "training-meta.json"
    with open(meta_f, "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])
    datasets = []
    excluded_labels = get_excluded()
    # for l in labels:
    #     if l not in ["mustelid", "deer", "sheep"]:
    #         excluded_labels.append(l)

    resampled_ds, remapped, labels, epoch_size = get_dataset(
        # dir,
        load_dataset,
        training_folder / "test",
        labels,
        batch_size=32,
        image_size=(160, 160),
        augment=True,
        # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
        resample=False,
        shuffle=False,
        include_features=False,
        remapped_labels=get_remapped(),
        excluded_labels=excluded_labels,
        include_track=True,
        num_frames=25,
        deterministic=True,
    )
    print("Ecpoh size is", epoch_size)
    # print(get_distribution(resampled_ds, len(labels), extra_meta=False))
    # return
    #
    save_dir = Path("./test-images")
    save_dir.mkdir(exist_ok=True)
    for e in range(1):
        batch_i = 0
        print("epoch", e)
        for x, y in resampled_ds:
            save_batch(x, y, labels, save_dir, tracks=True)
            # show_batch(x, y, labels, save=save_dir / f"{batch_i}.jpg", tracks=True)
            batch_i += 1
    # return


save_index = 0


def save_batch(image_batch, label_batch, labels, save_dir, tracks=False):
    global save_index
    if tracks:
        track_batch = label_batch[1]
        label_batch = label_batch[0]
    for n, img in enumerate(image_batch):
        img = np.uint8(img)
        file_title = (
            f"{labels[np.argmax(label_batch[n])]}-{track_batch[n]}-{save_index}.png"
        )
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


if __name__ == "__main__":
    main()
