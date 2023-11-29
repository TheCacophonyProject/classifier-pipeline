import matplotlib.pyplot as plt

import tensorflow as tf
from functools import partial
import numpy as np
import time
import math
import logging
from config.config import Config
import json

import sys
from ml_tools.logs import init_logging

AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64


def get_excluded():
    return ["human", "dog", "nothing", "other", "sheep", "unknown", "rodent"]


def get_remapped():
    return {"insect": "false-positive", "cat": "possum"}
    # return ["insect", "cat"]


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
    one_hot = args.get("one_hot", True)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.map(
        partial(
            read_irrecord,
            remap_lookup=remap_lookup,
            num_labels=len(labels),
            image_size=image_size,
            labeled=labeled,
            augment=augment,
            preprocess_fn=preprocess_fn,
            one_hot=one_hot,
            include_track=args.get("include_track", False),
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    if args.get("include_track", False):
        filter_excluded = lambda x, y: not tf.math.equal(tf.math.count_nonzero(y[0]), 0)
    else:
        filter_excluded = lambda x, y: not tf.math.equal(tf.math.count_nonzero(y), 0)

    dataset = dataset.filter(filter_excluded)

    return dataset


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1, fill_mode="nearest", fill_value=0),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.1),
    ]
)


def read_irrecord(
    example,
    image_size,
    remap_lookup,
    num_labels,
    labeled,
    augment,
    preprocess_fn=None,
    one_hot=True,
    include_track=False,
):
    tfrecord_format = {
        "image/augmented": tf.io.FixedLenFeature((), tf.int64, 0),
        "image/thermalencoded": tf.io.FixedLenFeature((), tf.string),
        "image/filteredencoded": tf.io.FixedLenFeature((), tf.string),
        "image/height": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/width": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/class/label": tf.io.FixedLenFeature((), tf.int64, -1),
    }
    if include_track:
        tfrecord_format["image/track_id"] = tf.io.FixedLenFeature((), tf.int64, -1)
        tfrecord_format["image/avg_mass"] = tf.io.FixedLenFeature((), tf.int64, -1)

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(
        example["image/thermalencoded"],
        example["image/filteredencoded"],
        image_size,
        augment,
    )
    augmented = tf.cast(example["image/augmented"], tf.bool)

    if not augmented and augment:
        image = data_augmentation(image)
    if preprocess_fn is not None:
        image = preprocess_fn(image)

    if labeled:
        label = tf.cast(example["image/class/label"], tf.int32)
        label = remap_lookup.lookup(label)
        if one_hot:
            label = tf.one_hot(label, num_labels)
        if include_track:
            track_id = tf.cast(example["image/track_id"], tf.int32)
            avg_mass = tf.cast(example["image/avg_mass"], tf.int32)
            label = (label, track_id, avg_mass)
        return image, label
    return image


def decode_image(image, filtered, image_size, augment):
    image = tf.image.decode_png(image, channels=1)
    image = tf.concat((image, image), axis=2)

    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, image_size[0], image_size[1])
    return image


def main():
    init_logging()

    from .tfdataset import get_dataset, get_distribution

    config = Config.load_from_file("classifier-ir.yaml")

    file = f"{config.base_folder}/training-data/training-meta.json"
    with open(file, "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])
    datasets = []
    # weights = [0.5] * len(labels)
    resampled_ds, remapped, labels, _ = get_dataset(
        load_dataset,
        f"{config.base_folder}/training-data/test",
        labels,
        batch_size=1,
        image_size=(160, 160),
        augment=False,
        excluded_labels=get_excluded(),
        remapped_labels=get_remapped(),
        deterministic=True,
        shuffle=False,
    )
    meta["remapped"] = remapped
    with open(file, "w") as f:
        json.dump(meta, f)
    from collections import Counter
    import cv2

    i = 0
    for e in range(2):
        # for batch_x, batch_y in resampled_ds:
        #     for x, y in zip(batch_x, batch_y):
        #         lbl = np.argmax(y)
        #         lbl = labels[lbl]
        #         print("X is", x.shape, lbl)
        #         cv2.imwrite(f"./images/{lbl}-{i}.png", x.numpy())
        #         # 1 / 0
        #         i += 1
        # # for x_2 in x:
        # #     print("max is", np.amax(x_2), x_2.shape)
        # #     assert np.amax(x_2) == 255
        # # show_batch(x, y, labels)
        # return
        print("epoch", e)
        dist = get_distribution(resampled_ds, len(labels), extra_meta=False)
        #
        # true_categories = tf.concat([y for x, y in resampled_ds], axis=0)
        # true_categories = np.int64(tf.argmax(true_categories, axis=1))
        # c = Counter(list(true_categories))
        # print("epoch is size", len(true_categories))
        for l, d in zip(labels, dist):
            print("after have", l, d)


def show_batch(image_batch, label_batch, labels):
    plt.figure(figsize=(10, 10))
    print("images in batch", len(image_batch))
    num_images = min(len(image_batch), 25)
    for n in range(num_images):
        print("imageas max is", np.amax(image_batch[n]), np.amin(image_batch[n]))
        # print("image bach", image_batch[n])
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(np.uint8(image_batch[n]))
        plt.title(labels[np.argmax(label_batch[n])])
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
