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

# seed = 1341
# tf.random.set_seed(seed)
# np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64

insect = None
fp = None


def get_excluded():
    return [
        "goat",
        "lizard",
        "not identifiable",
        "other",
        "pest",
        "pig",
        "sealion",
    ]


def get_remapped():
    return {
        "allbirds": "bird",
        "black swan": "bird",
        "brown quail": "bird",
        "california quail": "bird",
        "duck": "bird",
        "insect": "false-positive",
        "pheasant": "bird",
        "pukeko": "bird",
        "quail": "bird",
    }


def load_dataset(filenames, remap_lookup, num_labels, args):
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

    dataset = dataset.map(
        partial(
            read_tfrecord,
            image_size=image_size,
            remap_lookup=remap_lookup,
            num_labels=num_labels,
            labeled=labeled,
            augment=augment,
            preprocess_fn=preprocess_fn,
            include_features=include_features,
            only_features=only_features,
            one_hot=one_hot,
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    if only_features:
        filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x))
    else:
        filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x[0]))

    dataset = dataset.filter(filter_nan)

    filter_excluded = lambda x, y: not tf.math.equal(tf.math.count_nonzero(y), 0)
    dataset = dataset.filter(filter_excluded)
    return dataset


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
):
    logging.info(
        "Read tf record with image %s lbls %s labeld %s aug  %s  prepr %s only features %s one hot %s include fetures %s",
        image_size,
        num_labels,
        labeled,
        augment,
        preprocess_fn,
        only_features,
        one_hot,
        include_features,
    )
    load_images = not only_features
    tfrecord_format = {
        "image/class/label": tf.io.FixedLenFeature((), tf.int64, -1),
    }
    if load_images:
        tfrecord_format["image/thermalencoded"] = tf.io.FixedLenFeature(
            [25 * 32 * 32], dtype=tf.float32
        )

        tfrecord_format["image/filteredencoded"] = tf.io.FixedLenFeature(
            [25 * 32 * 32], dtype=tf.float32
        )

    if include_features or only_features:
        tfrecord_format["image/features"] = tf.io.FixedLenFeature(
            [36 * 5 + 8], dtype=tf.float32
        )
    example = tf.io.parse_single_example(example, tfrecord_format)
    if load_images:
        thermalencoded = example["image/thermalencoded"]
        filteredencoded = example["image/filteredencoded"]

        thermals = tf.reshape(thermalencoded, [25, 32, 32, 1])
        filtered = tf.reshape(filteredencoded, [25, 32, 32, 1])
        rgb_images = tf.concat((thermals, thermals, filtered), axis=3)
        rotation_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(0.1, fill_mode="nearest", fill_value=0),
            ]
        )
        # rotation augmentation before tiling
        if augment:
            logging.info("Augmenting")
            rgb_images = rotation_augmentation(rgb_images)
        rgb_images = tf.ensure_shape(rgb_images, (25, 32, 32, 3))
        image = tile_images(rgb_images)

        if augment:
            data_augmentation = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomBrightness(
                        0.2
                    ),  # better per frame or per sequence??
                    tf.keras.layers.RandomContrast(0.5),
                ]
            )
            image = data_augmentation(image)
        if preprocess_fn is not None:
            logging.info(
                "Preprocessing with %s.%s",
                preprocess_fn.__module__,
                preprocess_fn.__name__,
            )
            image = preprocess_fn(image)
    if labeled:
        label = tf.cast(example["image/class/label"], tf.int32)
        label = remap_lookup.lookup(label)

        if one_hot:
            if tf.math.equal(label, -1):
                label = tf.zeros(num_labels)
            else:
                label = tf.one_hot(label, num_labels)
        if include_features or only_features:
            features = example["image/features"]
            if only_features:
                return features, label
            return (image, features), label
        return image, label
    if only_features:
        return example["image/features"]
    elif include_features:
        return (image, example["image/features"])
    return image


def decode_image(thermals, filtereds, image_size):
    deoced_thermals = []
    decoded_filtered = []
    for thermal, filtered in zip(thermals, filtereds):
        image = tf.image.decode_png(image, channels=1)
        filtered = tf.image.decode_png(filtered, channels=1)
        decoded_thermal.append(image)
        decoded_filtered.append(filtered)
    return decoded_thermal, decoded_filtered


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
    config = Config.load_from_file()
    from .tfdataset import get_dataset, get_distribution

    # file = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/training-meta.json"
    file = f"{config.tracks_folder}/training-meta.json"
    with open(file, "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])
    datasets = []

    resampled_ds, remapped, labels, epoch_size = get_dataset(
        # dir,
        load_dataset,
        f"{config.tracks_folder}/training-data/train",
        labels,
        batch_size=32,
        image_size=(160, 160),
        # augment=True,
        # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
        resample=False,
        include_features=False,
        remapped_labels=get_remapped(),
        excluded_labels=get_excluded(),
    )
    print("Ecpoh size is", epoch_size)
    print(get_distribution(resampled_ds, len(labels)))
    return
    #
    for e in range(2):
        print("epoch", e)
        true_categories = [y for x, y in resampled_ds]
        true_categories = tf.concat(true_categories, axis=0)
        true_categories = np.int64(tf.argmax(true_categories, axis=1))
        c = Counter(list(true_categories))
        print("epoch is size", len(true_categories))
        for i in range(len(labels)):
            print("after have", labels[i], c[i])
        for x, y in resampled_ds:
            show_batch(x, y, labels)

    # return


def show_batch(image_batch, label_batch, labels):
    image_batch = image_batch[0]
    plt.figure(figsize=(10, 10))
    print("images in batch", len(image_batch), len(label_batch))
    num_images = min(len(image_batch), 25)
    for n in range(num_images):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(np.uint8(image_batch[n]))
        plt.title("C-" + str(image_batch[n]))
        plt.title(labels[np.argmax(label_batch[n])])
        plt.axis("off")
    # return
    plt.show()


if __name__ == "__main__":
    main()
