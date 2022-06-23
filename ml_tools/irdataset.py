import matplotlib.pyplot as plt

import tensorflow as tf
from functools import partial
import numpy as np
import time
import math
import logging
from config.config import Config
import json

seed = 1341
tf.random.set_seed(seed)
np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64


def load_dataset(
    filenames, image_size, num_labels, deterministic=False, labeled=True, augment=False
):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = (
        deterministic  # disable order, increase speed
    )
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(
            read_tfrecord,
            image_size=image_size,
            num_labels=num_labels,
            labeled=labeled,
            augment=augment,
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    return dataset


def get_distribution(dataset):
    true_categories = tf.concat([y for x, y in dataset], axis=0)
    num_labels = len(true_categories[0])
    if len(true_categories) == 0:
        return None
    true_categories = np.int64(tf.argmax(true_categories, axis=1))
    c = Counter(list(true_categories))
    dist = np.empty((num_labels), dtype=np.float32)
    for i in range(num_labels):
        dist[i] = c[i]
    return dist


def get_resampled(
    base_dir,
    batch_size,
    image_size,
    labels,
    reshuffle=True,
    deterministic=False,
    labeled=True,
    resample=True,
    augment=False,
    weights=None,
    stop_on_empty_dataset=True,
    distribution=None,
):

    num_labels = len(labels)
    global remapped
    global remapped_y
    datasets = []

    keys = []
    values = []
    remapped = {}
    remapped_y = {}

    for l in labels:
        remapped[l] = [l]
        keys.append(labels.index(l))
        values.append(labels.index(l))
    if "false-positive" in labels and "insect" in labels:
        remapped["false-positive"].append("insect")
        values[labels.index("insect")] = labels.index("false-positive")

        del remapped["insect"]

    if "possum" in labels and "cat" in labels:
        remapped["possum"].append("cat")
        values[labels.index("cat")] = labels.index("possum")

        del remapped["cat"]
    remapped_y = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values),
        ),
        default_value=tf.constant(-1),
        name="remapped_y",
    )
    weights = [1.0] * len(remapped)
    # if "human" in remapped:
    #     weights[labels.index("human")] = 0
    r_l = list(remapped.keys())
    weights[r_l.index("human")] = 0

    # weights[labels.index("cat")] = 0

    for k, v in remapped.items():
        filenames = []
        for label in v:
            filenames.append(tf.io.gfile.glob(f"{base_dir}/{label}*.tfrecord"))
        dataset = load_dataset(
            filenames,
            image_size,
            num_labels,
            deterministic=deterministic,
            labeled=labeled,
        )
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)

        datasets.append(dataset)
    resampled_ds = tf.data.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=stop_on_empty_dataset
    )
    resampled_ds = resampled_ds.shuffle(2048, reshuffle_each_iteration=reshuffle)
    resampled_ds = resampled_ds.prefetch(buffer_size=AUTOTUNE)
    resampled_ds = resampled_ds.batch(batch_size)
    return resampled_ds


#
def preprocess(data):
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return tf.keras.applications.inception_v3.preprocess_input(x), y


def get_dataset(
    filenames,
    batch_size,
    image_size,
    num_labels,
    reshuffle=True,
    deterministic=False,
    labeled=True,
    resample=True,
    augment=False,
):
    dataset = load_dataset(
        filenames,
        image_size,
        num_labels,
        deterministic=deterministic,
        labeled=labeled,
        augment=augment,
    )

    if resample:
        true_categories = [y for x, y in dataset]
        if len(true_categories) == 0:
            return None
        true_categories = np.int64(tf.argmax(true_categories, axis=1))
        c = Counter(list(true_categories))
        dist = np.empty((num_labels), dtype=np.float32)
        target_dist = np.empty((num_labels), dtype=np.float32)
        for i in range(num_labels):
            dist[i] = c[i]
        logging.info("Count is %s", dist)
        zeros = dist[dist == 0]
        non_zero_labels = num_labels - len(zeros)
        target_dist[:] = 1 / non_zero_labels

        dist_max = np.max(dist)
        dist_min = np.min(dist)
        dist = dist / np.sum(dist)
        # really this is what we want but when the values become too small they never get sampled
        # so need to try reduce the large gaps in distribution
        # can use class weights to adjust more, or just throw out some samples
        max_range = target_dist[0] / 2
        for i in range(num_labels):
            if dist[i] == 0:
                target_dist[i] = 0
            if dist[i] - dist_min > max_range:
                add_on = max_range
                if dist[i] - dist_min > max_range * 2:
                    add_on *= 2

                target_dist[i] += add_on

                dist[i] -= add_on
            elif dist_max - dist[i] > max_range:
                target_dist[i] -= max_range / 2.0

        rej = dataset.rejection_resample(
            class_func=class_func,
            target_dist=target_dist,
            initial_dist=dist,
        )

        dataset = rej.map(lambda extra_label, features_and_label: features_and_label)
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=reshuffle)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def read_tfrecord(example, image_size, num_labels, labeled, augment):
    tfrecord_format = {
        "image/thermalencoded": tf.io.FixedLenFeature((), tf.string),
        "image/filteredencoded": tf.io.FixedLenFeature((), tf.string),
        "image/height": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/width": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/class/label": tf.io.FixedLenFeature((), tf.int64, -1),
    }

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(
        example["image/thermalencoded"],
        example["image/filteredencoded"],
        image_size,
        augment,
    )

    if labeled:
        label = tf.cast(example["image/class/label"], tf.int32)
        global remapped_y
        label = remapped_y.lookup(label)
        onehot_label = tf.one_hot(label, num_labels)
        return image, onehot_label
    return image


def decode_image(image, filtered, image_size, augment):
    image = tf.image.decode_jpeg(image, channels=1)
    filtered = tf.image.decode_jpeg(filtered, channels=1)
    if augment:
        image = tf.image.random_brightness(image, 0.2)
        rand = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
        if tf.math.greater_equal(rand, 0.5):
            image = tf.image.flip_left_right(image)
            filtered = tf.image.flip_left_right(filtered)
    image = tf.concat((image, image, filtered), axis=2)
    if augment:
        image = tf.image.random_contrast(image, 0, 1, seed=None)

    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, image_size[0], image_size[1])
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image


def class_func(features, label):
    label = tf.argmax(label)
    return label


from collections import Counter


def main():
    config = Config.load_from_file()

    file = f"{config.tracks_folder}/training-meta.json"
    with open(file, "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])
    datasets = []
    # weights = [0.5] * len(labels)
    resampled_ds = get_resampled(
        f"{config.tracks_folder}/training-data/test",
        32,
        (160, 160),
        labels,
        # distribution=meta["counts"]["test"],
        stop_on_empty_dataset=True,
    )
    global remapped
    meta["remapped"] = remapped
    with open(file, "w") as f:
        json.dump(meta, f)
    for e in range(4):
        print("epoch", e)
        true_categories = tf.concat([y for x, y in resampled_ds], axis=0)
        true_categories = np.int64(tf.argmax(true_categories, axis=1))
        c = Counter(list(true_categories))
        print("epoch is size", len(true_categories))
        for i in range(len(labels)):
            print("after have", labels[i], c[i])


if __name__ == "__main__":

    main()
