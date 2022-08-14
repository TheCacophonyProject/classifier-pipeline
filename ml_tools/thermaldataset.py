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

# seed = 1341
# tf.random.set_seed(seed)
# np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64

insect = None
fp = None


def load_dataset(
    filenames,
    image_size,
    num_labels,
    deterministic=False,
    labeled=True,
    augment=False,
    preprocess_fn=None,
):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = (
        deterministic  # disable order, increase speed
    )

    dataset = tf.data.TFRecordDataset(filenames)
    # dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4)
    # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.map(
        partial(
            read_tfrecord,
            image_size=image_size,
            num_labels=num_labels,
            labeled=labeled,
            augment=augment,
            preprocess_fn=preprocess_fn,
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    # dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    # dataset = dataset.map(tf.keras.applications.inception_v3.preprocess_input)
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


#
def preprocess(data):
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return tf.keras.applications.inception_v3.preprocess_input(x), y


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
    augment=False,
    resample=True,
    preprocess_fn=None,
):
    num_labels = len(labels)
    global remapped_y
    remapped = {}
    keys = []
    values = []
    for l in labels:
        remapped[l] = [l]
        keys.append(labels.index(l))
        values.append(labels.index(l))
    if "false-positive" in labels and "insect" in labels:
        remapped["false-positive"].append("insect")
        values[labels.index("insect")] = labels.index("false-positive")
        del remapped["insect"]
    remapped_y = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values),
        ),
        default_value=tf.constant(-1),
        name="remapped_y",
    )
    filenames = tf.io.gfile.glob(f"{base_dir}/*.tfrecord")
    dataset = load_dataset(
        filenames,
        image_size,
        num_labels,
        deterministic=deterministic,
        labeled=labeled,
        augment=augment,
        preprocess_fn=preprocess_fn,
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
        print("Count is", dist)
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
            # if dist[i] - dist_min > max_range:
            #     add_on = max_range
            #     if dist[i] - dist_min > max_range * 2:
            #         add_on *= 2
            #
            #     target_dist[i] += add_on
            #
            #     dist[i] -= add_on
            elif dist_max - dist[i] > max_range:
                target_dist[i] -= max_range / 2.0
            target_dist[i] = max(0, target_dist[i])
        target_dist = target_dist / np.sum(target_dist)
        rej = dataset.rejection_resample(
            class_func=class_func,
            target_dist=target_dist,
            # initial_dist=dist,
        )
        dataset = rej.map(lambda extra_label, features_and_label: features_and_label)

    dataset = dataset.shuffle(2048, reshuffle_each_iteration=reshuffle)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset, remapped


def get_resampled_by_label(
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
    preprocess_fn=None,
):
    num_labels = len(labels)
    global remapped_y
    remapped = {}
    keys = []
    values = []
    for l in labels:
        remapped[l] = [l]
        keys.append(labels.index(l))
        values.append(labels.index(l))
    if "false-positive" in labels and "insect" in labels:
        remapped["false-positive"].append("insect")
        values[labels.index("insect")] = labels.index("false-positive")
        del remapped["insect"]
    remapped_y = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values),
        ),
        default_value=tf.constant(-1),
        name="remapped_y",
    )
    weights = [1.0] * len(remapped)
    datasets = []

    for k, v in remapped.items():
        filenames = []
        for label in v:
            safe_l = label.replace("/", "-")

            filenames.append(tf.io.gfile.glob(f"{base_dir}/{safe_l}-0*.tfrecord"))
        dataset = load_dataset(
            filenames,
            image_size,
            num_labels,
            deterministic=deterministic,
            labeled=labeled,
            augment=augment,
            preprocess_fn=preprocess_fn,
        )
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)

        datasets.append(dataset)
    resampled_ds = tf.data.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=stop_on_empty_dataset
    )
    resampled_ds = resampled_ds.shuffle(2048, reshuffle_each_iteration=reshuffle)
    resampled_ds = resampled_ds.prefetch(buffer_size=AUTOTUNE)
    resampled_ds = resampled_ds.batch(batch_size)
    return resampled_ds, remapped


def read_tfrecord(
    example, image_size, num_labels, labeled, augment=False, preprocess_fn=None
):
    tfrecord_format = {
        "image/thermalencoded": tf.io.FixedLenFeature([], tf.string),
        "image/filteredencoded": tf.io.FixedLenFeature([], tf.string),
        "image/class/label": tf.io.FixedLenFeature((), tf.int64, -1),
        # "image/clip_id": tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example, tfrecord_format)
    thermalencoded = tf.cast(example["image/thermalencoded"], tf.string)
    image = decode_image(
        example["image/thermalencoded"],
        example["image/filteredencoded"],
        image_size,
    )
    # image = decode_image image_size)
    # source_id = tf.cast(example["image/clip_id"], tf.int64)
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.1),
        ]
    )
    if augment:
        logging.info("Augmenting")
        image = data_augmentation(image)
    if preprocess_fn is not None:
        logging.info(
            "Preprocessing with %s.%s", preprocess_fn.__module__, preprocess_fn.__name__
        )
        image = preprocess_fn(image)

    if labeled:
        label = tf.cast(example["image/class/label"], tf.int32)
        global remapped_y
        label = remapped_y.lookup(label)
        onehot_label = tf.one_hot(label, num_labels)
        return image, onehot_label
    return image


def decode_image(image, filtered, image_size):
    image = tf.image.decode_png(image, channels=1)
    filtered = tf.image.decode_png(filtered, channels=1)
    image = tf.concat((image, image, filtered), axis=2)
    image = tf.cast(image, tf.float32)
    return image


def class_func(features, label):
    label = tf.argmax(label)
    return label


from collections import Counter

# test stuff
def main():
    init_logging()
    config = Config.load_from_file()
    # file = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/training-meta.json"
    file = f"{config.tracks_folder}/training-meta.json"
    with open(file, "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])
    by_label = meta.get("by_label", True)
    datasets = []
    # dir = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/validation"
    # weights = [0.5] * len(labels)
    if by_label:
        resampled_ds, remapped = get_resampled(
            # dir,
            f"{config.tracks_folder}/training-data/validation",
            32,
            (160, 160),
            labels,
            augment=False,
            stop_on_empty_dataset=False,
            preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
        )
    else:
        resampled_ds, remapped = get_resampled(
            # dir,
            f"{config.tracks_folder}/training-data/validation",
            32,
            (160, 160),
            labels,
            augment=False,
            preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
        )
    # print(get_distribution(resampled_ds))
    #
    for e in range(2):
        print("epoch", e)
        true_categories = tf.concat([y for x, y in resampled_ds], axis=0)
        true_categories = np.int64(tf.argmax(true_categories, axis=1))
        c = Counter(list(true_categories))
        print("epoch is size", len(true_categories))
        for i in range(len(labels)):
            print("after have", labels[i], c[i])

    # return
    for e in range(1):
        for x, y in resampled_ds:
            print("max is", np.amax(x), np.amin(x))
            return
            # show_batch(x, y, labels)


def show_batch(image_batch, label_batch, labels):
    plt.figure(figsize=(10, 10))
    print("images in batch", len(image_batch))
    num_images = min(len(image_batch), 25)
    for n in range(num_images):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(np.uint8(image_batch[n]))
        plt.title("C-" + str(label_batch[n]))
        plt.title(labels[np.argmax(label_batch[n])])
        plt.axis("off")
    plt.show()


if __name__ == "__main__":

    main()
