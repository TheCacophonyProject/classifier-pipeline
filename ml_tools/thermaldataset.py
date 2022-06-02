import math
import matplotlib.pyplot as plt

import tensorflow as tf
from functools import partial
import numpy as np
import time
from config.config import Config
import json

# seed = 1341
# tf.random.set_seed(seed)
# np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64

insect = None
fp = None


def load_dataset(
    filenames, image_size, num_labels, deterministic=False, labeled=True, augment=False
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


def get_dataset(
    filenames,
    batch_size,
    image_size,
    labels,
    reshuffle=True,
    deterministic=False,
    labeled=True,
    resample=True,
    augment=False,
):
    num_labels = len(labels)
    global insect
    global fp
    if "false-positive" in labels:
        fp = tf.constant(labels.index("false-positive"), tf.int64)
    if "insect" in labels:
        insect = tf.constant(labels.index("insect"), tf.int64)
    dataset = load_dataset(
        filenames, image_size, num_labels, deterministic=deterministic, labeled=labeled
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


def read_tfrecord(example, image_size, num_labels, labeled, augment=False):
    tfrecord_format = {
        "image/thermalencoded": tf.io.FixedLenFeature([], tf.string),
        "image/filteredencoded": tf.io.FixedLenFeature([], tf.string),
        "image/class/label": tf.io.FixedLenFeature((), tf.int64, -1),
    }

    example = tf.io.parse_single_example(example, tfrecord_format)
    thermalencoded = tf.cast(example["image/thermalencoded"], tf.string)
    image = decode_image(
        example["image/thermalencoded"],
        example["image/filteredencoded"],
        image_size,
        augment=augment,
    )
    # image = decode_image image_size)

    if labeled:
        # label = tf.cast(
        #     tf.reshape(example["image/object/class/label"], shape=[]), dtype=tf.int64
        # )
        label = tf.cast(example["image/class/label"], tf.int64)
        global insect
        global fp
        if insect is not None and fp is not None:
            if tf.math.equal(label, insect):
                label = fp
        onehot_label = tf.one_hot(label, num_labels)

        return image, onehot_label
    return image


def decode_image(image, filtered, image_size, augment=True):
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

# test stuff
def main():
    config = Config.load_from_file()

    train_files = tf.io.gfile.glob(
        f"{config.tracks_folder}/training-data/train/*.tfrecord"
    )
    print("got filename", train_files)
    file = f"{config.tracks_folder}/training-meta.json"
    with open(file, "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])
    dataset = get_dataset(train_files, 32, (160, 160), labels, augment=False)
    #
    # print("labels are", labels)
    # for e in range(4):
    #     print("epoch", e)
    #     true_categories = tf.concat([y for x, y in dataset], axis=0)
    #     true_categories = np.int64(tf.argmax(true_categories, axis=1))
    #     c = Counter(list(true_categories))
    #     print("epoch is size", len(true_categories))
    #     for i in range(len(labels)):
    #         print("after have", i, c[i])
    # return
    image_batch, label_batch = next(iter(dataset))
    for e in range(4):
        print("epoch", e)
        batch = 0
        for x, y in dataset:
            batch += 1
            true_categories = np.int64(tf.argmax(y, axis=1))
            c = Counter(true_categories)
            for i in range(len(labels)):
                print(batch, "batch have", labels[i], c[i])
            # show_batch(x, y, labels)


def show_batch(image_batch, label_batch, labels):
    plt.figure(figsize=(10, 10))
    print("images in batch", len(image_batch))
    num_images = min(len(image_batch), 25)
    for n in range(num_images):
        print("imageas max is", np.amax(image_batch[n]), np.amin(image_batch[n]))
        print("image bach", image_batch[n])
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        plt.title(labels[np.argmax(label_batch[n])])
        plt.axis("off")
    plt.show()


if __name__ == "__main__":

    main()
