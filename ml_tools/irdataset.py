import matplotlib.pyplot as plt

import tensorflow as tf
from functools import partial
import numpy as np
import time
import math
import logging

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
        label = tf.cast(example["image/class/label"], tf.int64)
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

# test crap
def main():
    train_files = tf.io.gfile.glob(
        "/home/gp/cacophony/classifier-data/irvideos/tracks/training-data/validation/*.tfrecord"
    )
    print("got filename", train_files)
    dataset = get_dataset(train_files, 32, (256, 256), 4)
    #
    for e in range(4):
        print("epoch", e)
        true_categories = tf.concat([y for x, y in dataset], axis=0)
        true_categories = np.int64(tf.argmax(true_categories, axis=1))
        c = Counter(list(true_categories))
        print("epoch is size", len(true_categories))
        for i in range(4):
            print("after have", i, c[i])
    # return
    # image_batch, label_batch = next(iter(dataset))
    for e in range(1):
        print("epoch", e)
        for x, y in dataset:
            show_batch(x, y)


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    labels = ["cat", "false-positive", "hedgehog", "possum"]
    # print(label_batch)
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        plt.title(labels[np.argmax(label_batch[n])])
        plt.axis("off")
    plt.show()


if __name__ == "__main__":

    main()
