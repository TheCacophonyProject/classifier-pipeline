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
            preprocess_fn=preprocess_fn,
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
    preprocess_fn=None,
):

    num_labels = len(labels)
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
    if "human" in r_l:
        weights[r_l.index("human")] = 0
    if "rodent" in r_l:
        weights[r_l.index("rodent")] = 0.04

    # weights[labels.index("cat")] = 0

    for k, v in remapped.items():
        filenames = []
        for label in v:
            filenames.append(tf.io.gfile.glob(f"{base_dir}/{label}-0*.tfrecord"))
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


#
def preprocess(data):
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return tf.keras.applications.inception_v3.preprocess_input(x), y


def read_tfrecord(
    example, image_size, num_labels, labeled, augment, preprocess_fn=None
):
    tfrecord_format = {
        "image/augmented": tf.io.FixedLenFeature((), tf.int64, 0),
        "image/thermalencoded": tf.io.FixedLenFeature((), tf.string),
        "image/filteredencoded": tf.io.FixedLenFeature((), tf.string),
        "image/height": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/width": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/class/label": tf.io.FixedLenFeature((), tf.int64, -1),
    }

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1, fill_mode="nearest", fill_value=0),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.1),
        ]
    )
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
        global remapped_y
        label = remapped_y.lookup(label)
        onehot_label = tf.one_hot(label, num_labels)
        return image, onehot_label
    return image


def decode_image(image, filtered, image_size, augment):
    image = tf.image.decode_png(image, channels=1)
    image = tf.concat((image, image, image), axis=2)

    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, image_size[0], image_size[1])
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
    resampled_ds, remapped = get_resampled(
        f"{config.tracks_folder}/training-data/test",
        1,
        (160, 160),
        labels,
        # distribution=meta["counts"]["test"],
        stop_on_empty_dataset=True,
        augment=False,
        # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
    )
    meta["remapped"] = remapped
    with open(file, "w") as f:
        json.dump(meta, f)

    for e in range(4):
        # for x, y in resampled_ds:
        # for x_2 in x:
        #     print("max is", np.amax(x_2), x_2.shape)
        #     assert np.amax(x_2) == 255
        # show_batch(x, y, labels)
        print("epoch", e)
        true_categories = tf.concat([y for x, y in resampled_ds], axis=0)
        true_categories = np.int64(tf.argmax(true_categories, axis=1))
        c = Counter(list(true_categories))
        print("epoch is size", len(true_categories))
        for i in range(len(labels)):
            print("after have", labels[i], c[i])


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
