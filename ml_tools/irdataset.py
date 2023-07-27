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


def get_weighting(dataset, labels, min_weigth=0.25, max_weight=4):
    excluded_labels = ["sheep"]

    dont_weight = ["mustelid", "wallaby", "human", "penguin"]
    dist = get_distribution(dataset, labels)
    zeros = dist[dist == 0]
    non_zero_labels = num_labels - len(zeros)

    total = np.sum(dist)
    weights = {}
    for i in range(num_labels):
        if labels[i] in dont_weight:
            weights[i] = 1
        if dist[i] == 0:
            weights[i] = 0
        else:
            weights[i] = (1 / dist[i]) * (total / non_zero_labels)
            # cap the weights
            weights[i] = min(weights[i], 4)
            weights[i] = max(weights[i], 0.25)
        logging.info("weights for %s is %s", labels[i], weights[i])
    return weights


def get_distribution(dataset, num_labels, batched=True):
    true_categories = [y for x, y in dataset]
    dist = np.zeros((num_labels), dtype=np.float32)
    if len(true_categories) == 0:
        return dist
    if batched:
        true_categories = tf.concat(true_categories, axis=0)
    if len(true_categories) == 0:
        return dist
    classes = []
    for y in true_categories:
        non_zero = tf.where(y).numpy()
        classes.extend(non_zero.flatten())
    classes = np.array(classes)

    c = Counter(list(classes))
    for i in range(num_labels):
        dist[i] = c[i]
    return dist


def load_dataset(filenames, num_labels, args):
    deterministic = args.get("deterministic", False)

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = (
        deterministic  # disable order, increase speed
    )
    dataset = tf.data.TFRecordDataset(filenames)
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
            read_tfrecord,
            num_labels=num_labels,
            image_size=image_size,
            labeled=labeled,
            augment=augment,
            preprocess_fn=preprocess_fn,
            one_hot=one_hot,
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    return dataset


def get_excluded_labels():
    return ["insect", "cat"]


def get_dataset(base_dir, labels, **args):
    excluded_labels = args.get("excluded_labels", [])
    global remapped_y
    remapped = {}
    keys = []
    values = []
    excluded_labels.append("insect")
    excluded_labels.append("cat")
    new_labels = labels.copy()
    for excluded in excluded_labels:
        if excluded in labels:
            new_labels.remove(excluded)
    for l in labels:
        keys.append(labels.index(l))
        if l in excluded_labels:
            remapped[l] = -1
            values.append(-1)
            logging.info("Excluding %s", l)
        else:
            remapped[l] = [l]
            values.append(new_labels.index(l))

    if "false-positive" in labels and "insect" in labels:
        remapped["false-positive"].append("insect")
        values[labels.index("insect")] = new_labels.index("false-positive")
        del remapped["insect"]

    if "possum" in labels and "cat" in labels:
        remapped["possum"].append("cat")
        values[labels.index("cat")] = new_labels.index("possum")

        del remapped["cat"]
    remapped_y = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values),
        ),
        default_value=tf.constant(-1),
        name="remapped_y",
    )
    num_labels = len(new_labels)
    logging.info("New labels are %s", new_labels)
    for k, v in zip(keys, values):
        logging.info("Mapping %s to %s", labels[k], new_labels[v])

    # 1 / 0
    filenames = tf.io.gfile.glob(f"{base_dir}/*.tfrecord")
    dataset = load_dataset(filenames, num_labels, args)
    if dataset is None:
        logging.warn("No dataset for %s", filenames)
        return None, None
    if not args.get("only_features"):
        logging.info("shuffling data")
        dataset = dataset.shuffle(
            4096, reshuffle_each_iteration=args.get("reshuffle", True)
        )
    # tf refues to run if epoch sizes change so we must decide a costant epoch size even though with reject res
    # it will chang eeach epoch, to ensure this take this repeat data and always take epoch_size elements
    dist = get_distribution(dataset, num_labels, batched=False)
    for label, d in zip(new_labels, dist):
        logging.info("Have %s: %s", label, d)
    epoch_size = np.sum(dist)
    logging.info("Setting dataset size to %s", epoch_size)
    if not args.get("only_features", False):
        dataset = dataset.repeat(2)
    scale_epoch = args.get("scale_epoch", None)
    if scale_epoch:
        epoch_size = epoch_size // scale_epoch
    dataset = dataset.take(epoch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    batch_size = args.get("batch_size", None)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset, remapped


def class_func(features, label):
    label = tf.argmax(label)
    return label


#
def preprocess(data):
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return tf.keras.applications.inception_v3.preprocess_input(x), y


def read_tfrecord(
    example, image_size, num_labels, labeled, augment, preprocess_fn=None, one_hot=True
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
        if one_hot:
            label = tf.one_hot(label, num_labels)
        return image, label
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
    resampled_ds, remapped = get_dataset(
        f"{config.tracks_folder}/training-data/test",
        labels,
        batch_size=1,
        image_size=(160, 160),
        augment=False,
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
