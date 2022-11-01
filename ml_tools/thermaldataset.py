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
from ml_tools.forestmodel import feature_mask

max_v = [
    55.990376,
    57.888,
    10.103797,
    10.595947,
    4.6359787,
    14.004379,
    16.872267,
    15.090072,
    1,
    9.00735,
    10.160552,
    14.971666,
    3.8627255,
    11.299641,
    9.566891,
    11.620289,
    1.7128139,
    21.84973,
    2.0132241,
    9.559188,
    3.8427675,
    11.813642,
    12.488583,
    9.508464,
    8.809811,
    2.9139967,
]
mean_v = [
    0.058490578,
    0.038204964,
    7.4425936,
    4.7920947,
    1.0412979,
    1.0230657,
    0.16372658,
    0.7525377,
    0,
    10.318023,
    2.1682348,
    36.300232,
    1.4874498,
    1.5263922,
    12.650131,
    1.0403844,
    0.45160577,
    0.012650883,
    -0.10378975,
    2.1496308,
    1.4747808,
    6.465994,
    0.8433721,
    11.38621,
    14.642979,
    0.55539894,
]

std_v = [
    0.33010894,
    0.1990348,
    5.117308,
    3.5965276,
    0.3059699,
    1.0528976,
    0.17207752,
    0.8468552,
    1,
    7.8997636,
    0.89097166,
    27.489365,
    0.39993584,
    1.6026465,
    12.699268,
    1.0943583,
    0.3201715,
    0.018906306,
    0.112067714,
    0.8889591,
    0.40037858,
    5.9949546,
    0.9455366,
    10.673864,
    13.96866,
    0.2524548,
]


# seed = 1341
# tf.random.set_seed(seed)
# np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64

insect = None
fp = None

USE_MVM = True


def load_dataset(
    filenames,
    image_size,
    num_labels,
    deterministic=False,
    labeled=True,
    augment=False,
    preprocess_fn=None,
    mvm=False,
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
            mvm=mvm,
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x[1]))
    dataset = dataset.filter(filter_nan)
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
    mvm=False,
    scale_epoch=None,
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
        mvm=mvm,
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

        dist = dist / np.sum(dist)
        dist_max = np.max(dist)
        dist_min = np.min(dist)
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
            elif dist_max - dist[i] > (max_range * 2):
                target_dist[i] = dist[i]
                # target_dist[i] -= max_range / 2.0
            target_dist[i] = max(0, target_dist[i])
        target_dist = target_dist / np.sum(target_dist)

        if "sheep" in labels:
            sheep_i = labels.index("sheep")
            target_dist[sheep_i] = 0
        rej = dataset.rejection_resample(
            class_func=class_func,
            target_dist=target_dist,
            # initial_dist=dist,
        )
        dataset = rej.map(lambda extra_label, features_and_label: features_and_label)

    dataset = dataset.shuffle(4096, reshuffle_each_iteration=reshuffle)
    # tf refues to run if epoch sizes change so we must decide a costant epoch size even though with reject res
    # it will chang eeach epoch, to ensure this take this repeat data and always take epoch_size elements
    epoch_size = len([0 for x, y in dataset])
    logging.info("Setting dataset size to %s", epoch_size)
    dataset = dataset.repeat(2)
    if scale_epoch:
        epoch_size = epoch_size // scale_epoch
    dataset = dataset.take(epoch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    if batch_size is not None:
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
    # resampled_ds = resampled_ds.batch(batch_size)
    return resampled_ds, remapped


def read_tfrecord(
    example,
    image_size,
    num_labels,
    labeled,
    augment=False,
    preprocess_fn=None,
    mvm=False,
):
    tf_mean = tf.constant(mean_v)
    tf_std = tf.constant(std_v)
    tfrecord_format = {
        "image/thermalencoded": tf.io.FixedLenFeature([25 * 32 * 32], dtype=tf.float32),
        "image/filteredencoded": tf.io.FixedLenFeature(
            [25 * 32 * 32], dtype=tf.float32
        ),
        "image/class/label": tf.io.FixedLenFeature((), tf.int64, -1),
    }
    if mvm:
        tfrecord_format["image/features"] = tf.io.FixedLenFeature(
            [36 * 5 + 1], dtype=tf.float32
        )
    example = tf.io.parse_single_example(example, tfrecord_format)
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
            "Preprocessing with %s.%s", preprocess_fn.__module__, preprocess_fn.__name__
        )
        image = preprocess_fn(image)
    if labeled:
        label = tf.cast(example["image/class/label"], tf.int32)
        global remapped_y
        label = remapped_y.lookup(label)
        onehot_label = tf.one_hot(label, num_labels)
        if mvm:
            features = example["image/features"]
            mask = feature_mask()
            features = tf.boolean_mask(features, mask)
            features = features - tf_mean
            features = features / tf_std
            return (image, features), onehot_label
        return image, onehot_label
    if mvm:
        return (image, features)
    return image


def decode_image(thermals, filtereds, image_size):
    deoced_thermals = []
    decoded_filtered = []
    for thermal, filtered in zip(thermals, filtereds):
        image = tf.image.decode_png(image, channels=1)
        filtered = tf.image.decode_png(filtered, channels=1)
        decoded_thermal.append(image)
        decoded_filtered.append(filtered)
        # image = tf.concat((image, image, filtered), axis=2)
        # image = tf.cast(image, tf.float32)
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
            f"{config.tracks_folder}/training-data/test",
            32,
            (160, 160),
            labels,
            augment=False,
            stop_on_empty_dataset=False,
            preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
            resample=True,
        )
    else:

        resampled_ds, remapped = get_resampled(
            # dir,
            f"{config.tracks_folder}/training-data/test",
            None,
            (160, 160),
            labels,
            augment=True,
            # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
            resample=True,
            mvm=True,
        )
    # print(get_distribution(resampled_ds))
    #
    #
    # for e in range(2):
    #     print("epoch", e)
    #     true_categories = [y for x, y in resampled_ds]
    #     true_categories = tf.concat(true_categories, axis=0)
    #     true_categories = np.int64(tf.argmax(true_categories, axis=1))
    #     c = Counter(list(true_categories))
    #     print("epoch is size", len(true_categories))
    #     for i in range(len(labels)):
    #         print("after have", labels[i], c[i])
    #
    # # return
    for e in range(1):
        minimum_features = None
        max_features = None
        mean_features = None
        count = 0
        a = [x[1] for x, y in resampled_ds]
        std = np.std(a, axis=0)
        mean_v = np.mean(a, axis=0)
        max_v = np.max(a, axis=0)
        min_v = np.min(a, axis=0)
        print("STD", std.shape)
        for v in std:
            print(v)
        print("MEAN")
        for v in mean_v:
            print(v)
        print("Min")
        for m in min_v:
            print(m)

        #
        # print("STD is", std.shape)
        # print(
        #     std,
        # )
        # print("MEAN")
        # print(np.mean(a, axis=1))


def show_batch(image_batch, label_batch, labels):
    features = image_batch[1]
    for f in features:
        for v in f:
            print(v)
        # print(f.shape, f.dtype)
        return
    image_batch = image_batch[0]
    print("features are", features.shape, image_batch.shape)
    plt.figure(figsize=(10, 10))
    print("images in batch", len(image_batch))
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
