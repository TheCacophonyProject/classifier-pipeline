from collections import Counter
import tensorflow as tf
from functools import partial
import numpy as np
import logging
import random
import math

AUTOTUNE = tf.data.AUTOTUNE


def get_weighting(
    dataset, labels, min_weigth=0.25, max_weight=4, excluded_labels=[], dont_weight=[]
):
    num_labels = len(labels)
    dist = get_distribution(dataset, num_labels)
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


def get_distribution(
    labels, dataset, num_labels, batched=True, one_hot=True, extra_meta=False
):
    if extra_meta:
        true_categories = [y[0] for x, y in dataset]
    else:
        true_categories = [y for x, y in dataset]
    bird_index = labels.index("bird")
    dist = np.zeros((num_labels), dtype=np.float32)
    if len(true_categories) == 0:
        return dist
    if batched:
        true_categories = tf.concat(true_categories, axis=0)
    if len(true_categories) == 0:
        return dist
    classes = []
    if one_hot:
        for y in true_categories:
            y_np = y.numpy()
            non_zero = np.flatnonzero(y_np)
            if len(non_zero) > 1:
                if np.any(y_np[non_zero] != 1):
                    # fractional/soft labels e.g. 0.9 bird, 0.1 deer - just take the majority label
                    non_zero = [np.argmax(y_np)]
                elif bird_index in non_zero:
                    # bird tag
                    # just choose the more specific tag
                    non_zero = non_zero[non_zero != bird_index]

            classes.extend(non_zero)
    else:
        classes = true_categories.flatten()
    classes = np.array(classes)

    c = Counter(list(classes))
    for i in range(num_labels):
        dist[i] = c[i]
    return dist


def apply_label_mapping(labels, excluded_labels, label_mapping, model_labels=None):
    logging.info("Excluding %s", excluded_labels)
    if model_labels is not None:
        tf_mappings = {}
        filtered_labels = model_labels

        logging.info("Mapping DS labels %s to model labels %s", labels, model_labels)
        # if we are loading a model with different labels we need to map the dataset labels
        # to the equivalent model labels
        for l_i, og_lbl in enumerate(labels):
            # keys.append(l_i)
            try:
                lbl = og_lbl
                if lbl in label_mapping:
                    lbl = label_mapping[lbl]

                mdl_i = model_labels.index(lbl)
                # if lbl not in remapped:
                # remapped[lbl] = []
                # remapped[lbl].append(og_lbl)
                # values.append(mdl_i)
                tf_mappings[l_i] = mdl_i
            except:
                # remapped[og_lbl] = -1
                # values.append(-1)
                tf_mappings[l_i] = -1
        return model_labels, tf_mappings
    # get new labels after excluding and removing remapped labels
    filtered_labels = labels.copy()
    for excluded in excluded_labels:
        if excluded in filtered_labels:
            filtered_labels.remove(excluded)
    for remapped_lbl in label_mapping.keys():
        if remapped_lbl in filtered_labels:
            filtered_labels.remove(remapped_lbl)

    tf_mappings = {}
    # label indexes in tf records are hard coded base of labels at the time of writing so need to make sure we use those indexes
    for l in labels:
        if l not in filtered_labels:
            tf_mappings[labels.index(l)] = -1
            logging.info("Excluding %s", l)
        else:
            tf_mappings[labels.index(l)] = filtered_labels.index(l)

    # add the remapped labels to the correct place
    for k, v in label_mapping.items():
        if k in excluded_labels:
            continue
        if k in labels and v in filtered_labels:
            # remapped[v].append(k)
            # values[labels.index(k)] = filtered_labels.index(v)
            tf_mappings[labels.index(k)] = filtered_labels.index(v)
            # del remapped[k]
    return filtered_labels, tf_mappings


def get_dataset(load_function, base_dir, labels, **args):
    model_labels = args.get("model_labels")

    excluded_labels = args.get("excluded_labels", [])
    to_remap = args.get("remapped_labels", {})
    tf_mappings = args.get("tf_mappings")
    shuffle_size = 4096
    if args.get("num_frames", 25) == 1:
        shuffle_size *= 20

    keys = list(tf_mappings.keys())
    values = list(tf_mappings.values())
    remap_lookup = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values),
        ),
        default_value=tf.constant(-1),
        name="remapped_y",
    )
    num_labels = len(labels)
    # logging.info("New labels are %s from original %s", filtered_labels, labels)
    # for k, v in zip(keys, values):
    #     if labels[k]!= filtered_labels[v]:
    #         logging.info(
    #             "Mapping %s to %s", labels[k], filtered_labels[v] if v >= 0 else "nothing"
    #         )

    # 1 / 0
    filenames = tf.io.gfile.glob(f"{base_dir}/*.tfrecord")
    if not args.get("deterministic"):
        random.shuffle(filenames)

    dataset = load_function(filenames, remap_lookup, labels, args)
    if not args.get("one_hot", True):
        filter_excluded = lambda x, y: not tf.math.less(y, 0)
    else:
        if not args.get("include_track", False):
            filter_excluded = lambda x, y: not tf.math.equal(
                tf.math.count_nonzero(y), 0
            )
        else:
            filter_excluded = lambda x, y: not tf.math.equal(
                tf.math.count_nonzero(y[0]), 0
            )

    dataset = dataset.filter(filter_excluded)
    if dataset is None:
        logging.warn("No dataset for %s", filenames)
        return None, None

    if args.get("epoch_size") is not None:
        dataset = dataset.take(args.get("epoch_size"))
        logging.info("Setting dataset to %s", args.get("epoch_size"))
    if args.get("cache", False):
        dataset = dataset.cache()
    if (
        not args.get("only_features")
        and args.get("shuffle", True)
        and not args.get("resample")
    ):
        logging.info("shuffling data with buffer %s", shuffle_size)
        dataset = dataset.shuffle(
            shuffle_size, reshuffle_each_iteration=args.get("reshuffle", True)
        )

    batch_size = args.get("batch_size", None)

    # tf refuses to run if epoch sizes change so we must decide a costant epoch size even though with reject res
    # it will chang eeach epoch, to ensure this take this repeat data and always take epoch_size elements
    if not args.get("only_features"):
        dist = get_distribution(
            labels,
            dataset,
            num_labels,
            batched=False,
            one_hot=args.get("one_hot", True),
            extra_meta=args.get("include_track", False),
        )
        for label, d in zip(labels, dist):
            logging.info("Have %s: %s", label, d)
        epoch_size = np.sum(dist)
        logging.info("Setting dataset size to %s", epoch_size)
        if not args.get("only_features", False):
            dataset = dataset.repeat(2)
        if batch_size is not None:
            epoch_size = math.ceil(epoch_size / batch_size)
            epoch_size = int(epoch_size * batch_size)
        dataset = dataset.take(epoch_size)
        scale_epoch = args.get("scale_epoch", None)
        if scale_epoch:
            epoch_size = epoch_size // scale_epoch
            dataset = dataset.take(epoch_size)
    else:
        epoch_size = 1
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    augment = args.get("augment", False)
    if augment:
        logging.info("Augmenting on batches")
        dataset = dataset.map(
            lambda x, y: (
                {
                    "input_image": data_augmentation(x["input_image"], training=True),
                    "input_mask": x["input_mask"],
                },
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    preprocess_fn = args.get("preprocess_fn")
    if preprocess_fn is not None:
        logging.info(
            "Preprocessing with %s.%s",
            preprocess_fn.__module__,
            preprocess_fn.__name__,
        )
        dataset = dataset.map(
            lambda x, y: (
                {
                    "input_image": preprocess_fn(x["input_image"], training=True),
                    "input_mask": x["input_mask"],
                },
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset, epoch_size


brightness_contrast_aug = tf.keras.Sequential(
    [
        tf.keras.layers.RandomBrightness(0.2),  # better per frame or per sequence??
        tf.keras.layers.RandomContrast(0.5),
    ]
)


def data_augmentation(image, training=True):
    # only apply brightness/contrast to channels 2,3 (thermal_norm, filtered),
    # leave channel 1 (raw thermal) untouched
    raw = image[..., :1]
    augmented = brightness_contrast_aug(image[..., 1:], training=training)
    return tf.concat([raw, augmented], axis=-1)


def resample(dataset, labels):
    excluded_labels = ["sheep"]
    num_labels = len(labels)
    true_categories = [y for x, y in dataset]
    if len(true_categories) == 0:
        logging.info("no data")
        return None
    true_categories = np.int64(tf.argmax(true_categories, axis=1))
    c = Counter(list(true_categories))
    dist = np.empty((num_labels), dtype=np.float32)
    target_dist = np.empty((num_labels), dtype=np.float32)
    for i in range(num_labels):
        if labels[i] in excluded_labels:
            logging.info("Excluding %s for %s", c[i], labels[i])
            dist[i] = 0
        else:
            dist[i] = c[i]
            logging.info("Have %s for %s", dist[i], labels[i])
    zeros = dist[dist == 0]
    non_zero_labels = num_labels - len(zeros)
    target_dist[:] = 1 / non_zero_labels

    dist = dist / np.sum(dist)
    dist_max = np.max(dist)
    # really this is what we want but when the values become too small they never get sampled
    # so need to try reduce the large gaps in distribution
    # can use class weights to adjust more, or just throw out some samples
    max_range = target_dist[0] / 2
    for i in range(num_labels):
        if dist[i] == 0:
            target_dist[i] = 0
        elif dist_max - dist[i] > (max_range * 2):
            target_dist[i] = dist[i]

        target_dist[i] = max(0, target_dist[i])
    target_dist = target_dist / np.sum(target_dist)
