from collections import Counter
import tensorflow as tf
from functools import partial
import numpy as np
import logging
import random

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


def get_distribution(dataset, num_labels, batched=True, one_hot=True, extra_meta=False):
    if extra_meta:
        true_categories = [y[0] for x, y in dataset]
    else:
        true_categories = [y for x, y in dataset]

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
            non_zero = tf.where(y).numpy()
            classes.extend(non_zero.flatten())
    else:
        classes = true_categories.flatten()
    classes = np.array(classes)

    c = Counter(list(classes))
    for i in range(num_labels):
        dist[i] = c[i]
    return dist


def get_dataset(load_function, base_dir, labels, **args):
    land_birds = [
        "pukeko",
        "california quail",
        "brown quail",
        "black swan",
        "quail",
        "pheasant",
        "penguin",
        "duck",
        "chicken",
        "rooster",
    ]
    excluded_labels = args.get("excluded_labels", [])
    to_remap = args.get("remapped_labels", {})
    logging.info("Excluding %s", excluded_labels)
    remapped = {}
    keys = []
    values = []
    # excluded_labels.append("insect")
    # excluded_labels.append("cat")
    new_labels = labels.copy()
    for excluded in excluded_labels:
        if excluded in labels:
            new_labels.remove(excluded)
    for remapped_lbl in to_remap.keys():
        if remapped_lbl in labels:
            new_labels.remove(remapped_lbl)
    for l in labels:
        keys.append(labels.index(l))
        if l not in new_labels:
            remapped[l] = -1
            values.append(-1)
            logging.info("Excluding %s", l)
        else:
            remapped[l] = [l]
            values.append(new_labels.index(l))
    for k, v in to_remap.items():
        if k in labels and v in labels:
            remapped[v].append(k)
            values[labels.index(k)] = new_labels.index(v)
            del remapped[k]
    remap_lookup = tf.lookup.StaticHashTable(
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
        logging.info(
            "Mapping %s to %s", labels[k], new_labels[v] if v >= 0 else "nothing"
        )

    # 1 / 0
    filenames = tf.io.gfile.glob(f"{base_dir}/*.tfrecord")
    if not args.get("deterministic"):
        random.shuffle(filenames)

    dataset = load_function(filenames, remap_lookup, new_labels, args)
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

    if args.get("resample"):
        logging.info("RESAMPLING")
        # seems the only way to get even distribution
        label_ds = []
        for i, l in enumerate(new_labels):
            l_mask = np.zeros((len(new_labels)))
            l_mask[i] = 1
            # mask = tf.constant(mask, dtype=tf.float32)

            l_filter = lambda x, y: tf.math.reduce_all(tf.math.equal(y, l_mask))
            l_dataset = dataset.filter(l_filter)
            l_dataset = l_dataset.shuffle(40096, reshuffle_each_iteration=True)

            label_ds.append(l_dataset)
        dataset = tf.data.Dataset.sample_from_datasets(
            label_ds,
            # weights=[1 / len(new_labels)] * len(new_labels),
            stop_on_empty_dataset=True,
            rerandomize_each_iteration=True,
        )
    if args.get("cache", False):
        dataset = dataset.cache()
    if (
        not args.get("only_features")
        and args.get("shuffle", True)
        and not args.get("resample")
    ):
        logging.info("shuffling data")
        dataset = dataset.shuffle(
            4096, reshuffle_each_iteration=args.get("reshuffle", True)
        )
    # tf refues to run if epoch sizes change so we must decide a costant epoch size even though with reject res
    # it will chang eeach epoch, to ensure this take this repeat data and always take epoch_size elements
    if not args.get("only_features"):
        dist = get_distribution(
            dataset,
            num_labels,
            batched=False,
            one_hot=args.get("one_hot", True),
            extra_meta=args.get("include_track", False),
        )
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
    else:
        epoch_size = 1
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    batch_size = args.get("batch_size", None)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset, remapped, new_labels, epoch_size


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
