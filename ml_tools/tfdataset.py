from collections import Counter
import tensorflow as tf
from functools import partial
import numpy as np
import logging

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


def load_dataset(read_function, filenames, remap_lookup, num_labels, args):
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
            read_function,
            remap_lookup=remap_lookup,
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


def get_dataset(read_function, base_dir, labels, **args):
    excluded_labels = args.get("excluded_labels", [])
    to_remap = args.get("remapped_labels", {})

    remapped = {}
    keys = []
    values = []
    # excluded_labels.append("insect")
    # excluded_labels.append("cat")
    new_labels = labels.copy()
    for excluded in excluded_labels:
        if excluded in labels:
            new_labels.remove(excluded)
    for excluded in to_remap.keys():
        if excluded in labels:
            new_labels.remove(excluded)
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
        logging.info("Mapping %s to %s", labels[k], new_labels[v])

    # 1 / 0
    filenames = tf.io.gfile.glob(f"{base_dir}/*.tfrecord")
    dataset = load_dataset(read_function, filenames, remap_lookup, num_labels, args)
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
    return dataset, remapped, new_labels
