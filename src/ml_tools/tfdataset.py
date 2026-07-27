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
    bird_index = -1
    if "bird" in labels:
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

    augment = args.get("augment", False)
    if augment:
        # logging.info("Augmenting on batches")
        fp_index = labels.index("false-positive")
        fp_index = tf.constant(fp_index)
        dataset = dataset.map(
            lambda x, y: (
                {
                    "input_image": data_augmentation(x["input_image"], y, fp_index),
                    "input_mask": x["input_mask"],
                },
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # if doing rnn batch after augment
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
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

    # dataset = dataset.map(resize_mosaic, num_parallel_calls=tf.data.AUTOTUNE)
    # doing this early would speed things up but for testing it best performance it wont matter too much
    if args.get("single_input", False):
        logging.info("Loading single input")
        dataset = dataset.map(
            lambda x, y: (x["input_image"], y), num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset, epoch_size


# @tf.function
# def resize_mosaic(x, label):
#     # the smallest regions we accept are 4 by 4 pixels which will be enlarged by preprocessing into 16 by 16
#     # efficient architecture will shrink features that 16 by 16 pixels down to 1 by 1 representation.
#     # To avoid this we double the image size this seems to improve accuracy by 1-2%
#     # Ensure image is in floats before resizing to maintain smooth gradients
#     image = x["input_image"]
#     image = tf.image.convert_image_dtype(image, tf.float32)

#     # Pixel-perfect 2.0x upscale to stop fractional stretching
#     image = tf.image.resize(image, [320, 320], method="bicubic")

#     # Optional: Clip values to 0-1 range to prevent any bicubic ringing overshoot
#     image = tf.clip_by_value(image, 0.0, 255.0)

#     return {"input_image": image, "input_mask": x["input_mask"]}, label


@tf.function
def bright_contrast_augmentation(image, labels, fp_index):

    # 1. Determine if this sample is a false-positive tensor
    is_false_positive = tf.equal(labels[fp_index], 1.0)

    # 2. Define what happens for a false-positive (Return the image untouched)
    def skip_augmentation():
        return image

    # 3. Define what happens for an animal (Use official random functions)
    def apply_official_augmentation():
        # Generate uniform seeds required by stateless functions
        bright_seed = tf.random.uniform([2], minval=0, maxval=1000, dtype=tf.int32)
        contrast_seed = tf.random.uniform([2], minval=0, maxval=1000, dtype=tf.int32)

        # Apply official library transformations to the whole mosaic
        augmented = tf.image.stateless_random_brightness(
            image, max_delta=25.0, seed=bright_seed
        )
        augmented = tf.image.stateless_random_contrast(
            augmented, lower=0.8, upper=1.2, seed=contrast_seed
        )

        # Ensure pixel boundaries stay valid between 0 and 255
        return tf.clip_by_value(augmented, 0.0, 255.0)

    # Natively route the image inside the TF data pipeline graph
    augmented_image = tf.cond(
        is_false_positive, skip_augmentation, apply_official_augmentation
    )

    return augmented_image


def apply_channel_isolated_transforms(frame, bright_seed, contrast_seed):
    # only apply brightness/contrast to channels 2,3 (thermal_norm, filtered),
    # leave channel 1 (raw thermal) untouched
    channel_0 = frame[:, :, 0:1]  # Kept raw and untouched
    channels_1_2 = frame[:, :, 1:3]  # This 2-channel tensor gets augmented

    # Apply brightness only to channels 1 and 2
    channels_1_2 = tf.image.stateless_random_brightness(
        channels_1_2, max_delta=0.1, seed=bright_seed
    )

    # Apply contrast only to channels 1 and 2
    channels_1_2 = tf.image.stateless_random_contrast(
        channels_1_2, lower=0.8, upper=1.2, seed=contrast_seed
    )
    channels_1_2 = tf.clip_by_value(
        channels_1_2, clip_value_min=0.0, clip_value_max=255.0
    )  #

    # Reconstruct the 3-channel frame by splicing the pieces back together
    modified_frame = tf.concat([channel_0, channels_1_2], axis=-1)

    return modified_frame


brightness_contrast_aug = tf.keras.Sequential(
    [
        tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 255.0)),
        tf.keras.layers.RandomContrast(0.5, value_range=(0.0, 255.0)),
    ]
)


def data_augmentation(image, y, fp_index, training=True):
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
