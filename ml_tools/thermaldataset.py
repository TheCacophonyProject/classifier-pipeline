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

std_v = [
    0.15178847,
    0.027188646,
    0.8006742,
    0.116820745,
    0.018843347,
    0.11099239,
    0.015115197,
    0.012320447,
    0.00834021,
    0.329471,
    0.04178793,
    0.0347461,
    0.022275506,
    0.5834877,
    0.07105486,
    0.060715888,
    0.036653668,
    0.39131087,
    0.09954733,
    0.25194815,
    0.3400745,
    0.07829856,
    0.21469903,
    0.19698487,
    0.05322211,
    0.12876655,
    0.0511503,
    0.013950486,
    0.033352852,
    0.044274736,
    0.011132633,
    0.02844392,
    0.02625876,
    0.007514648,
    0.017128784,
    0.111962445,
    5.1221814,
    0.3431858,
    17.55858,
    3.5914004,
    0.30473834,
    1.0399487,
    0.17141788,
    0.14244178,
    0.10674831,
    2.5097816,
    0.38684186,
    0.33887756,
    0.21738514,
    3.76918,
    0.5712618,
    0.51205426,
    0.30125138,
    3.7100444,
    0.9942686,
    2.3026998,
    3.3430483,
    0.8423652,
    2.0555134,
    1.9229096,
    0.57602894,
    1.1992636,
    0.5754329,
    0.16434622,
    0.35606363,
    0.5118486,
    0.13673806,
    0.3135709,
    0.3143866,
    0.09890821,
    0.19578817,
    0.21177165,
    7.927947,
    0.8920534,
    27.46837,
    5.548869,
    0.39970443,
    5.966005,
    1.1091628,
    0.939299,
    0.77037466,
    10.794262,
    1.8495663,
    1.6004384,
    1.251242,
    14.1369505,
    2.474899,
    2.1609187,
    1.5738066,
    14.170539,
    5.854833,
    9.569272,
    12.7385235,
    4.984896,
    8.47711,
    8.890397,
    4.285393,
    6.2739615,
    2.4881508,
    1.0854256,
    1.6666657,
    2.1699314,
    0.91762626,
    1.4487588,
    1.5917629,
    0.7473355,
    1.1073718,
    0.31953955,
    0.15178847,
    0.027188646,
    0.8006742,
    0.116820745,
    0.018843347,
    0.11099239,
    0.015115197,
    0.012320447,
    0.00834021,
    0.329471,
    0.04178793,
    0.0347461,
    0.022275506,
    0.5834877,
    0.07105486,
    0.060715888,
    0.036653668,
    0.39131087,
    0.09954733,
    0.25194815,
    0.3400745,
    0.07829856,
    0.21469903,
    0.19698487,
    0.05322211,
    0.12876655,
    0.0511503,
    0.013950486,
    0.033352852,
    0.044274736,
    0.011132633,
    0.02844392,
    0.02625876,
    0.007514648,
    0.017128784,
    0.111962445,
    7.89409,
    0.89023757,
    27.364487,
    5.5273733,
    0.4001557,
    5.9426584,
    1.1057465,
    0.9364583,
    0.76861244,
    10.711998,
    1.837606,
    1.5904051,
    1.2455189,
    13.997581,
    2.4562283,
    2.1443408,
    1.5653683,
    14.053731,
    5.8327866,
    9.494816,
    12.636312,
    4.9669333,
    8.413056,
    8.8396435,
    4.2758307,
    6.241441,
    2.471447,
    1.0821238,
    1.6556802,
    2.155183,
    0.9149555,
    1.4393982,
    1.5839542,
    0.74572027,
    1.1021901,
    0.25196075,
    781.5659,
]
mean_v = [
    0.0775348,
    0.0184866,
    0.17836875,
    0.043902412,
    0.012626628,
    0.021495642,
    0.0035725704,
    0.002592158,
    0.0017336052,
    0.05838492,
    0.00982762,
    0.007427965,
    0.00459327,
    0.09296366,
    0.015483195,
    0.012066118,
    0.0069819894,
    0.079731025,
    0.019891461,
    0.05146406,
    0.06302364,
    0.0138297,
    0.039454263,
    0.038030125,
    0.008967926,
    0.024306273,
    0.013587593,
    0.0033395255,
    0.0087557,
    0.010673989,
    0.0023517516,
    0.0066972063,
    0.0064035966,
    0.0014733237,
    0.0040744897,
    -0.10393414,
    7.435397,
    1.449636,
    21.730503,
    4.788892,
    1.0398704,
    1.019794,
    0.16331418,
    0.12718,
    0.09059019,
    2.3248448,
    0.3675765,
    0.29620314,
    0.18705869,
    3.3680613,
    0.5306535,
    0.43374568,
    0.25865704,
    3.4652042,
    0.96486276,
    2.1687937,
    2.8522239,
    0.74997395,
    1.761103,
    1.7225814,
    0.5047326,
    1.0735674,
    0.5475687,
    0.15530284,
    0.34295163,
    0.4471794,
    0.12006074,
    0.27630955,
    0.27425075,
    0.08154664,
    0.17121173,
    0.24049263,
    10.316648,
    2.1681795,
    36.265934,
    7.6294994,
    1.4859247,
    6.4596114,
    1.0668923,
    0.8410836,
    0.6629097,
    11.425764,
    1.8498229,
    1.522181,
    1.0764292,
    14.704571,
    2.382863,
    1.995958,
    1.3144724,
    14.834788,
    6.2911367,
    10.257662,
    12.623511,
    4.9804153,
    8.574546,
    8.250467,
    3.9312124,
    5.886599,
    2.4140992,
    1.0354444,
    1.6724486,
    2.0209582,
    0.8128887,
    1.3785493,
    1.3409489,
    0.632448,
    0.95295936,
    0.45117363,
    0.0775348,
    0.0184866,
    0.17836875,
    0.043902412,
    0.012626628,
    0.021495642,
    0.0035725704,
    0.002592158,
    0.0017336052,
    0.05838492,
    0.00982762,
    0.007427965,
    0.00459327,
    0.09296366,
    0.015483195,
    0.012066118,
    0.0069819894,
    0.079731025,
    0.019891461,
    0.05146406,
    0.06302364,
    0.0138297,
    0.039454263,
    0.038030125,
    0.008967926,
    0.024306273,
    0.013587593,
    0.0033395255,
    0.0087557,
    0.010673989,
    0.0023517516,
    0.0066972063,
    0.0064035966,
    0.0014733237,
    0.0040744897,
    -0.10393414,
    10.239126,
    2.1496806,
    36.087593,
    7.5856633,
    1.4732772,
    6.4381003,
    1.0633264,
    0.83849317,
    0.661174,
    11.367273,
    1.8399847,
    1.5147517,
    1.0718305,
    14.611557,
    2.3673713,
    1.9838716,
    1.3074843,
    14.754991,
    6.2712064,
    10.206222,
    12.560379,
    4.966598,
    8.535155,
    8.212366,
    3.9222562,
    5.8623166,
    2.4005082,
    1.0321054,
    1.6637,
    2.0102568,
    0.8105333,
    1.3718503,
    1.3345402,
    0.63097554,
    0.9488823,
    0.5551107,
    544.0967,
]
max_v = [
    5.266641,
    0.7295258,
    55.874355,
    12.407583,
    0.42574856,
    4.6970706,
    0.5913621,
    0.4891105,
    0.5105152,
    18.541414,
    1.8234379,
    1.3877827,
    1.3213683,
    32.52449,
    3.1560757,
    2.902113,
    2.2729917,
    17.635914,
    3.7969172,
    11.300453,
    15.681583,
    3.3487377,
    9.637456,
    11.559932,
    3.2512615,
    8.063315,
    1.7205237,
    0.5322259,
    1.1013454,
    1.53471,
    0.43452173,
    0.98158187,
    0.8223772,
    0.45946372,
    0.56265736,
    0.1218277,
    59.146835,
    6.826627,
    363.75354,
    42.90071,
    2.642128,
    15.768243,
    3.0670645,
    2.103203,
    2.4106874,
    45.111282,
    5.565148,
    4.6710477,
    5.129014,
    66.91348,
    8.009847,
    7.891436,
    5.5637903,
    54.692432,
    15.177426,
    37.247868,
    49.73333,
    13.531644,
    33.229866,
    29.094265,
    9.657692,
    18.89423,
    7.739102,
    2.4851654,
    4.6395707,
    6.7167616,
    2.0008981,
    4.0531044,
    7.1430173,
    1.902089,
    4.2391486,
    0.98569405,
    81.47396,
    9.658119,
    447.86182,
    93.05687,
    3.0322921,
    86.93836,
    14.231551,
    14.154847,
    11.425444,
    149.80586,
    24.475191,
    23.96301,
    18.465813,
    158.59717,
    25.904163,
    24.993992,
    23.303602,
    158.59717,
    86.93836,
    136.87172,
    142.558,
    80.54633,
    125.66128,
    94.24729,
    55.38941,
    70.02093,
    25.904163,
    14.231551,
    21.588293,
    24.993992,
    14.154847,
    21.145641,
    23.303602,
    11.224309,
    16.532688,
    1.0,
    5.266641,
    0.7295258,
    55.874355,
    12.407583,
    0.42574856,
    4.6970706,
    0.5913621,
    0.4891105,
    0.5105152,
    18.541414,
    1.8234379,
    1.3877827,
    1.3213683,
    32.52449,
    3.1560757,
    2.902113,
    2.2729917,
    17.635914,
    3.7969172,
    11.300453,
    15.681583,
    3.3487377,
    9.637456,
    11.559932,
    3.2512615,
    8.063315,
    1.7205237,
    0.5322259,
    1.1013454,
    1.53471,
    0.43452173,
    0.98158187,
    0.8223772,
    0.45946372,
    0.56265736,
    0.1218277,
    80.22052,
    9.651446,
    409.3049,
    80.64929,
    3.0133426,
    86.93618,
    14.130492,
    14.057236,
    11.425389,
    149.80127,
    24.474245,
    23.962605,
    17.860376,
    158.58932,
    25.218443,
    24.821898,
    22.51365,
    158.5896,
    86.93673,
    136.86754,
    142.5539,
    80.54497,
    125.65789,
    94.24673,
    55.38919,
    70.020584,
    25.22434,
    14.1322775,
    21.587366,
    24.827751,
    14.058092,
    21.14517,
    22.575365,
    11.224263,
    16.016043,
    1.2910514,
    5355.0,
]
min_v = [
    0.0024567149,
    0.00041773738,
    -98.94636,
    0.00018393743,
    0.00024931348,
    2.000577e-06,
    3.753841e-07,
    3.321249e-08,
    3.016718e-08,
    8.151944e-06,
    4.532685e-07,
    2.0593902e-08,
    8.996015e-08,
    0.0,
    0.0,
    0.0,
    0.0,
    6.9304488e-06,
    1.1758193e-06,
    6.9304488e-06,
    2.6487057e-06,
    3.985186e-09,
    2.6487057e-06,
    6.3630896e-06,
    1.402394e-07,
    4.3063815e-06,
    1.3096371e-06,
    3.076307e-07,
    7.7549294e-07,
    7.4281337e-07,
    1.0732978e-09,
    6.6022767e-07,
    5.447441e-07,
    3.00843e-08,
    2.5661197e-07,
    -0.4988542,
    2.7620313,
    1.0562624,
    1.0029786,
    0.24650556,
    0.23686479,
    0.049913026,
    0.00514635,
    0.004097767,
    0.0014722049,
    0.09457448,
    0.0094766645,
    0.0073745004,
    0.0026115745,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1413091,
    0.044131838,
    0.08782365,
    0.07706492,
    0.027799856,
    0.049964942,
    0.036259208,
    0.011387103,
    0.022686789,
    0.014638636,
    0.004568144,
    0.009061815,
    0.011063126,
    0.0030667211,
    0.006853062,
    0.0037210577,
    0.0011650543,
    0.0023222072,
    0.006621363,
    3.1884089,
    1.1166701,
    1.1562647,
    0.38648725,
    0.34201077,
    0.18428358,
    0.018387238,
    0.015434656,
    0.0065906085,
    0.21205981,
    0.023061559,
    0.01946192,
    0.009188475,
    0.0,
    0.0,
    0.0,
    0.0,
    0.25144592,
    0.16254942,
    0.17913234,
    0.20892958,
    0.11469433,
    0.17540546,
    0.10607244,
    0.05943781,
    0.0854247,
    0.028246945,
    0.01621867,
    0.019429315,
    0.025273267,
    0.015434656,
    0.018803911,
    0.010267803,
    0.006257438,
    0.008269104,
    -0.37620667,
    0.0024567149,
    0.00041773738,
    -98.94636,
    0.00018393743,
    0.00024931348,
    2.000577e-06,
    3.753841e-07,
    3.321249e-08,
    3.016718e-08,
    8.151944e-06,
    4.532685e-07,
    2.0593902e-08,
    8.996015e-08,
    0.0,
    0.0,
    0.0,
    0.0,
    6.9304488e-06,
    1.1758193e-06,
    6.9304488e-06,
    2.6487057e-06,
    3.985186e-09,
    2.6487057e-06,
    6.3630896e-06,
    1.402394e-07,
    4.3063815e-06,
    1.3096371e-06,
    3.076307e-07,
    7.7549294e-07,
    7.4281337e-07,
    1.0732978e-09,
    6.6022767e-07,
    5.447441e-07,
    3.00843e-08,
    2.5661197e-07,
    -0.4988542,
    2.9720984,
    1.0015243,
    1.1329304,
    0.3863033,
    0.33826277,
    0.184254,
    0.018384408,
    0.015368973,
    0.006587795,
    0.21168202,
    0.02302543,
    0.019347746,
    0.009186907,
    0.0,
    0.0,
    0.0,
    0.0,
    0.25080606,
    0.16252005,
    0.17895478,
    0.20215565,
    0.11449913,
    0.17524897,
    0.10597302,
    0.05943491,
    0.08538564,
    0.028185762,
    0.016215863,
    0.019412337,
    0.025034452,
    0.01540511,
    0.018684408,
    0.0102582965,
    0.006257161,
    0.00826537,
    0.033692285,
    9.0,
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
    tree_mode=False,
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
            tree_mode=tree_mode,
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
    tree_mode=False,
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
        tree_mode=tree_mode,
    )
    if resample:
        true_categories = [y for x, y in dataset]
        if len(true_categories) == 0:
            return None
        true_categories = np.int64(tf.argmax(true_categories, axis=1))
        print(true_categories)
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

    # dataset = dataset.shuffle(4096, reshuffle_each_iteration=reshuffle)
    # tf refues to run if epoch sizes change so we must decide a costant epoch size even though with reject res
    # it will chang eeach epoch, to ensure this take this repeat data and always take epoch_size elements
    epoch_size = len([0 for x, y in dataset])
    logging.info("Setting dataset size to %s", epoch_size)
    if not tree_mode:
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
    tree_mode=False,
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
            # mask = feature_mask()
            # features = tf.boolean_mask(features, mask)
            features = features - tf_mean
            features = features / tf_std
            if tree_mode:
                return features, label

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
        print("MAX")
        for m in max_v:
            print(m)
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
