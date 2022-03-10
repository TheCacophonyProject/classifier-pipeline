import matplotlib.pyplot as plt

import tensorflow as tf
from functools import partial

AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64


def load_dataset(filenames, image_size, num_labels, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(
            read_tfrecord, image_size=image_size, num_labels=num_labels, labeled=labeled
        ),
        num_parallel_calls=AUTOTUNE,
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def get_dataset(filenames, batch_size, image_size, num_labels, labeled=True):
    dataset = load_dataset(filenames, image_size, num_labels, labeled=labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def read_tfrecord(example, image_size, num_labels, labeled):
    tfrecord_format = {
        "image/thermalencoded": tf.io.FixedLenFeature((), tf.string),
        "image/filteredencoded": tf.io.FixedLenFeature((), tf.string),
        "image/height": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/width": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/class/label": tf.io.FixedLenFeature((), tf.int64, -1),
        # "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        # "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        # "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        # "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    }

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image/thermalencoded"], image_size)
    if labeled:
        # label = tf.cast(
        #     tf.reshape(example["image/object/class/label"], shape=[]), dtype=tf.int64
        # )
        label = tf.cast(example["image/class/label"], tf.int64)
        onehot_label = tf.one_hot(label, num_labels)

        return image, onehot_label
    return image


def decode_image(image, image_size):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)

    return image


# test crap
def main():
    train_files = tf.io.gfile.glob(
        "/home/gp/cacophony/classifier-pipeline/training-data/train/*.tfrecord"
    )
    print("got filename", train_files)
    train_dataset = get_dataset(train_files, 32, (256, 256))

    image_batch, label_batch = next(iter(train_dataset))
    show_batch(image_batch, label_batch)


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    labels = ["nothing", "possum"]
    print(label_batch)
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        print(image_batch[n].shape)
        plt.title(labels[label_batch[n]])
        plt.axis("off")
    plt.show()


if __name__ == "__main__":

    main()
