import random
import logging
import tensorflow.keras as keras
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ml_tools.dataset import TrackChannels

FRAME_SIZE = 48


class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        dataset,
        labels,
        num_classes,
        batch_size,
        preprocess_fn=None,
        dim=(FRAME_SIZE, FRAME_SIZE, 3),
        n_channels=5,
        shuffle=True,
        sequence_size=27,
        lstm=False,
        use_thermal=False,
        use_filtered=False,
    ):
        self.labels = labels
        self.preprocess_fn = preprocess_fn
        self.use_thermal = use_thermal
        self.use_filtered = use_filtered
        self.lstm = lstm
        # default
        if not self.use_thermal and not self.use_filtered and not self.lstm:
            self.use_filtered = True
        self.dim = dim
        self.augment = dataset.enable_augmentation
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.dataset = dataset
        self.size = len(dataset.frame_samples)
        if not self.lstm:
            self.size = self.size
        self.indexes = np.arange(self.size)
        self.labels = dataset.labels
        self.shuffle = shuffle
        self.n_classes = num_classes
        self.n_channels = n_channels
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"

        return int(np.floor(self.dataset.frames / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        X, y, clips = self._data(indexes)
        # if self.dataset.name == "train":
        #     #
        #     fig = plt.figure(figsize=(48, 48))
        #     for i in range(len(X)):
        #         axes = fig.add_subplot(4, 10, i + 1)
        #         axes.set_title(
        #             "{} - {} track {} frame {}".format(
        #                 self.labels[np.argmax(np.array(y[i]))],
        #                 clips[i].clip_id,
        #                 clips[i].track_id,
        #                 clips[i].frame_num,
        #             )
        #         )
        #         plt.imshow(tf.keras.preprocessing.image.array_to_img(X[i]))
        #     plt.savefig("testimage.png")
        #     plt.close(fig)
        # raise "save err"

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data(self, indexes):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.lstm:
            X = np.empty((self.batch_size, self.sequence_size, *self.dim))
        else:
            X = np.empty((self.batch_size, *self.dim))

        y = np.empty((self.batch_size), dtype=int)
        clips = []
        # Generate data
        for i, index in enumerate(indexes):
            segment_i = index
            frame = self.dataset.frame_samples[segment_i]
            data, label = self.dataset.fetch_frame(frame)

            if self.lstm:
                data = [
                    data[:, 0, :, :],
                    data[:, 1, :, :],
                    data[:, 4, :, :],
                ]

                data = np.transpose(data, (1, 2, 3, 0))
            else:
                if self.use_thermal:
                    channel = TrackChannels.thermal
                else:
                    channel = TrackChannels.filtered
                data = data[channel]

                # normalizes data, constrast stretch good or bad?
                if self.augment:
                    percent = random.randint(0, 2)
                else:
                    percent = 0
                max = int(np.percentile(data, 100 - percent))
                min = int(np.percentile(data, percent))
                # print("max", np.amax(data), "98 p[ercentile]", np.percentile(data, 98))
                # print(min)
                # print(max)
                if max == min:
                    logging.error(
                        "frame max and min are the same clip %s track %s frame %s",
                        frame.clip_id,
                        frame.track_id,
                        frame.frame_num,
                    )
                    continue

                data -= min
                data = data / (max - min)
                np.clip(data, a_min=0, a_max=None, out=data)

                data = data[np.newaxis, :]
                data = np.transpose(data, (1, 2, 0))
                data = np.repeat(data, 3, axis=2)

            if self.augment:
                data = augement_frame(data, self.dim)
                data = np.clip(data, a_min=0, a_max=None, out=data)
            else:
                data = reisze_cv(data, self.dim)

            # pre proce expects values in range 0-255
            if self.preprocess_fn:
                data = data * 255
                data = self.preprocess_fn(data)
            X[i,] = data
            y[i] = self.labels.index(label)
            clips.append(frame)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes), clips


def resize(image, dim):
    image = convert(image)
    image = tf.image.resize(image, dim[0], dim[1])
    return image.numpy()


def reisze_cv(image, dim, interpolation=cv2.INTER_LINEAR, extra_h=0, extra_v=0):
    return cv2.resize(
        image, dsize=(dim[0] + extra_h, dim[1] + extra_v), interpolation=interpolation,
    )


def convert(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def augement_frame(frame, dim):
    frame = reisze_cv(
        frame,
        dim,
        extra_h=random.randint(0, int(FRAME_SIZE * 0.1)),
        extra_v=random.randint(0, int(FRAME_SIZE * 0.1)),
    )

    image = convert(frame)
    # image = tf.image.resize(
    #     image, [FRAME_SIZE + random.randint(0, 4), FRAME_SIZE + random.randint(0, 4)],
    # )  # Add 6 pixels of padding
    image = tf.image.random_crop(
        image, size=[dim[0], dim[1], 3]
    )  # Random crop back to 28x28
    if random.random() > 0.50:
        rotated = tf.image.rot90(image)
    if random.random() > 0.50:
        flipped = tf.image.flip_left_right(image)

        # maybes thisd should only be sometimes, as otherwise our validation set
    # if random.random() > 0.20:
    image = tf.image.random_contrast(image, 0.8, 1.2)
    # image = tf.image.random_brightness(image, max_delta=0.05)  # Random brightness
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)
    return image.numpy()
