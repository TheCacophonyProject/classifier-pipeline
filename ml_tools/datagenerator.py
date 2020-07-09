import random
import logging
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ml_tools.dataset import TrackChannels

FRAME_SIZE = 48


class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        dataset,
        num_classes,
        batch_size,
        preprocess_fn=None,
        dim=(48, 48, 3),
        n_channels=5,
        shuffle=True,
        sequence_size=27,
        lstm=False,
        use_thermal=False,
        use_filtered=False,
    ):
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
        #     print(y)
        #     raise "save err"

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

                np.clip(data, a_min=0, a_max=None, out=data)

                # normalizes data
                max = np.amax(data)
                min = np.amin(data)
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

                data = data[np.newaxis, :]
                data = np.transpose(data, (1, 2, 0))
                data = np.repeat(data, 3, axis=2)

            if self.augment:
                data = augement_frame(data)
                data = np.clip(data, a_min=0, a_max=None, out=data)
            else:
                data = resize(data)

            # pre proce expects values in range 0-255
            if self.preprocess_fn:
                data = data * 255
                data = self.preprocess_fn(data)
            X[i,] = data
            y[i] = self.dataset.labels.index(label)
            clips.append(frame)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes), clips


def resize(image):
    image = convert(image)
    image = tf.image.resize(image, [FRAME_SIZE, FRAME_SIZE])
    return image.numpy()


def convert(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def augement_frame(frame):
    image = convert(frame)
    image = tf.image.resize(
        image, [FRAME_SIZE + random.randint(0, 4), FRAME_SIZE + random.randint(0, 4)],
    )  # Add 6 pixels of padding
    image = tf.image.random_crop(
        image, size=[FRAME_SIZE, FRAME_SIZE, 3]
    )  # Random crop back to 28x28
    if random.random() > 0.50:
        rotated = tf.image.rot90(image)
    if random.random() > 0.50:
        flipped = tf.image.flip_left_right(image)
    image = tf.image.random_contrast(image, 0.2, 0.5)
    # image = tf.image.random_brightness(image, max_delta=0.05)  # Random brightness
    return image.numpy()
