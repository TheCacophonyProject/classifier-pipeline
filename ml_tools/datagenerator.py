import random
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot

FRAME_SIZE = 48


class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        dataset,
        num_classes,
        batch_size=32,
        dim=(48, 48, 3),
        n_channels=5,
        shuffle=True,
        sequence_size=27,
        lstm=False,
        thermal_only=False,
    ):
        self.thermal_only = thermal_only
        self.lstm = lstm
        self.dim = dim
        self.augment = dataset.enable_augmentation
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.dataset = dataset
        self.size = len(dataset.segments)
        if not self.lstm:
            self.size = self.size
        self.indexes = np.arange(self.size)
        self.labels = dataset.labels
        self.shuffle = shuffle
        self.n_classes = num_classes
        self.n_channels = n_channels
        self.on_epoch_end()
        print("augmenting?", self.augment)

    def __len__(self):
        "Denotes the number of batches per epoch"

        return int(np.floor(100 / self.batch_size))

        # return int(np.floor(len(self.dataset.segments) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        X, y = self._data(indexes)
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

        # Generate data
        for i, index in enumerate(indexes):
            if self.lstm:
                segment_i = index
            else:
                segment_i = index
                 # int(index / self.sequence_size)
                slice_i = np.random.randint(26)
                 # index % 27
            segment = self.dataset.segments[segment_i]
            data,temp_median = self.dataset.fetch_segment(segment, augment=self.augment, preprocess=False)
            # data = np.array(data)
            data = np.float32(data[slice_i])
            data[0] -= np.float32(temp_median)[slice_i]

            data[0] = np.clip(data[0], a_min=0,a_max=None)
            if self.lstm:
                data = [
                    data[:, 0, :, :],
                    data[:, 1, :, :],
                    data[:, 4, :, :],
                ]

                data = np.transpose(data, (1, 2, 3, 0))
            elif self.thermal_only:
                # data = data[slice_i]
                data = data[np.newaxis, 0, :, :]
                data = np.repeat(data, 3, axis=0)
                data = np.transpose(data, (1, 2, 0))
            else:
                data = data[slice_i]
                data = [
                    data[0, :, :],
                    data[1, :, :],
                    data[4, :, :],
                ]

                data = np.transpose(data, (1, 2, 0))
                # Store sample

            if self.augment:
                data = augement_frame(data)
            else:
                data = resize(data)

            X[i,] = data
            y[i] = self.dataset.labels.index(segment.label)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def resize(image):
    sess = tf.Session();
    with sess.as_default():
        image = convert(image)
        image = tf.image.resize_with_crop_or_pad(image, FRAME_SIZE, FRAME_SIZE) # Add 6 pixels of padding
        return image.eval()

def convert(image):
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    return image

def augement_frame(frame):
    sess = tf.Session();
    with sess.as_default():     # data[:, 0, :, :] -= np.float32(reference_level)[:, np.newaxis, np.newaxis]
      image = convert(frame)
      # print(image)
      image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
      image = tf.image.resize_with_crop_or_pad(image, FRAME_SIZE+6, FRAME_SIZE+6) # Add 6 pixels of padding
      image = tf.image.random_crop(image, size=[FRAME_SIZE, FRAME_SIZE, 1]) # Random crop back to 28x28
      if random.random() > 0.50:
          rotated = tf.image.rot90(image)
      if random.random() > 0.50:
          flipped = tf.image.flip_left_right(image)
      image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
      # pyplot.show
      # print(image.eval())
      return image.eval()
