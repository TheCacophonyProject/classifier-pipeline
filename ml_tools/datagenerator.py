import tensorflow.keras as keras
import numpy as np


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
        augment=False,
        sequence_size=27,
        lstm=False,
    ):
        self.lstm = lstm
        self.dim = dim
        self.augment = augment
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.dataset = dataset
        self.size = len(dataset.segments)
        if not self.lstm:
            self.size = self.size * 27
        self.indexes = np.arange(self.size)
        self.labels = dataset.labels
        self.shuffle = shuffle
        self.n_classes = num_classes
        self.n_channels = n_channels
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"

        # return int(np.floor(100 / self.batch_size))

        return int(np.floor(len(self.dataset.segments) / self.batch_size))

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
                segment_i = int(index / self.sequence_size)
                slice_i = index % 27
            segment = self.dataset.segments[segment_i]
            data = self.dataset.fetch_segment(segment, augment=self.augment)
            if self.lstm:
                data = [
                    data[:, 0, :, :],
                    data[:, 1, :, :],
                    data[:, 4, :, :],
                ]

                data = np.transpose(data, (1, 2, 3, 0))
                # Store sample
                X[i,] = data
            else:
                data = data[slice_i]
                data = [
                    data[0, :, :],
                    data[1, :, :],
                    data[4, :, :],
                ]

                data = np.transpose(data, (1, 2, 0))
                # Store sample
            X[i,] = data
            y[i] = self.dataset.labels.index(segment.label)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
