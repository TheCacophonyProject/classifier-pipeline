from PIL import Image, ImageDraw, ImageFont, ImageColor
from pathlib import Path
import pickle
import math
import random
import logging
import tensorflow.keras as keras
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ml_tools.dataset import TrackChannels
import multiprocessing
import time
from ml_tools.dataset import filtered_is_valid
from ml_tools import tools
from ml_tools.preprocess import preprocess_movement, preprocess_frame

FRAME_SIZE = 48
FRAMES_PER_SECOND = 9


class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        dataset,
        labels,
        num_classes,
        batch_size,
        model_preprocess=None,
        dim=(FRAME_SIZE, FRAME_SIZE, 3),
        n_channels=5,
        shuffle=True,
        use_thermal=False,
        use_filtered=False,
        buffer_size=128,
        epochs=10,
        load_threads=1,
        preload=True,
        use_movement=False,
        balance_labels=True,
        keep_epoch=False,
        randomize_epoch=True,
        cap_samples=True,
        cap_at=None,
        square_width=5,
        label_cap=None,
        use_dots=False,
    ):
        self.use_dots = use_dots
        self.label_cap = label_cap
        self.cap_at = cap_at
        self.cap_samples = cap_samples
        self.randomize_epoch = randomize_epoch
        self.use_previous_epoch = None
        self.keep_epoch = keep_epoch
        self.balance_labels = balance_labels

        self.labels = labels
        self.model_preprocess = model_preprocess
        self.use_thermal = use_thermal
        self.use_filtered = use_filtered

        if not self.use_thermal and not self.use_filtered:
            self.use_thermal = True
        self.movement = use_movement
        self.square_width = square_width
        if use_movement:
            dim = (dim[0] * self.square_width, dim[1] * self.square_width, dim[2])
        self.dim = dim
        self.augment = dataset.enable_augmentation
        self.batch_size = batch_size
        self.dataset = dataset
        self.samples = None
        self.shuffle = shuffle
        self.n_classes = len(self.labels)
        self.n_channels = n_channels
        self.cur_epoch = 0
        self.loaded_epochs = 0
        self.epochs = epochs
        self.epoch_data = []
        self.epoch_stats = []
        self.preload = preload
        if self.preload:
            self.load_queue = multiprocessing.Queue()
        if self.preload:
            self.preloader_queue = multiprocessing.Queue(buffer_size)

        # load epoch
        self.load_next_epoch()
        self.epoch_stats.append({})
        self.epoch_data.append(([None] * len(self), [None] * len(self)))

        if self.preload:
            self.preloader_threads = [
                multiprocessing.Process(
                    target=preloader, args=(self.preloader_queue, self.load_queue, self)
                )
                for _ in range(load_threads)
            ]
            for thread in self.preloader_threads:
                thread.start()
        logging.info(
            "datagen for %s shuffle %s cap %s",
            self.dataset.name,
            self.shuffle,
            self.cap_samples,
        )

    def stop_load(self):
        if not self.preload:
            return
        for thread in self.preloader_threads:
            if hasattr(thread, "terminate"):
                thread.terminate()
            else:
                thread.exit()
        self.preloader_queue = None

    def get_epoch_predictions(self, epoch=-1):
        if self.keep_epoch:
            return self.epoch_data[epoch][1]
        return None

    def __len__(self):
        "Denotes the number of batches per epoch"

        return int(math.ceil(len(self.samples) / self.batch_size))

    def loadbatch(self, samples):
        start = time.time()
        # samples = self.samples[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self._data(samples)

        logging.debug("%s  Time to get data %s", self.dataset.name, time.time() - start)

        return X, y

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        logging.debug("%s requsting index %s", self.dataset.name, index)
        if self.keep_epoch and self.use_previous_epoch is not None:
            X = self.epoch_data[self.use_previous_epoch][0][
                index * self.batch_size : (index + 1) * self.batch_size
            ]

            y = self.epoch_data[self.use_previous_epoch][1][
                index * self.batch_size : (index + 1) * self.batch_size
            ]
        else:
            if index == 0 and self.epoch_data[self.cur_epoch][0][0] is not None:
                # when tensorflow uses model.fit it requests index 0 twice
                X = self.epoch_data[self.cur_epoch][0][index]
                y = self.epoch_data[self.cur_epoch][1][index]
            elif self.preload:
                X, y = self.preloader_queue.get()
            else:
                X, y = self.loadbatch(index)

            if self.epoch_data[self.cur_epoch][0][index] is None:
                epoch_stats = self.epoch_stats[self.cur_epoch]
                out_y = np.argmax(y, axis=1)
                indices, counts = np.unique(out_y, return_counts=True)
                for i, label_index in enumerate(indices):
                    label = self.labels[label_index]
                    count = counts[i]
                    epoch_stats.setdefault(label, 0)
                    epoch_stats[label] += count
        # always keep a copy of epoch data
        if index == 0 or (self.keep_epoch and self.use_previous_epoch is None):
            self.epoch_data[self.cur_epoch][0][index] = X
            self.epoch_data[self.cur_epoch][1][index] = y

        # can start loading next epoch of training before validation
        # if (index + 1) == len(self):
        #     self.load_next_epoch(True)
        return X, y

    def resuse_previous_epoch(self):
        """
            This is used in hyper training to speeed it up just load one epoch into memory
            and shuffle each time
        """
        if self.use_previous_epoch is None:
            # [batch, segment, height, width, rgb]
            X = [item for batch in self.epoch_data[self.cur_epoch][0] for item in batch]
            y = [item for batch in self.epoch_data[self.cur_epoch][1] for item in batch]
        else:
            # [segment, height, width, rgb]

            X = self.epoch_data[self.use_previous_epoch][0]
            y = self.epoch_data[self.use_previous_epoch][1]
        if self.shuffle:
            X = np.asarray(X)
            y = np.asarray(y)
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
        self.epoch_data[self.cur_epoch] = None
        self.epoch_data[0] = (X, y)
        self.use_previous_epoch = 0
        logging.info("Reusing previous epoch data for epoch %s", self.cur_epoch)
        self.stop_load()

    def load_next_epoch(self, reuse=False):
        if self.loaded_epochs >= self.epochs:
            return
        if self.randomize_epoch is False and reuse:
            return self.resuse_previous_epoch()

        else:
            self.samples = self.dataset.epoch_samples(
                cap_samples=self.cap_samples,
                replace=False,
                random=self.randomize_epoch,
                cap_at=self.cap_at,
                label_cap=self.label_cap,
            )
            self.samples = [sample.id for sample in self.samples]

            if self.shuffle:
                np.random.shuffle(self.samples)
        if self.preload:
            for index in range(len(self)):
                samples = self.samples[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]
                pickled_samples = pickle.dumps((self.loaded_epochs + 1, samples))
                self.load_queue.put(pickled_samples)
        self.loaded_epochs += 1

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.load_next_epoch(reuse=True)
        batches = len(self)
        if not self.keep_epoch:
            # zero last epoch
            self.epoch_data[-1] = None

        self.epoch_stats.append({})
        self.epoch_data.append(([None] * batches, [None] * batches))

        last_stats = self.epoch_stats[-1]
        logging.info("epoch ended for %s %s", self.dataset.name, last_stats)
        self.cur_epoch += 1

    def _data(self, samples, to_categorical=True):
        "Generates data containing batch_size samples"
        # Initialization
        X = np.empty((len(samples), *self.dim,))

        y = np.empty((len(samples)), dtype=int)
        data_i = 0
        if self.use_thermal:
            channel = TrackChannels.thermal
        else:
            channel = TrackChannels.filtered

        for sample in samples:
            label = self.dataset.mapped_label(sample.label)
            if label not in self.labels:
                continue
            if self.movement:
                try:
                    frames = self.dataset.fetch_track(sample.track, preprocess=False)
                except Exception as inst:
                    logging.error("Error fetching sample %s %s", sample, inst)
                    continue

                indices = np.arange(len(frames))
                np.random.shuffle(indices)
                frame_data = []
                for frame_i in indices[: sample.frames]:
                    frame_data.append(frames[frame_i].copy())

                if len(frame_data) < 5:
                    logging.error(
                        "Important frames filtered for %s %s / %s",
                        sample,
                        len(frame_data),
                        len(sample.track.important_frames),
                    )
                    continue

                # repeat some frames if need be
                if len(frame_data) < self.square_width ** 2:
                    missing = self.square_width ** 2 - len(frame_data)
                    np.random.shuffle(indices)
                    for frame_i in indices[:missing]:
                        frame_data.append(frames[frame_i].copy())
                ref = []
                regions = []
                for r in sample.track.track_bounds:
                    regions.append(tools.Rectangle.from_ltrb(*r))
                frame_data = sorted(
                    frame_data, key=lambda frame_data: frame_data.frame_number
                )

                for frame in frame_data:
                    ref.append(sample.track.frame_temp_median[frame.frame_number])

                data = preprocess_movement(
                    frames,
                    frame_data,
                    self.square_width,
                    regions,
                    channel,
                    preprocess_fn=self.model_preprocess,
                    augment=self.augment,
                    use_dots=self.use_dots,
                    reference_level=ref,
                )
            else:
                try:
                    data, label = self.dataset.fetch_sample(
                        sample, augment=self.augment, channels=channel
                    )

                    if label not in self.labels:
                        continue
                except Exception as inst:
                    logging.error("Error fetching samples %s %s", sample, inst)
                    continue

                data = preprocess_frame(
                    data, self.dim, None, self.augment, self.model_preprocess,
                )
            if data is None:
                logging.debug(
                    "error pre processing frame (i.e.max and min are the same)sample %s",
                    sample,
                )
                continue

            X[data_i] = data
            y[data_i] = self.labels.index(label)
            data_i += 1
        # remove data that was null
        X = X[:data_i]
        y = y[:data_i]
        if to_categorical:
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return np.array(X), y


# continue to read examples until queue is full
def preloader(q, load_queue, datagen):
    """ add a segment into buffer """
    logging.info(
        " -started async fetcher for %s augment=%s",
        datagen.dataset.name,
        datagen.augment,
    )
    while True:
        if not q.full():
            samples = pickle.loads(load_queue.get())
            datagen.loaded_epochs = samples[0]
            segments = []
            for sample_id in samples[1]:
                segments.append(datagen.dataset.segments_by_id[sample_id])
            q.put(datagen.loadbatch(segments))

        else:
            time.sleep(0.1)


def savemovement(data, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    r = Image.fromarray(np.uint8(data[:, :, 0] * 255))
    g = Image.fromarray(np.uint8(data[:, :, 1] * 255))
    b = Image.fromarray(np.uint8(data[:, :, 2] * 255))
    concat = np.concatenate((r, g, b), axis=1)  # horizontally
    img = Image.fromarray(np.uint8(concat))
    img.save(filename + "rgb.png")
