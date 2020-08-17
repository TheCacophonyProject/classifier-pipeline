from PIL import Image, ImageDraw, ImageFont, ImageColor

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
from ml_tools.dataset import Preprocessor

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
        lstm=False,
        use_thermal=False,
        use_filtered=False,
        buffer_size=128,
        epochs=10,
        load_threads=1,
        preload=True,
        use_movement=False,
        balance_labels=True,
        keep_epoch=False,
    ):
        self.use_previous_epoch = None
        self.keep_epoch = keep_epoch
        self.balance_labels = balance_labels

        self.labels = labels
        self.model_preprocess = model_preprocess
        self.use_thermal = use_thermal
        self.use_filtered = use_filtered
        self.lstm = lstm
        # default
        if not self.use_thermal and not self.use_filtered and not self.lstm:
            self.use_thermal = True
        if use_movement:
            self.movement = use_movement
            self.square = int(math.sqrt(round(dataset.segment_length * 9)))

        self.dim = dim
        self.augment = dataset.enable_augmentation
        self.batch_size = batch_size
        self.dataset = dataset

        self.samples = None
        self.shuffle = shuffle
        self.n_classes = len(self.labels)
        self.n_channels = n_channels
        self.cur_epoch = 0
        self.epochs = epochs
        self.epoch_data = []
        self.preload = preload
        if self.preload:
            self.load_queue = multiprocessing.Queue()
        self.load_next_epoch()
        self.on_epoch_end()

        if self.preload:
            self.preloader_queue = multiprocessing.Queue(buffer_size)
            self.preloader_stop_flag = False

            self.preloader_threads = [
                multiprocessing.Process(
                    target=preloader, args=(self.preloader_queue, self.load_queue, self)
                )
                for _ in range(load_threads)
            ]
            for thread in self.preloader_threads:
                thread.start()

    def stop_load(self):
        if not self.preload:
            return
        self.preloader_stop_flag = True
        for thread in self.preloader_threads:
            if hasattr(thread, "terminate"):
                # note this will corrupt the queue, so reset it
                thread.terminate()
                self.preloader_queue = None
            else:
                thread.exit()

    def get_epoch_predictions(self, epoch):
        if self.keep_epoch:
            print("getting predictions", epoch)
            return self.epoch_data[epoch][1]
        return none

    # get all data
    def get_data(self, epoch=0, catog=False):
        X, y = self._data(self.samples, to_categorical=catog)
        return X, y

    def __len__(self):
        "Denotes the number of batches per epoch"

        return int(len(self.samples) / self.batch_size)

    def loadbatch(self, index):
        start = time.time()
        samples = self.samples[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self._data(samples)
        logging.debug("Time to get data %s", time.time() - start)

        return X, y

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        logging.debug("%s requsting index %s", self.dataset.name, index)
        if self.keep_epoch and self.use_previous_epoch is not None:
            return (
                self.epoch_data[self.use_previous_epoch][0],
                self.epoch_data[self.use_previous_epoch][1],
            )
        if self.preload:
            X, y = self.preloader_queue.get()
        else:
            X, y = self.loadbatch(index)
        if self.keep_epoch:
            self.epoch_data[self.cur_epoch - 1][0][index] = X
            self.epoch_data[self.cur_epoch - 1][1][index] = y
            # (X, y))
        if (index + 1) == len(self):
            self.load_next_epoch()
        return X, y

    def load_next_epoch(self):
        self.samples = self.dataset.epoch_samples(replace=False)
        if self.shuffle:
            np.random.shuffle(self.samples)

        # for some reason it always requests 0 twice
        self.load_queue.put(0)
        for i in range(len(self)):
            self.load_queue.put(i)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        batches = len(self)
        self.epoch_data.append(([None] * batches, [None] * batches))
        logging.debug("epoch ended for %s", self.dataset.name)

        # self.load_next_epoch()
        self.cur_epoch += 1

    def square_clip(self, data):
        i = 0
        frame_size = Preprocessor.FRAME_SIZE
        background = np.zeros((self.square * frame_size, self.square * frame_size))
        for x in range(self.square):
            for y in range(self.square):
                i += 1
                if i >= len(data):
                    frame = data[-1]
                else:
                    frame = data[i]
                background[
                    x * frame_size : (x + 1) * frame_size,
                    y * frame_size : (y + 1) * frame_size,
                ] = np.float32(frame)
        return background

    def _data(self, samples, to_categorical=True):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        # if self.lstm:
        #     X = []np.empty((len(samples), self.sequence_size, *self.dim))
        # else:
        X = []
        # np.empty((len(samples), *self.dim))

        y = np.empty((len(samples)), dtype=int)
        if self.use_thermal:
            channels = TrackChannels.thermal
        else:
            channels = TrackChannels.filtered
        # Generate data
        data_i = 0
        for sample in samples:
            if not self.movement:
                data, label = self.dataset.fetch_sample(
                    sample, augment=self.augment, channels=channels
                )

                if label not in self.labels:
                    continue
            if self.lstm:
                data = preprocess_lstm(
                    data,
                    self.dim,
                    self.use_thermal,
                    self.augment,
                    self.model_preprocess,
                    filter_channels=True,
                )
            elif self.movement:
                data = self.dataset.fetch_segment(
                    sample, augment=self.augment, preprocess=False
                )
                label = self.dataset.mapped_label(sample.label)
                if self.use_thermal:
                    channel = TrackChannels.thermal
                else:
                    channel = TrackChannels.filtered

                segment = Preprocessor.apply(
                    data,
                    sample.track.frame_temp_median[
                        sample.start_frame : sample.start_frame + len(data)
                    ],
                    sample.track.frame_velocity[
                        sample.start_frame : sample.start_frame + len(data)
                    ],
                    augment=self.augment,
                )
                segment = segment[:, channel]

                dots, overlay = self.dataset.movement(
                    sample.start_frame, sample.track, data
                )
                start = time.time()
                square = self.square_clip(segment)
                max = np.amax(square)
                min = np.amin(square)
                if max == min:
                    continue
                square -= min
                square = square / (max - min)

                np.clip(square, a_min=0, a_max=None, out=square)

                dots = dots / 255
                overlay = overlay / np.amax(overlay)

                data = np.empty((square.shape[0], square.shape[1], 3))

                data[:, :, 0] = square
                data[:, :, 1] = dots
                data[:, :, 2] = overlay
                # print("max data", np.amax(square), np.amax(dots), np.amax(overlay))
                data = preprocess_movement(
                    data,
                    self.dim,
                    self.use_thermal,
                    self.augment,
                    self.model_preprocess,
                    filter_channels=False,
                )
                savemovement(
                    data,
                    "samples/{}-{}".format(sample.track.unique_id, sample.start_frame),
                )
            else:
                data = preprocess_frame(
                    data,
                    self.dim,
                    self.use_thermal,
                    self.augment,
                    self.model_preprocess,
                    filter_channels=False,
                )
            if data is None:
                logging.error(
                    "error pre processing frame (i.e.max and min are the same)sample %s",
                    sample,
                )
                continue

            X.append(data)
            y[data_i] = self.labels.index(label)
            data_i += 1
        # print(data_i, len(y), len(y[:data_i]))
        X = X[:data_i]
        y = y[:data_i]
        if to_categorical:
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return np.array(X), y


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


def preprocess_movement(
    data,
    output_dim,
    use_thermal=True,
    augment=False,
    preprocess_fn=None,
    filter_channels=True,
):

    if preprocess_fn:
        for i, frame in enumerate(data):

            frame = frame * 255
            data[i] = preprocess_fn(frame)
    return data


def preprocess_lstm(
    data,
    output_dim,
    use_thermal=True,
    augment=False,
    preprocess_fn=None,
    filter_channels=True,
):
    if filter_channels:
        if use_thermal:
            channel = TrackChannels.thermal
        else:
            channel = TrackChannels.filtered
        data = data[:, channel]
    # normalizes data, constrast stretch good or bad?
    # if augment:
    #     percent = random.randint(0, 2)
    # else:
    percent = 0
    max = int(np.percentile(data, 100 - percent))
    min = int(np.percentile(data, percent))
    if max == min:
        #     print("max and min are same")
        return None
    data -= min
    data = data / (max - min)
    np.clip(data, a_min=0, a_max=None, out=data)
    data = data[..., np.newaxis]

    # data = np.transpose(data, (2, 3, 0))
    data = np.repeat(data, output_dim[2], axis=3)
    # pre proce expects values in range 0-255
    if preprocess_fn:
        for i, frame in enumerate(data):

            frame = frame * 255
            data[i] = preprocess_fn(frame)
    return data


def preprocess_frame(
    data,
    output_dim,
    use_thermal=True,
    augment=False,
    preprocess_fn=None,
    filter_channels=True,
):

    if filter_channels:
        if use_thermal:
            channel = TrackChannels.thermal
        else:
            channel = TrackChannels.filtered
        data = data[channel]

    # normalizes data, constrast stretch good or bad?
    # if augment:
    #     percent = random.randint(0, 2)
    # else:
    #     percent = 0
    percent = 0
    max = int(np.percentile(data, 100 - percent))
    min = int(np.percentile(data, percent))
    if max == min:
        return None

    data -= min
    data = data / (max - min)
    np.clip(data, a_min=0, a_max=None, out=data)

    data = data[..., np.newaxis]
    data = np.repeat(data, output_dim[2], axis=2)

    if augment:
        data = augement_frame(data, output_dim)
        data = np.clip(data, a_min=0, a_max=None, out=data)
    else:
        data = reisze_cv(data, output_dim)
        data = convert(data)

    # pre proce expects values in range 0-255
    if preprocess_fn:
        data = data * 255
        data = preprocess_fn(data)
    return data


# continue to read examples until queue is full
def preloader(q, load_queue, dataset):
    """ add a segment into buffer """
    logging.info(
        " -started async fetcher for %s augment=%s",
        dataset.dataset.name,
        dataset.augment,
    )
    while not dataset.preloader_stop_flag:
        if not q.full():
            batch_i = load_queue.get()
            q.put(dataset.loadbatch(batch_i))
            if batch_i + 1 == len(dataset):
                dataset.cur_epoch += 1
            logging.debug(
                "Preloader %s loaded batch epoch %s batch %s",
                dataset.dataset.name,
                dataset.cur_epoch,
                batch_i,
            )
        else:
            time.sleep(0.1)


def saveimages(X, filename="testimage"):
    fig = plt.figure(figsize=(100, 100))
    rows = int(np.ceil(len(X) / 20))
    for x_i, x_img in enumerate(X):
        axes = fig.add_subplot(rows, 20, x_i + 1)
        plt.axis("off")
        x_img = np.asarray(x_img)
        if len(x_img.shape) == 2:
            x_img = x_img[..., np.newaxis]
            x_img = np.repeat(x_img, 3, axis=2)

        img = plt.imshow(tf.keras.preprocessing.image.array_to_img(x_img))
        img.set_cmap("hot")

    plt.savefig("{}.png".format(filename), bbox_inches="tight")
    plt.close(fig)


def normalize(data, new_max=1):
    max = np.amax(data)
    min = np.amin(data)
    data -= min
    data = data / (max - min) * new_max
    return data


def savemovement(data, filename):
    r = Image.fromarray(np.uint8(data[:, :, 0]))
    g = Image.fromarray(np.uint8(data[:, :, 1]))
    b = Image.fromarray(np.uint8(data[:, :, 2]))
    normalize(r, 255)
    normalize(g, 255)
    normalize(b, 255)
    concat = np.concatenate((r, g, b), axis=1)  # horizontally
    img = Image.fromarray(np.uint8(concat))
    d = ImageDraw.Draw(img)
    img.save(filename + "rgb.png")


def savebatch(X, y):
    fig = plt.figure(figsize=(48, 48))
    for i in range(len(X)):
        if i >= 19:
            break

        for x_i, img in enumerate(X[i]):
            axes = fig.add_subplot(20, 27, i * 27 + x_i + 1)
            axes.set_title(
                "{} ".format(
                    y[i]
                    # , clips[i].clip_id, clips[i].track_id, clips[i].frame_num,
                )
            )
            plt.axis("off")

            img = plt.imshow(tf.keras.preprocessing.image.array_to_img(img))
            img.set_cmap("hot")

    plt.savefig("testimage.png", bbox_inches="tight")
    plt.close(fig)
    print("saved image")
    # raise "DONE"
