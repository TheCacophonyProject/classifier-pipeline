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

FRAME_SIZE = 48


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
        sequence_size=27,
        lstm=False,
        use_thermal=False,
        use_filtered=False,
        buffer_size=64,
        epochs=10,
        load_threads=1,
        preload=True,
        resample=False,
    ):
        self.resample = resample

        self.labels = labels
        self.model_preprocess = model_preprocess
        self.use_thermal = use_thermal
        self.use_filtered = use_filtered
        self.lstm = lstm
        # default
        if not self.use_thermal and not self.use_filtered and not self.lstm:
            self.use_thermal = True
        self.dim = dim
        self.augment = dataset.enable_augmentation
        self.batch_size = batch_size
        self.dataset = dataset

        if self.lstm:
            self.sequence_size = sequence_size
            self.size = dataset.rows
        else:
            self.size = len(dataset.frame_samples)

        self.indexes = np.arange(self.size)
        self.shuffle = shuffle
        self.n_classes = len(self.labels)
        self.n_channels = n_channels
        self.cur_epoch = 0
        self.epochs = epochs

        self.preload = preload
        if self.preload:
            self.load_queue = multiprocessing.Queue(len(self) + 1)
        self.on_epoch_end(load=preload)

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

    # get all data
    def get_data(self, catog=False):
        X, y, _ = self._data(self.indexes, to_categorical=catog)
        return X, y

    def __len__(self):
        "Denotes the number of batches per epoch"

        return int(np.floor(self.dataset.rows / self.batch_size))

    def loadbatch(self, index):
        start = time.time()
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, y, clips = self._data(indexes)
        logging.debug("Time to get data %s", time.time() - start)

        return X, y

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        if self.preload:
            X, y = self.preloader_queue.get()
        else:
            X, y = self.loadbatch(index)
        return X, y

    def on_epoch_end(self, load=True):
        "Updates indexes after each epoch"
        if self.resample:
            self.dataset.resample("wallaby")
            self.size = len(self.dataset.segments)
            self.indexes = np.arange(self.dataset.rows)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        if load:
            # for some reason it always requests 0 twice
            self.load_queue.put(0)
            for i in range(len(self)):
                self.load_queue.put(i)

        self.cur_epoch += 1

    def _data(self, indexes, to_categorical=True):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.lstm:
            X = np.empty((len(indexes), self.sequence_size, *self.dim))
        else:
            X = np.empty((len(indexes), *self.dim))

        y = np.empty((len(indexes)), dtype=int)
        clips = []
        if self.use_thermal:
            channels = TrackChannels.thermal
        else:
            channels = TrackChannels.filtered
        # Generate data
        data_i = 0
        for index in indexes:
            segment_i = index
            if self.lstm:
                segment = self.dataset.segments[index]
                data = self.dataset.fetch_segment(segment, augment=self.augment)
                if self.dataset.label_mapping:
                    label = self.dataset.label_mapping[segment.label]
                else:
                    label = segment.label
                frame = segment
            else:
                frame = self.dataset.frame_samples[segment_i]
                data, label = self.dataset.fetch_frame(frame, channels=channels)
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
                data = self.dataset.fetch_segment(segment, augment=self.augment)
                # logging.error(
                #     "error pre processing frame (i.e.max and min are the same) clip %s track %s frame %s",
                #     frame.clip_id,
                #     frame.track_id,
                #     frame.frame_num,
                # )
                continue

            X[data_i,] = data
            y[data_i] = self.labels.index(label)
            clips.append(frame)
            data_i += 1
        # print(data_i, len(y), len(y[:data_i]))
        X = X[:data_i]
        y = y[:data_i]
        if to_categorical:
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y, clips


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
    if augment:
        percent = random.randint(0, 2)
    else:
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
    if augment:
        percent = random.randint(0, 2)
    else:
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
                "loaded batch %s %s %s",
                dataset.cur_epoch,
                dataset.dataset.name,
                batch_i,
            )
        else:
            time.sleep(0.1)


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
    raise "DONE"
