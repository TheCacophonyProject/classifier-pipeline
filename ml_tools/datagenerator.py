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
from ml_tools.dataset import Preprocessor, filtered_is_valid
from ml_tools import tools
from scipy import ndimage

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
        randomize_epoch=True,
        cap_samples=True,
        cap_at=None,
        type=0,
        square_width=5,
        label_cap=None,
    ):
        self.label_cap = label_cap
        self.type = type
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
        self.lstm = lstm

        if not self.use_thermal and not self.use_filtered and not self.lstm:
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
        self.on_epoch_end()
        self.cur_epoch = 0

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
            "datagen for %s shuffle %s cap %s type %s",
            self.dataset.name,
            self.shuffle,
            self.cap_samples,
            self.type,
        )

    def stop_load(self):
        if not self.preload:
            return
        for thread in self.preloader_threads:
            if hasattr(thread, "terminate"):
                # note this will corrupt the queue, so reset it
                thread.terminate()
            else:
                thread.exit()
        self.preloader_queue = None

    def get_epoch_predictions(self, epoch=-1):
        if self.keep_epoch:
            return self.epoch_data[epoch][1]
        return None

    # get all data
    def get_data(self, epoch=0, catog=False):
        X, y = self._data(self.samples, to_categorical=catog)
        return X, y

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
            if index == 0 and self.epoch_data[self.cur_epoch][0][index] is not None:
                # when tensorflow fits it requests index 0 twice
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

    def load_next_epoch(self, reuse=False):

        if self.loaded_epochs >= self.epochs:
            return
        if self.randomize_epoch is False and reuse:
            # [batch, segment, height, width, rgb]
            # or [segment, height, width, rgb]
            if self.use_previous_epoch is None:
                # in batches
                X = [
                    item
                    for batch in self.epoch_data[self.cur_epoch][0]
                    for item in batch
                ]
                y = [
                    item
                    for batch in self.epoch_data[self.cur_epoch][1]
                    for item in batch
                ]
            else:
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
            return
        else:
            self.samples = self.dataset.epoch_samples(
                cap_samples=self.cap_samples,
                replace=False,
                random=self.randomize_epoch,
                cap_at=self.cap_at,
                label_cap=self.label_cap,
            )

            #
            if self.shuffle:
                np.random.shuffle(self.samples)
        self.samples = [sample.id for sample in self.samples]

        if self.preload:
            # DEBUGGING GP

            tracks_by_label = {}
            for sample_id in self.samples:
                sample = self.dataset.segments_by_id[sample_id]
                track = sample.track
                label_tracks = tracks_by_label.setdefault(track.label, [])
                label_tracks.append(track.unique_id)
            for key, value in tracks_by_label.items():
                ids = list(value)
                ids.sort()

                logging.info(
                    "%s samples for %s %s %s",
                    self.dataset.name,
                    key,
                    self.loaded_epochs + 1,
                    len(ids),
                    # ids,
                )
            for index in range(len(self)):
                samples = self.samples[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]
                pickled_samples = pickle.dumps((self.loaded_epochs + 1, samples))
                self.load_queue.put(pickled_samples)
        self.loaded_epochs += 1

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.load_next_epoch(reuse=len(self.epoch_data) > 0)
        batches = len(self)
        if not self.keep_epoch and len(self.epoch_data) > 0:
            # zero last epoch
            self.epoch_data[-1] = None
        last_stats = None
        if len(self.epoch_stats) > 0:
            last_stats = self.epoch_stats[-1]
        self.epoch_stats.append({})
        self.epoch_data.append(([None] * batches, [None] * batches))

        logging.info("epoch ended for %s %s", self.dataset.name, last_stats)
        self.cur_epoch += 1

    def _data(self, samples, to_categorical=True):
        "Generates data containing batch_size samples"
        # Initialization
        if self.lstm:
            X = np.empty((len(samples), samples[0].frames, *self.dim))
        else:
            X = np.empty(
                (
                    len(samples),
                    *self.dim,
                )
            )

        y = np.empty((len(samples)), dtype=int)
        # Generate data
        data_i = 0
        if self.use_thermal:
            channel = TrackChannels.thermal
        else:
            channel = TrackChannels.filtered

        for sample in samples:
            if self.movement:
                try:
                    if self.type >= 3:
                        data = self.dataset.fetch_track(sample.track, preprocess=False)

                    else:
                        data = self.dataset.fetch_segment(sample, preprocess=False)
                except Exception as inst:
                    logging.error("Error fetching sample %s %s", sample, inst)
                    continue
                label = self.dataset.mapped_label(sample.label)
                if self.type == 3:
                    segment_data = data[
                        sample.start_frame : sample.start_frame + sample.frames
                    ]
                    ref = sample.track.frame_temp_median[
                        sample.start_frame : sample.start_frame + len(segment_data)
                    ]
                elif self.type >= 4:
                    frames = np.random.choice(
                        sample.track.important_frames,
                        min(sample.frames, len(sample.track.important_frames)),
                        replace=False,
                    )
                    if len(frames) < 5:
                        logging.error(
                            "Important frames filtered for %s %s / %s",
                            sample,
                            len(frames),
                            len(sample.track.important_frames),
                        )
                        continue

                    # repeat some frames if need be
                    if len(frames) < self.square_width ** 2:
                        missing = self.square_width ** 2 - len(frames)
                        frames.extend(
                            np.random.choice(
                                sample.track.important_frames,
                                min(missing, len(sample.track.important_frames)),
                                replace=False,
                            )
                        )
                    segment_data = []
                    ref = []

                    frames = [frame.frame_num for frame in frames]
                    # sort??? or rather than random just apply a 1 second step
                    # frames = sample.frame_indices
                    frames.sort()
                    for frame_num in frames:
                        segment_data.append(data[frame_num])
                        ref.append(sample.track.frame_temp_median[frame_num])
                    segment = (segment_data, ref)
                else:
                    segment_data = data
                    ref = sample.track.frame_temp_median[
                        sample.start_frame : sample.start_frame + len(data)
                    ]
                    segment = (segment_data, ref)

                if self.type < 3:
                    regions = sample.track.track_bounds[
                        sample.start_frame : sample.start_frame + sample.frames
                    ]
                else:
                    regions = sample.track.track_bounds

                data = preprocess_movement(
                    data,
                    segment,
                    self.square_width,
                    regions,
                    channel,
                    self.model_preprocess,
                    sample,
                    self.dataset.name,
                    self.type,
                    augment=self.augment,
                    epoch=self.loaded_epochs,
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
                if self.lstm:
                    data = preprocess_lstm(
                        data,
                        self.dim,
                        channel,
                        self.augment,
                        self.model_preprocess,
                    )
                else:
                    data = preprocess_frame(
                        data,
                        self.dim,
                        None,
                        self.augment,
                        self.model_preprocess,
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


def resize(image, dim):
    image = convert(image)
    image = tf.image.resize(image, dim[0], dim[1])
    return image.numpy()


def resize_cv(image, dim, interpolation=cv2.INTER_LINEAR, extra_h=0, extra_v=0):
    return cv2.resize(
        image,
        dsize=(dim[0] + extra_h, dim[1] + extra_v),
        interpolation=interpolation,
    )


def convert(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def augement_frame(frame, dim):
    frame = resize_cv(
        frame,
        dim,
        extra_h=random.randint(0, int(FRAME_SIZE * 0.05)),
        extra_v=random.randint(0, int(FRAME_SIZE * 0.05)),
    )

    image = convert(frame)
    # image = tf.image.resize(
    #     image, [FRAME_SIZE + random.randint(0, 4), FRAME_SIZE + random.randint(0, 4)],
    # )  # Add 6 pixels of padding
    image = tf.image.random_crop(
        image, size=[dim[0], dim[1], 3]
    )  # Random crop back to 48x 48
    # if random.random() > 0.50:
    #     image = tf.image.rot90(image)
    if random.random() > 0.50:
        image = tf.image.flip_left_right(image)

        # maybes thisd should only be sometimes, as otherwise our validation set
    # if random.random() > 0.20:
    image = tf.image.random_contrast(image, 0.8, 1.2)
    # image = tf.image.random_brightness(image, max_delta=0.05)  # Random brightness
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)
    return image.numpy()


def square_clip(data, square_width, type=None, augment=False):
    # lay each frame out side by side in rows
    frame_size = Preprocessor.FRAME_SIZE
    background = np.zeros((square_width * frame_size, square_width * frame_size))

    i = 0
    success = False
    for x in range(square_width):
        for y in range(square_width):
            if i >= len(data):
                # if type >= 4:
                #     frame_i = random.randint(0, len(data) - 1)
                #     frame = data[frame_i]
                # else:
                frame = data[-1]
            else:
                frame = data[i]
            frame, norm_success = normalize(frame)

            if not norm_success:
                continue
            success = True
            background[
                x * frame_size : (x + 1) * frame_size,
                y * frame_size : (y + 1) * frame_size,
            ] = np.float32(frame)
            i += 1

    return background, success


def square_clip_flow(data_flow_h, data_flow_v, square_width, type=None):
    # lay each frame out side by side in rows
    frame_size = Preprocessor.FRAME_SIZE
    background = np.zeros((square_width * frame_size, square_width * frame_size))

    i = 0
    success = False
    for x in range(square_width):
        for y in range(square_width):
            if i >= len(data_flow_h):
                if type >= 4:
                    frame_i = random.randint(0, len(data_flow_h) - 1)
                    flow_h = data_flow_h[frame_i]
                    flow_v = data_flow_v[frame_i]

                else:
                    flow_v = data_flow_v[-1]
                    flow_h = data_flow_h[-1]

            else:
                flow_v = data_flow_v[i]
                flow_h = data_flow_h[i]

            flow_magnitude = (
                np.linalg.norm(np.float32([flow_h, flow_v]), ord=2, axis=0) / 4.0
            )
            frame, norm_success = normalize(flow_magnitude)

            if not norm_success:
                continue
            success = True
            background[
                x * frame_size : (x + 1) * frame_size,
                y * frame_size : (y + 1) * frame_size,
            ] = np.float32(frame)
            i += 1

    return background, success


def dots_movement(
    frames,
    regions,
    label=None,
    dim=None,
    channel=TrackChannels.filtered,
    require_movement=False,
    use_mask=False,
    augment=False,
):
    """Return 2 images describing the movement, one has dots representing
    the centre of mass, the other is a collage of all frames
    """
    if len(frames) == 0:
        return
    channel = TrackChannels.filtered
    i = 0
    overlay = np.zeros((126, 166))
    dots = np.zeros((dim))

    prev = None
    prev_overlay = None
    line_colour = 60
    dot_colour = 120
    img = Image.fromarray(np.uint8(dots))  # ignore alpha

    d = ImageDraw.Draw(img)
    # draw movment lines and draw frame overlay
    center_distance = 0
    min_distance = 2
    for i, frame in enumerate(frames):
        if isinstance(regions[i], tools.Rectangle):
            region = regions[i]
        else:
            region = tools.Rectangle.from_ltrb(*regions[i])
        x = int(region.mid_x)
        y = int(region.mid_y)

        # writing dot image
        if prev is not None:
            d.line(prev + (x, y), fill=line_colour, width=1)
        prev = (x, y)

        # writing overlay image
        if require_movement and prev_overlay:
            center_distance = tools.eucl_distance(
                prev_overlay,
                (
                    x,
                    y,
                ),
            )

        if (
            prev_overlay is None or center_distance > min_distance
        ) or not require_movement:
            if filtered_is_valid(frame, label):
                if use_mask:
                    frame = frame[channel] * (frame[TrackChannels.mask] + 0.5)
                else:
                    frame = frame[channel]
                if augment:
                    frame, region = crop_region(frame, region)
                frame = np.float32(frame)
                if augment:
                    contrast_adjust = tools.random_log(0.9, (1 / 0.9))
                    frame *= contrast_adjust
                subimage = region.subimage(overlay)
                subimage[:, :] += frame
                center_distance = 0
                min_distance = pow(region.width / 2.0, 2)
                prev_overlay = (x, y)
        prev = (x, y)

    # then draw dots so they go over the top
    for i, frame in enumerate(frames):
        if isinstance(regions[i], tools.Rectangle):
            region = regions[i]
        else:
            region = tools.Rectangle.from_ltrb(*regions[i])
        x = int(region.mid_x)
        y = int(region.mid_y)
        d.point((x, y), fill=dot_colour)

    return np.array(img), overlay


def crop_region(frame, region):
    # randomly crop frame, and return updated region
    max_height_offset = int(np.clip(region.height * 0.1, 1, 2))
    max_width_offset = int(np.clip(region.width * 0.1, 1, 2))
    top_offset = random.randint(0, max_height_offset)
    bottom_offset = random.randint(0, max_height_offset)
    left_offset = random.randint(0, max_width_offset)
    right_offset = random.randint(0, max_width_offset)

    region.x += left_offset
    region.y += top_offset
    region.width -= left_offset + right_offset
    region.height -= top_offset + bottom_offset
    frame = frame[
        top_offset : region.height + top_offset,
        left_offset : region.width + left_offset,
    ]

    return frame, region


def preprocess_movement(
    data,
    segment,
    square_width,
    regions,
    channel,
    preprocess_fn=None,
    sample=None,
    dataset=None,
    type=0,
    augment=False,
    epoch=0,
):
    segment, flipped = Preprocessor.apply(*segment, augment=augment, default_inset=0)
    segment = segment[:, channel]
    # as long as one frame is fine
    square, success = square_clip(segment, square_width, type, augment=augment)
    if not success:
        return None

    dots, overlay = dots_movement(
        data,
        regions,
        label=sample.label if sample else None,
        dim=square.shape,
        channel=channel,
        require_movement=True,
        use_mask=False,
        augment=augment,
    )

    dots = dots / 255
    overlay, success = normalize(overlay)
    if augment:
        extra_h = random.randint(0, 24) - 12
        extra_v = random.randint(0, 24) - 12
        overlay = resize_cv(overlay, overlay.shape, extra_h=extra_h, extra_v=extra_v)
        # extra_v = 12
        degrees = random.randint(0, 24) - 12

        overlay = ndimage.rotate(
            overlay, degrees, order=1, reshape=False, mode="nearest"
        )
        # overlay = tfa.image.rotate(
        #     overlay, 90 * math.pi / 180, interpolation="BILINEAR"
        # )
    if not success:
        return None
    if flipped:
        overlay = np.flip(overlay, axis=1)
        dots = np.flip(dots, axis=1)

    height, width = overlay.shape
    data = np.empty((square.shape[0], square.shape[1], 3))
    if type == 4:
        data[:, :, 0] = square
        data[:, :, 1] = square
        data[:, :, 2] = square
    elif type == 11:
        data[:, :, 0] = square
        square_one, success = square_clip(segment[25:], square_width, type)
        data[:, :, 1] = square_one
        data[:, :, 2] = overlay
    elif type >= 9:
        data[:, :, 0] = square
        data[:, :, 1] = np.zeros((dots.shape))
        data[:, :, 2] = np.zeros((dots.shape))
        data[:, :, 2][:height, :width] = overlay
    else:
        data[:, :, 0] = square
        data[:, :, 1] = dots
        data[:, :, 2] = np.zeros((dots.shape))
        data[:, :, 2][:height, :width] = overlay

    # # #
    # savemovement(
    #     data,
    #     "samples/{}/{}/{}-{}-{}-{}".format(
    #         dataset,
    #         epoch,
    #         sample.label,
    #         sample.track.clip_id,
    #         sample.track.track_id,
    #         flipped,
    #     ),
    # )

    if preprocess_fn:
        for i, frame in enumerate(data):
            frame = frame * 255
            data[i] = preprocess_fn(frame)
    return data


def preprocess_lstm(
    data,
    output_dim,
    channel,
    augment=False,
    preprocess_fn=None,
):

    data = data[:, channel]
    # normalizes data, constrast stretch good or bad?
    # if augment:
    #     percent = random.randint(0, 2)
    # else:
    data, success = normalize(data)
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
    channel,
    augment=False,
    preprocess_fn=None,
):
    if channel is not None:
        data = data[channel]
    data, success = normalize(data)
    np.clip(data, a_min=0, a_max=None, out=data)
    data = data[..., np.newaxis]
    data = np.repeat(data, 3, axis=2)

    if augment:
        data = augement_frame(data, output_dim)
        data = np.clip(data, a_min=0, a_max=None, out=data)
    else:
        data = resize_cv(data, output_dim)
        data = convert(data)

    # pre proce expects values in range 0-255
    if preprocess_fn:
        data = data * 255
        data = preprocess_fn(data)
    return data


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


def normalize(data, min=None, max=None, new_max=1):
    if max is None:
        max = np.amax(data)
    if min is None:
        min = np.amin(data)
    if max == min:
        if max == 0:
            return np.zeros((data.shape)), False
        return data / max, False
    data -= min
    data = data / (max - min) * new_max
    return data, True


def savemovement(data, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    r = Image.fromarray(np.uint8(data[:, :, 0] * 255))
    g = Image.fromarray(np.uint8(data[:, :, 1] * 255))
    b = Image.fromarray(np.uint8(data[:, :, 2] * 255))
    concat = np.concatenate((r, g, b), axis=1)  # horizontally
    img = Image.fromarray(np.uint8(concat))
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
