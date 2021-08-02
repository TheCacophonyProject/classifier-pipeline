from collections import deque
import pickle
import math
import logging
import tensorflow.keras as keras
import numpy as np
import multiprocessing
import time
import gc
from ml_tools import tools
from ml_tools.preprocess import preprocess_movement, preprocess_frame
from track.track import TrackChannels
from ml_tools.frame import Frame
from collections import Counter


from queue import Queue, Empty, Full
import threading

FRAMES_PER_SECOND = 9


class GeneartorParams:
    def __init__(self, output_dim, params):
        self.augment = params.get("augment", False)
        self.use_movement = params.get("use_movement")
        self.model_preprocess = params.get("model_preprocess")
        self.channel = params.get("channel")
        self.square_width = params.get("square_width", 5)
        self.output_dim = output_dim
        self.mvm = params.get("mvm", False)
        self.type = params.get("type", 1)
        self.segment_type = params.get("segment_type", 1)
        self.keep_edge = params.get("keep_edge", False)
        self.maximum_preload = params.get("maximum_preload", 100)


# Datagenerator consists of 3 processes
# 1: Loads the raw batch data from numpy files (Batches are fed into a queue from process 3)
# 2: Preprocess the raw batch data fed into a queue from process 1
# 3 is managed by Keras and requests preprocess batches fed into a queue by process 2


class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        dataset,
        labels,
        output_dim,
        **params,
    ):
        self.params = GeneartorParams(output_dim, params)
        self.use_previous_epoch = None
        self.labels = labels
        self.dataset = dataset
        self.epochs = params.get("epochs", 10)
        self.samples = None
        self.randomize_epoch = params.get("randomize_epoch", True)
        self.cap_at = params.get("cap_at")
        self.cap_samples = params.get("cap_samples", True)
        self.label_cap = params.get("label_cap")
        self.shuffle = params.get("shuffle", True)
        self.batch_size = params.get("batch_size", 16)
        self.epoch_samples = []
        self.sample_size = None
        self.cur_epoch = 0
        self.loaded_epochs = 0
        self.epoch_stats = []
        self.eager_load = params.get("eager_load", False)
        use_threads = False
        if use_threads:
            # This will be slower, but use less memory
            # Queue is also extremely slow so should be change to deque
            self.load_queue = Queue()
            self.preloader_queue = Queue(params.get("buffer_size", 128))
        else:
            self.load_queue = multiprocessing.Queue()
            self.preloader_queue = multiprocessing.Queue(params.get("buffer_size", 128))

        self.segments = None
        self.segments = []
        # load epoch
        self.epoch_labels = []
        self.load_next_epoch()
        self.epoch_data = []

        if use_threads:
            self.preloader_thread = threading.Thread(
                target=preloader,
                args=(
                    self.preloader_queue,
                    self.load_queue,
                    self.labels,
                    self.dataset.name,
                    self.dataset.db,
                    self.dataset.segments_by_id,
                    self.params,
                    self.dataset.label_mapping,
                    self.dataset.numpy_data,
                ),
            )
        else:
            self.preloader_thread = multiprocessing.Process(
                target=preloader,
                args=(
                    self.preloader_queue,
                    self.load_queue,
                    self.labels,
                    self.dataset.name,
                    self.dataset.db,
                    self.dataset.segments_by_id,
                    self.params,
                    self.dataset.label_mapping,
                    self.dataset.numpy_data,
                ),
            )
        self.preloader_thread.start()
        logging.info(
            "datagen for %s shuffle %s cap %s type %s epochs %s preprocess %s keep_edge %s",
            self.dataset.name,
            self.shuffle,
            self.cap_samples,
            self.params.type,
            self.epochs,
            self.params.model_preprocess,
            self.params.keep_edge,
        )

    def stop_load(self):
        if not self.load_queue:
            return
        logging.info("stopping %s", self.dataset.name)
        self.load_queue.put("STOP")
        time.sleep(4)
        if hasattr(self.preloader_thread, "terminate"):
            self.preloader_thread.terminate()
        del self.preloader_queue
        del self.preloader_thread
        del self.load_queue
        self.load_queue = None
        self.preloader_queue = None

    def get_epoch_labels(self, epoch=-1):
        return self.epoch_labels[epoch]

    def __len__(self):
        "Denotes the number of batches per epoch"

        return int(math.ceil(self.sample_size / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        start = time.time()
        if index == len(self) // 2 and self.eager_load:
            logging.info(
                "%s on epoch %s index % s loading next epoch data",
                self.dataset.name,
                self.loaded_epochs,
                index,
            )
            self.load_next_epoch()

        if index == 0 and len(self.epoch_data) > self.cur_epoch:
            # when tensorflow uses model.fit it requests index 0 twice
            X, y = self.epoch_data[self.cur_epoch]
        else:
            try:
                X, y, y_original = get_with_timeout(self.preloader_queue, 30)
            except:
                logging.error(
                    "%s error getting preloaded data",
                    self.dataset.name,
                    exc_info=True,
                )

            # always keep a copy of epoch data
            if index == 0:
                self.epoch_data.append((X, y))

            self.epoch_labels[self.cur_epoch].extend(y_original)
        return X, y

    def load_next_epoch(self):
        if self.loaded_epochs >= self.epochs:
            return
        self.epoch_labels.append([])
        logging.info("%s loading epoch %s", self.dataset.name, self.loaded_epochs)
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

        # first epoch needs sample size set, others set at epoch end
        if self.cur_epoch == 0:
            self.sample_size = len(self.samples)

        logging.info(
            "%s num of batches %s",
            self.dataset.name,
            len(self),
        )
        batches = []
        for index in range(len(self)):
            samples = self.samples[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            batches.append(samples)

        logging.info("%s adding %s", self.dataset.name, len(batches))
        if len(batches) > 0:
            pickled_batches = pickle.dumps((self.loaded_epochs + 1, batches))
            self.load_queue.put(pickled_batches)

        self.epoch_samples.append(len(self.samples))
        self.dataset.segments = []
        gc.collect()
        self.loaded_epochs += 1

    def reload_samples(self):
        logging.info("%s reloading samples", self.dataset.name)
        if self.shuffle:
            np.random.shuffle(self.samples)
            batches = []
        for index in range(len(self)):
            samples = self.samples[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            batches.append(samples)

            logging.info(self.dataset.name, "adding", len(batches))
            if len(batches) > 0:
                pickled_batches = pickle.dumps((self.loaded_epochs + 1, batches))
                self.load_queue.put(pickled_batches)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if not self.eager_load:
            self.load_next_epoch()
        self.sample_size = self.epoch_samples[self.cur_epoch]
        logging.info(
            "%s setting sample size from %s and last epoch was %s",
            self.dataset.name,
            len(self.epoch_samples),
            self.cur_epoch,
        )
        batches = len(self)

        self.epoch_data[self.cur_epoch] = None
        last_stats = self.epoch_label_count()
        logging.info("epoch ended for %s %s", self.dataset.name, last_stats)
        self.cur_epoch += 1

    def epoch_label_count(self):
        labels = self.epoch_labels[self.cur_epoch]
        return Counter(labels)


def loadbatch(labels, segments, data, params, mapped_labels):
    X, y, y_orig = _data(labels, segments, data, params, mapped_labels)
    return X, y, y_orig


def get_cached_frames(db, sample):
    track_frames = db[sample.unique_track_id]
    frames = []
    for f_i in sample.frame_indices:
        frames.append(track_frames[f_i].copy())
    return frames


def _data(labels, samples, data, params, mapped_labels, to_categorical=True):
    "Generates data containing batch_size samples"
    # Initialization
    start = time.time()
    X = np.empty(
        (
            len(samples),
            *params.output_dim,
        )
    )
    y = np.empty((len(samples)), dtype=int)
    data_i = 0
    y_original = []
    mvm = []
    for sample, frame_data in zip(samples, data):
        label = mapped_labels[sample.label]
        if label not in labels:
            continue
        if params.use_movement:
            if len(frame_data) < 5:
                logging.error(
                    "Not enough frame data for %s %s", sample, len(frame_data)
                )
                continue

            # repeat some frames if need be
            while len(frame_data) < params.square_width ** 2:
                missing = params.square_width ** 2 - len(frame_data)
                indices = np.arange(len(frame_data))
                np.random.shuffle(indices)
                for frame_i in indices[:missing]:
                    frame_data.append(frame_data[frame_i].copy())
            ref = []
            frame_data = sorted(
                frame_data, key=lambda frame_data: frame_data.frame_number
            )

            for frame in frame_data:
                ref.append(sample.frame_temp_median[frame.frame_number])
            data = preprocess_movement(
                None,
                frame_data,
                params.square_width,
                None,
                params.channel,
                preprocess_fn=params.model_preprocess,
                augment=params.augment,
                reference_level=ref,
                sample=sample,
                type=params.type,
                keep_edge=params.keep_edge,
            )
            if data is not None:
                mvm.append(sample.movement_data)
        else:

            frame = preprocess_frame(
                frame_data,
                params.augment,
                frame_sample.temp_median,
                frame_sample.velocity,
                params.output_dim,
                preprocess_fn=params.model_preprocess,
                sample=sample,
            )
        if data is None:
            logging.debug(
                "error pre processing frame (i.e.max and min are the same)sample %s",
                sample,
            )
            continue
        y_original.append(sample.label)
        X[data_i] = data
        y[data_i] = labels.index(label)
        data_i += 1
    # remove data that was null
    X = X[:data_i]
    y = y[:data_i]
    if len(X) == 0:
        logging.error("Empty length of x")

    if to_categorical:
        y = keras.utils.to_categorical(y, num_classes=len(labels))
    total_time = time.time() - start
    logging.debug(
        "augment? %s took to preprocess %s",
        params.augment,
        total_time,
    )
    if params.mvm:
        return [np.array(X), np.array(mvm)], y, y_original
    return np.array(X), y, y_original


def load_from_numpy(f, frames_by_track, tracks, channels, name):
    start = time.time()
    count = 0
    data = [None] * len(channels)
    for track_info, frame_indices, u_id, regions_by_frames in tracks:
        frame_indices.sort()
        track_data = frames_by_track.setdefault(u_id, {})
        background = track_data.get("background")
        if background is None:
            f.seek(track_info["background"])
            background = np.load(f, allow_pickle=True)
            track_data["background"] = background
        for frame_i in frame_indices:
            if frame_i in track_data:
                continue
            count += 1
            frame_info = track_info[frame_i]
            for i, channel in enumerate(channels):
                f.seek(frame_info[channel])
                channel_data = np.load(f, allow_pickle=True)
                data[i] = channel_data

            frame = Frame.from_channel(data, channels, frame_i, flow_clipped=True)
            track_data[frame_i] = frame
            frame.region = tools.Rectangle.from_ltrb(*regions_by_frames[frame_i])
            frame.filtered = frame.thermal - frame.region.subimage(background)
    logging.info("%s time to load %s frames %s", name, count, time.time() - start)
    return frames_by_track


def load_batch_frames(
    track_frames,
    numpy_meta,
    batches_by_id,
    segments_by_id,
    channels,
    name,
):
    # loads batches from numpy file, by increasing file location, into supplied track_frames dictionary
    # returns loaded batches as segments
    all_batches = []
    with numpy_meta as f:
        data_by_track = {}
        for batch in batches_by_id:
            batch_segments = []
            for s_id in batch:
                segment = segments_by_id[s_id]
                batch_segments.append(segment)
                track_segments = data_by_track.setdefault(
                    segment.unique_track_id,
                    (segment.track_info, [], segment.unique_track_id, {}),
                )
                regions_by_frames = track_segments[3]
                regions_by_frames.update(segment.track_bounds)
                track_segments[1].extend(segment.frame_indices)
            all_batches.append(batch_segments)
        # sort by position in file

        track_segments = sorted(
            data_by_track.values(),
            key=lambda track_segment: track_segment[0]["background"],
        )

        load_from_numpy(f, track_frames, track_segments, channels, name)

    return all_batches


def preloader(
    q,
    load_queue,
    labels,
    name,
    db,
    segments_by_id,
    params,
    label_mapping,
    numpy_meta,
):
    """add a segment into buffer"""
    logging.info(
        " -started async fetcher for %s augment=%s numpyfile %s preload amount %s",
        name,
        params.augment,
        numpy_meta.filename,
        params.maximum_preload,
    )
    # filtered always loaded
    channels = [TrackChannels.thermal]
    if params.type == 3:
        channels.append(TrackChannels.flow)

    epoch = 0
    # dictionary with keys track_uids and then dicitonary of frame_ids
    track_frames = {}
    batch_q = multiprocessing.Queue()

    # this thread does the data pre processing
    p_preprocess = multiprocessing.Process(
        target=process_batches,
        args=(batch_q, q, labels, params, label_mapping, name, params.maximum_preload),
    )
    p_preprocess.start()
    while True:
        try:
            item = get_with_timeout(load_queue, 30)
        except:
            logging.error("%s epoch %s error", name, epoch, exc_info=True)
            return

        if item == "STOP":
            batch_q.put("STOP")
            logging.info("%s received stop", name)
            break
        try:
            epoch, batches = pickle.loads(item)
            logging.info(
                "%s Preloader got %s batches for epoch %s",
                name,
                len(batches),
                epoch,
            )
            total = 0

            preload_amount = max(1, params.maximum_preload // 2)
            load_until = preload_amount
            loaded_up_to = 0
            while loaded_up_to < len(batches):
                if batch_q.qsize() < load_until and loaded_up_to < len(batches):
                    next_load = batches[loaded_up_to : loaded_up_to + preload_amount]
                    logging.info(
                        "%s loading more data, have segments: %s  loading %s - %s of %s qsize %s ",
                        name,
                        batch_q.qsize(),
                        loaded_up_to,
                        loaded_up_to + len(next_load),
                        len(batches),
                        q.qsize(),
                    )
                    batch_data = load_batch_frames(
                        track_frames,
                        numpy_meta,
                        next_load,
                        segments_by_id,
                        channels,
                        name,
                    )
                    loaded_batches = []
                    for segments in batch_data:
                        segment_data = []
                        for seg in segments:
                            frame_data = get_cached_frames(track_frames, seg)
                            segment_data.append(frame_data)
                        loaded_batches.append((segments, segment_data))
                    batch_q.put(loaded_batches)
                    # put all at once save queue overheader
                    del track_frames
                    track_frames = {}
                    gc.collect()
                    if loaded_up_to == 0:
                        load_until = 2
                    loaded_up_to = loaded_up_to + len(next_load)
                    logging.info(
                        "%s loaded more data qsize is %s loaded up to %s load until %s",
                        name,
                        batch_q.qsize(),
                        loaded_up_to,
                        load_until,
                    )
                    total += 1
                else:
                    time.sleep(2)

            logging.info("%s loaded epoch %s", name, epoch)
        except Exception as inst:
            logging.error("%s epoch %s error %s", name, epoch, inst, exc_info=True)
            pass


def process_batches(batch_queue, q, labels, params, label_mapping, name, stop_at):
    # runs through loaded frames and applies appropriate prperocessing and then sends them to queue for training
    total = 0
    b_total = 0
    g_total = 0
    p_total = 0
    while True:
        while stop_at is not None and q.qsize() > stop_at:
            logging.debug(" %s Q has maximum preload", name)
            time.sleep(2)
        if total % 50 == 0:
            logging.info(
                "Loaded %s batches %s to load and %s waiting to be trained have %s b %s g %s p %s ",
                total,
                name,
                batch_queue.qsize(),
                q.qsize(),
                b_total,
                g_total,
                p_total,
            )
            b_total = 0
            g_total = 0
            p_total = 0
        g_time = time.time()
        batches = None
        batches = get_with_timeout(batch_queue, 30)
        if batches == "STOP":
            logging.info("%s process batch thread received stop", name)
            return
        g_total += time.time() - g_time
        for segments, data in batches:
            b_time = time.time()
            batch_data = loadbatch(labels, segments, data, params, label_mapping)
            b_total += time.time() - b_time
            p_time = time.time()
            try:
                put_with_timeout(q, batch_data, 30)
                total += 1
            except Exception as e:
                logging.error(
                    "%s batch Put error",
                    name,
                    exc_info=True,
                )
                raise e
            p_total += time.time() - p_time
        while batch_queue.qsize() == 0:
            logging.info(" %s loaded all the data", name)
            time.sleep(10)


# Found hanging problems with blocking forever so using this as workaround
# keeps trying to put data in queue until complete
def put_with_timeout(queue, data, timeout):
    while True:
        try:
            queue.put(data, block=True, timeout=timeout)
            break
        except (Full):
            pass
        except Exception as e:
            raise e


# keeps trying to get data in queue until complete
def get_with_timeout(queue, timeout):
    while True:
        try:
            queue_data = queue.get(block=True, timeout=timeout)
            break
        except (Empty):
            pass
        except Exception as e:
            raise e
    return queue_data
