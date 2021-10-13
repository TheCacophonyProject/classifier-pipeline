import math
import logging
import tensorflow.keras as keras
import numpy as np
import multiprocessing
import time
import gc
from ml_tools.preprocess import preprocess_movement, preprocess_frame, FrameTypes
from ml_tools.frame import TrackChannels
from ml_tools.frame import Frame
from collections import Counter, deque
from queue import Empty, Full
import threading
from ml_tools.mplogs import worker_configurer
import psutil
import os
import traceback
import sys

FRAMES_PER_SECOND = 9


class GeneartorParams:
    def __init__(self, output_dim, params):
        self.augment = params.get("augment", False)
        self.use_segments = params.get("use_segments", True)

        self.model_preprocess = params.get("model_preprocess")
        self.channel = params.get("channel")
        self.square_width = params.get("square_width", 5)
        self.output_dim = output_dim
        self.mvm = params.get("mvm", False)
        self.frame_size = params.get("frame_size", 32)

        self.keep_edge = params.get("keep_edge", False)
        self.maximum_preload = params.get("maximum_preload", 100)
        self.red_type = params.get("red_type", FrameTypes.thermal_tiled.name)
        self.green_type = params.get("green_type", FrameTypes.filtered_tiled.name)
        self.blue_type = params.get("blue_type", FrameTypes.filtered_tiled.name)


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
        log_q,
        **params,
    ):
        self.log_q = log_q
        self.logger = logging.getLogger()
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
        self.lazy_load = params.get("lazy_load", False)
        self.preload = params.get("preload", False)
        self.threads = True
        self.segments = []
        # load epoch
        self.epoch_labels = []
        self.epoch_data = []
        if self.preload:
            if self.threads:
                self.epoch_queue = deque()
                # m = multiprocessing.Manager()
                self.train_queue = deque()
            # self.train_queue = multiprocessing.Queue(self.params.maximum_preload)
            else:
                self.epoch_queue = multiprocessing.Queue()
                # m = multiprocessing.Manager()
                self.train_queue = multiprocessing.Queue(self.params.maximum_preload)
            # self.train_queue = multiprocessing.Queue(self.params.maximum_preload)
            self.preloader_thread = None
        self.load_next_epoch()
        self.logger.info(
            "datagen for %s shuffle %s cap %s epochs %s gen params %s memory %s",
            self.dataset.name,
            self.shuffle,
            self.cap_samples,
            self.epochs,
            self.params.__dict__,
            psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
        )

    def stop_load(self):
        if self.preload:
            if not self.epoch_queue:
                return
            self.logger.info("stopping %s", self.dataset.name)
            self.epoch_queue.appendleft("STOP")
            del self.train_queue
            self.train_queue = None
            if self.preloader_thread:
                self.preloader_thread.join(20)
                if self.preloader_thread.is_alive():
                    self.preloader_thread.kill()

    def get_epoch_labels(self, epoch=-1):
        return self.epoch_labels[epoch]

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(math.ceil(self.sample_size / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        start = time.time()
        if index == 0 and len(self.epoch_data) > self.cur_epoch:
            # when tensorflow uses model.fit it requests index 0 twice
            X, y = self.epoch_data[self.cur_epoch]
        else:
            if index == 0 and self.preload and self.lazy_load:
                self.preload_samples()
            try:
                X, y, y_original = self.get_item(index)
            except:
                self.logger.error(
                    "%s error getting preloaded data",
                    self.dataset.name,
                    exc_info=True,
                )

            # always keep a copy of epoch data
            if index == 0:
                self.epoch_data.append((X, y))
            self.epoch_labels[self.cur_epoch].extend(y_original)
        if np.isnan(np.sum(X)) or np.isnan(np.sum(y)):
            self.logger.error("Got nan in batch %s", index)
            return [[]], [[]]
        return X, y

    def get_item(self, index):
        if self.preload:
            while True:
                try:
                    return self.train_queue.popleft()
                except:
                    self.logger.debug("train_queue cant get item")
                    time.sleep(5)
            # return get_with_timeout(
            #     self.train_queue,
            #     30,
            #     f"train_queue get_item {self.dataset.name}",
            #     sleep_time=1,
            # )
        else:
            batch_segments = [
                self.samples[index * self.batch_size : (index + 1) * self.batch_size]
            ]
            segment_db = load_batch_frames(
                self.dataset.numpy_data,
                batch_segments,
                self.dataset.name,
            )
            batch_segments = batch_segments[0]
            segment_data = [None] * len(batch_segments)
            for i, seg in enumerate(batch_segments):
                segment_data[i] = (seg[1], seg[2], segment_db[seg[0]])

            return loadbatch(
                self.labels,
                segment_data,
                self.params,
                self.dataset.label_mapping,
            )

    def load_next_epoch(self):
        if self.loaded_epochs >= self.epochs:
            return
        self.epoch_labels.append([])
        self.logger.info(
            "%s loading epoch %s shuffling %s",
            self.dataset.name,
            (self.loaded_epochs + 1),
            self.shuffle,
        )
        self.samples = self.dataset.epoch_samples(
            cap_samples=self.cap_samples,
            replace=False,
            random=self.randomize_epoch,
            cap_at=self.cap_at,
            label_cap=self.label_cap,
        )
        # self.samples = np.uint32([sample.id for sample in self.samples])
        self.samples = [
            (sample.id, sample.label, sample.unique_track_id, sample.frame_indices)
            for sample in self.samples
        ]

        if self.shuffle:
            np.random.shuffle(self.samples)

        # first epoch needs sample size set, others set at epoch end
        if self.cur_epoch == 0:
            self.sample_size = len(self.samples)

        if self.preload and not self.lazy_load:
            self.preload_samples()
        if not self.preload:
            self.loaded_epochs += 1

        self.epoch_samples.append(len(self.samples))

    def stop_preload(self):
        if self.preloader_thread.is_alive():
            self.preloader_thread.join(30)
            self.logger.warn("Still alive after join terminating %s", self.dataset.name)
            if self.preloader_thread.is_alive():
                self.preloader_thread.terminate()

    def preload_samples(self):
        if self.preloader_thread is not None:
            self.stop_preload()
        self.preloader_thread = threading.Thread(
            target=preloader,
            args=(
                self.train_queue,
                self.epoch_queue,
                self.labels,
                self.dataset.name,
                self.params,
                self.dataset.label_mapping,
                self.dataset.numpy_data,
                self.log_q,
            ),
        )
        self.preloader_thread.start()
        batches = []
        for index in range(len(self)):
            samples = self.samples[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            batches.append(samples)
        if len(batches) > 0:
            self.epoch_queue.append((self.loaded_epochs + 1, batches))
        self.loaded_epochs += 1
        # self.epoch_queue.put("STOP")

    def reload_samples(self):
        self.logger.debug("%s reloading samples", self.dataset.name)
        if self.shuffle:
            np.random.shuffle(self.samples)
        batches = []
        for index in range(len(self)):
            samples = self.samples[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            batches.append(samples)

        if len(batches) > 0:
            self.epoch_queue.append((self.loaded_epochs + 1, batches))

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.load_next_epoch()
        self.sample_size = self.epoch_samples[self.cur_epoch]
        batches = len(self)
        self.epoch_data[self.cur_epoch] = None
        last_stats = self.epoch_label_count()
        self.logger.info(
            "%s epoch ended %s",
            self.dataset.name,
            last_stats,
        )

        self.cur_epoch += 1

    def epoch_label_count(self):
        labels = self.epoch_labels[self.cur_epoch]
        return Counter(labels)


def loadbatch(labels, data, params, mapped_labels, logger):
    X, y, y_orig = _data(labels, data, params, mapped_labels, logger)
    return X, y, y_orig


def _data(labels, data, params, mapped_labels, logger, to_categorical=True):
    "Generates data containing batch_size samples"
    # Initialization
    start = time.time()
    X = np.empty(
        (
            len(data),
            *params.output_dim,
        )
    )
    y = np.empty((len(data)), dtype=int)
    data_i = 0
    y_original = []
    mvm = []
    for label_original, u_id, frame_data in data:
        label = mapped_labels[label_original]
        try:
            if label not in labels:
                continue
            if params.use_segments:
                # temps = [frame.frame_temp_median for frame in frame_data]
                data = preprocess_movement(
                    frame_data,
                    params.square_width,
                    params.frame_size,
                    red_type=params.red_type,
                    green_type=params.green_type,
                    blue_type=params.blue_type,
                    preprocess_fn=params.model_preprocess,
                    augment=params.augment,
                    reference_level=None,
                    sample="test",
                    keep_edge=params.keep_edge,
                )
            else:
                data = preprocess_frame(
                    frame_data[0],
                    params.frame_size,
                    params.augment,
                    sample.temp_median,
                    sample.velocity,
                    params.output_dim,
                    preprocess_fn=params.model_preprocess,
                    sample=sample,
                )
            if data is None:
                continue
            y_original.append(label_original)
            X[data_i] = data
            y[data_i] = labels.index(label)
            if np.isnan(np.sum(data)) or labels.index(label) is None:
                logger.warn(
                    "Nan in data for %s %s",
                    u_id,
                    [frame.frame_number for frame in frame_data],
                )
                continue
            if np.amin(data) < -1 or np.amax(data) > 1:
                logger.warn(
                    "Data out of bounds for %s %s",
                    u_id,
                    [frame.frame_number for frame in frame_data],
                )
                continue
            data_i += 1
        except Exception as e:
            shapes = [frame.thermal.shape for frame in frame_data]
            frame_in = [frame.frame_number for frame in frame_data]

            logger.error(
                "Error getting data for %s frame shape %s indexes %s",
                u_id,
                shapes,
                frame_in,
                exc_info=True,
            )
            raise e
    # remove data that was null
    X = X[:data_i]
    y = y[:data_i]
    if len(X) == 0:
        logger.warn("Empty length of x")
    assert len(X) == len(y)
    if to_categorical:
        y = keras.utils.to_categorical(y, num_classes=len(labels))
    total_time = time.time() - start

    if params.mvm:
        return [np.array(X), np.array(mvm)], y, y_original
    return np.array(X), y, y_original


def load_from_numpy(numpy_meta, tracks, name, logger, size):
    start = time.time()
    count = 0
    track_is = 0
    prev = 0
    seek = 0
    segment_db = {}
    try:
        with numpy_meta as f:
            for _, segments, u_id in tracks:
                numpy_info = numpy_meta.track_info[u_id]
                track_is = u_id
                start_frame = numpy_info["start_frame"]
                if "data" not in numpy_info:
                    logger.warn("No data for", u_id)
                seek = numpy_info["data"]
                f.seek(numpy_info["data"])
                thermals = np.load(f, allow_pickle=False)
                filtered = np.load(f, allow_pickle=False)
                # regions = np.load(f, allow_pickle=False)

                for id, segment_frames in segments:
                    segment_data = np.empty(len(segment_frames), dtype=object)
                    segment_db[id] = segment_data

                    for i, frame_i in enumerate(segment_frames):
                        relative_f = frame_i - start_frame
                        count += 1
                        thermal = thermals[relative_f]
                        filter = filtered[relative_f]
                        frame = Frame.from_channels(
                            [thermal, filter],
                            [TrackChannels.thermal, TrackChannels.filtered],
                            frame_i,
                            flow_clipped=True,
                        )
                        # frame.region = regions[i]
                        # meta[0][relative_f]
                        # frame.frame_temp_median = meta[1][relative_f]
                        segment_data[i] = frame

            logger.debug(
                "%s time to load %s frames %s",
                name,
                count,
                time.time() - start,
            )
    except:
        logger.error(
            "%s error loading numpy file seek %s cur %s prev %s",
            name,
            seek,
            track_is,
            prev,
            exc_info=True,
        )
    return segment_db


def load_batch_frames(numpy_meta, batches, name, logger, size):
    track_frames = {}
    # loads batches from numpy file, by increasing file location, into supplied track_frames dictionary
    # returns loaded batches as segments
    all_batches = []
    data_by_track = {}
    for batch in batches:
        for s_id, label, track_id, frames in batch:
            track_segments = data_by_track.setdefault(
                track_id,
                (
                    numpy_meta.track_info[track_id]["data"],
                    [],
                    track_id,
                ),
            )
            track_segments[1].append((s_id, frames))
    # sort by position in file
    track_segments = sorted(
        data_by_track.values(),
        key=lambda track_segment: track_segment[0],
    )
    logger.info("%s loading tracks from numpy file", name)
    segment_db = load_from_numpy(numpy_meta, track_segments, name, logger, size)

    return segment_db


def preloader(
    train_queue, epoch_queue, labels, name, params, label_mapping, numpy_meta, log_q
):
    # worker_configurer(log_q)
    logger = logging.getLogger(f"Preload-{name}")
    # init_logging()
    """add a segment into buffer"""
    logger.info(
        " -started async fetcher for %s augment=%s numpyfile %s preload amount %s mem %s",
        name,
        params.augment,
        numpy_meta.filename,
        params.maximum_preload,
        psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
    )
    epoch = 0
    # this does the data pre processing
    processes = 8

    preload_amount = max(1, params.maximum_preload)
    max_jobs = max(1, params.maximum_preload)
    chunk_size = processes * 30
    while True:
        try:
            item = epoch_queue.popleft()
            break
        except:
            logger.info("Waiting for samples")
            time.sleep(5)
    # item = get_with_timeout(epoch_queue, 1, f"epoch_queue preloader {name}")
    try:
        epoch, batches = item
        count = 0

        logger.info(
            "%s preloader got %s batches for epoch %s mem %s",
            name,
            len(batches),
            epoch,
            psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
        )
        total = 0
        # Once process_batch starts to back up
        loaded_up_to = 0
        while len(batches) > 0:
            start = time.time()
            logger.info(
                "%s preloader memory %s",
                name,
                psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            )
            next_load = batches[:preload_amount]

            logger.info(
                "%s preloader loading %s - %s ",
                name,
                loaded_up_to,
                loaded_up_to + len(next_load),
            )
            loaded_up_to = loaded_up_to + len(next_load)

            segment_db = load_batch_frames(
                numpy_meta, next_load, name, logger, 36 if params.augment else 32
            )
            logger.info(
                "post load %s mem %s",
                len(next_load),
                psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            )

            data = []
            new_jobs = 0
            done = 0
            logger.info("post load %s", len(next_load))
            for batch_i, segments in enumerate(next_load):
                start = time.time()
                segment_data = [None] * len(segments)
                for i, seg in enumerate(segments):
                    segment_data[i] = (seg[1], seg[2], segment_db[seg[0]])
                # logger.info("preload %s", done)
                preprocessed = loadbatch(
                    labels, segment_data, params, label_mapping, logger
                )
                # logger.info("postload%s", done)

                # data.append(segment_data)
                train_queue.append(preprocessed)
                # put_with_timeout(
                #     train_queue,
                #     preprocessed,
                #     1,
                #     f"train_queue-process_batches",
                # )
                # logger.info("post put%s", done)
                done += 1
            results = None
            segment_db = None
            data = None
            del batches[:preload_amount]
            next_load = None
            segment_data = None
            logger.info(
                "%s preloader loaded up to %s time per batch %s",
                name,
                loaded_up_to,
                time.time() - start,
            )
            total += 1
        del batches
        # gc.collect()
        logger.info("%s preloader loaded epoch %s batches", name, epoch)

        # break
    except Exception as inst:
        logger.error("%s preloader epoch %s error %s", name, epoch, inst, exc_info=True)


LOG_EVERY = 25
labels = None
params = None
label_mapping = None


def init_process(l, p, map):
    global labels, params, label_mapping
    labels = l
    params = p
    label_mapping = map


def process_batch(segment_data, logger):
    # runs through loaded frames and applies appropriate prperocessing and then sends them to queue for training
    # try:
    # init_logging()
    global labels, params, label_mapping
    try:
        preprocessed = loadbatch(labels, segment_data, params, label_mapping, logger)
    except Exception as e:
        logger.error("Error processing batch", e)
        # self.loggererror("Error processing batch ", exc_info=True)
        return None
    return preprocessed


# Found hanging problems with blocking forever so using this as workaround
# keeps trying to put data in queue until complete
def put_with_timeout(queue, data, timeout, name=None, sleep_time=10):
    while True:
        try:
            queue.put(data, block=True, timeout=timeout)
            break
        except (Full):
            print("%s cant put cause full", name)
            time.sleep(sleep_time)
        except Exception as e:
            print(
                "%s put error %s t/o %s sleep %s",
                name,
                e,
                timeout,
                sleep_time,
                exc_info=True,
            )
            raise e


# keeps trying to get data in queue until complete
def get_with_timeout(queue, timeout, name=None, sleep_time=10):
    while True:
        try:
            queue_data = queue.get(block=True, timeout=timeout)
            # queue.task_done()
            return queue_data
        except (Empty):
            print("%s cant get cause empty", name)
            time.sleep(sleep_time)
        except Exception as e:
            print(
                "%s get error %s t/o %s sleep %s",
                name,
                e,
                timeout,
                sleep_time,
                exc_info=True,
            )
            raise e


import sys
import attr


def get_size(obj, name="base", seen=None, depth=0):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if attr.has(obj):
        obj = attr.asdict(obj)
        size += sum(
            [get_size(v, f"{name}.{k}", seen, depth + 1) for k, v in obj.items()]
        )
        size += sum([get_size(k, f"{name}.{k}", seen, depth + 1) for k in obj.keys()])
    if isinstance(obj, dict):
        size += sum(
            [get_size(v, f"{name}.{k}", seen, depth + 1) for k, v in obj.items()]
        )
        size += sum([get_size(k, f"{name}.{k}", seen, depth + 1) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):

        size += get_size(obj.__dict__, f"{name}.dict", seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        # print("iter??")

        size += sum([get_size(i, f"{name}.iter", seen, depth + 1) for i in obj])
    # if size * 1e-6 > 0.1 and depth <= 1:
    #     print(name, " size ", round(size * 1e-6, 2), "MB")
    return size
