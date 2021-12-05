import math
import logging
import tensorflow.keras as keras
import numpy as np

# from multiprocessing import Queue
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

# from concurrent.futures.process import ProcessPoolExecutor
import pickle
import tracemalloc
from multiprocess import Queue, Process
from pathos.multiprocessing import ProcessPool

FRAMES_PER_SECOND = 9

item_c = 0


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
        self.threads = False
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
                self.epoch_queue = Queue()
                # m = multiprocessing.Manager()
                self.train_queue = Queue(self.params.maximum_preload)
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
            # if not self.epoch_queue:
            #     return
            self.logger.info("stopping %s", self.dataset.name)
            # self.epoch_queue.appendleft("STOP")
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
            X, y, weights = self.epoch_data[self.cur_epoch]
        else:
            if index == 0 and self.preload and self.lazy_load:
                self.preload_samples()
            try:
                X, y, y_original, weights = self.get_item(index)
                global item_c
                item_c -= 1
            except:
                self.logger.error(
                    "%s error getting preloaded data",
                    self.dataset.name,
                    exc_info=True,
                )

            # always keep a copy of epoch data
            if index == 0:
                self.epoch_data.append((X, y, weights))
            self.epoch_labels[self.cur_epoch].extend(y_original)
        return X, y, weights

    def get_item(self, index):
        if self.preload:
            # while True:
            #     try:
            #         return self.train_queue.popleft()
            #     except:
            #         self.logger.debug("train_queue cant get item")
            #         time.sleep(5)
            return get_with_timeout(
                self.train_queue,
                30,
                f"train_queue get_item {self.dataset.name}",
                sleep_time=1,
            )
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
        self.loaded_epochs += 1
        self.preloader_thread = Process(
            target=preloader,
            args=(
                self.samples,
                self.batch_size,
                self.train_queue,
                self.labels,
                self.dataset.name,
                self.params,
                self.dataset.label_mapping,
                self.dataset.numpy_data,
                self.log_q,
            ),
        )
        self.preloader_thread.start()
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
            self.epoch_queue.put((self.loaded_epochs + 1, batches))

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
    X, y, y_orig, weights = _data(labels, data, params, mapped_labels, logger)
    return X, y, y_orig, weights


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
    weights = np.empty((len(data)), dtype=np.float32)

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
            if label == "bird":
                weight = 1.6
            elif label == "wallaby":
                # wallabies not so important better to predict birds
                weight = 0.6
            else:
                weight = 1
            weights[data_i] = weight
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
    weights = weights[:data_i]
    if len(X) == 0:
        logger.warn("Empty length of x")
    assert len(X) == len(y)
    if to_categorical:
        y = keras.utils.to_categorical(y, num_classes=len(labels))
        # , weights
    total_time = time.time() - start

    if params.mvm:
        return [np.array(X), np.array(mvm)], y, y_original
    return np.array(X), y, y_original, weights


def load_from_numpy(numpy_meta, batches, name, logger, size):
    start = time.time()
    count = 0
    segment_db = {}
    try:
        with open(numpy_meta.filename, "rb") as f:
            for s_id, label, track_id, frames in batches:
                try:
                    s_offset = numpy_meta.track_info[track_id]["segments"][s_id]
                    f.seek(s_offset)
                    thermals = np.load(f, allow_pickle=False)
                    filtered = np.load(f, allow_pickle=False)
                    segment_data = []
                    segment_db[s_id] = segment_data
                    for thermal, filtered, frame_i in zip(thermals, filtered, frames):
                        # seems to leek memory without np.copy() go figure
                        frame = Frame.from_channels(
                            [np.copy(thermal), np.copy(filtered)],
                            [TrackChannels.thermal, TrackChannels.filtered],
                            frame_i,
                            flow_clipped=True,
                        )
                        count += 1
                        segment_data.append(frame)
                except:
                    logger.error("%s error loading %s segment %s", name, track_id, s_id)
            logger.debug(
                "%s time to load %s frames %s",
                name,
                count,
                time.time() - start,
            )
    except:
        logger.error(
            "%s error loading numpy file ",
            name,
            exc_info=True,
        )
    return segment_db


def load_batch_frames(numpy_meta, batches, name, logger, size):
    track_frames = {}
    # loads batches from numpy file, by increasing file location, into supplied track_frames dictionary
    # returns loaded batches as segments
    all_batches = []
    data_by_track = {}

    batches = sorted(
        batches,
        key=lambda batch_item: numpy_meta.track_info[batch_item[2]]["segments"][
            batch_item[0]
        ],
    )
    logger.info(
        "%s loading tracks from numpy file pre mem %s",
        name,
        psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
    )
    segment_db = load_from_numpy(numpy_meta, batches, name, logger, size)
    logger.info(
        "loaded %s mem %s",
        len(segment_db),
        psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
    )
    return segment_db


LOG_EVERY = 25
labels = None
params = None
label_mapping = None
#
#
# def preloader(
#     segments,
#     batch_size,
#     loaded_queue,
#     l,
#     name,
#     p,
#     l_map,
#     numpy_meta,
#     log_q,
# ):
#     worker_configurer(log_q)
#
#     global labels, params, label_mapping, logger_q
#     logger_q = log_q
#     labels = l
#     params = p
#     label_mapping = l_map
#     logger = logging.getLogger(f"Preload-{name}")
#     logger.info("WORKING")
#     preload_amount = 400  # max(1, params.maximum_preload)
#     orig_lbls = list(l_map.keys())
#     while len(segments) > 0:
#         next_load = segments[: batch_size * preload_amount]
#         data = []
#         while len(next_load) > 0:
#             batch_segments = next_load[:batch_size]
#             batch_data = []
#             for i, seg in enumerate(batch_segments):
#                 segment_data = []
#                 for z in range(25):
#                     f = Frame(
#                         np.random.rand(36, 36),
#                         np.random.rand(36, 36),
#                         None,
#                         frame_number=z,
#                     )
#                     segment_data.append(f)
#                 batch_data.append((orig_lbls[i % len(l_map)], i, segment_data))
#             data.append(batch_data)
#             next_load = next_load[batch_size:]
#         batch_segments = None
#         logger.debug(
#             "processing %s mem %s",
#             len(data),
#             psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
#         )
#         with ProcessPool(
#             max_workers=4,
#             # initializer=init_process,
#             # initargs=(labels, params, label_mapping, log_q),
#         ) as pool:
#             results = pool.uimap(process_batch, data, chunksize=50)
#             for res in results:
#                 while True:
#                     try:
#                         loaded_queue.put(res, block=True, timeout=10)
#                         break
#                     except (Full):
#                         time.sleep(5)
#         del pool
#         results = None
#         segment_db = None
#         data = None
#         next_load = None
#         segment_data = None
#         logger.debug(
#             "loaded batch qsize %s mem %s",
#             loaded_queue.qsize(),
#             psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
#         )
#         segments = segments[batch_size * preload_amount :]
#         while loaded_queue.qsize() > max(10, preload_amount // 2):
#             logger.debug("waiting for less items")
#             time.sleep(10)

#
# def preprocess(batch):
#     X = []
#     y = []
#     orig = []
#     weights = []
#     try:
#         for frames in batch:
#             for item in frames:
#                 item.thermal = item.thermal / (
#                     np.amax(item.thermal) - np.amin(item.thermal)
#                 )
#             X.append(np.random.rand(160, 160, 3))
#             y.append(0)
#             orig.append("TEST")
#             weights.append(1.0)
#         weights = np.array(weights)
#         y = np.array(y)
#         X = np.array(X)
#         orig = np.array(orig)
#         y = keras.utils.to_categorical(y, num_classes=6)
#         return X, y, orig, weights
#     except e:
#         print("error preprocess", e)


def preloader(
    samples,
    batch_size,
    train_queue,
    l,
    name,
    p,
    l_map,
    numpy_meta,
    log_q,
):
    global labels, params, label_mapping, logger_q
    logger_q = log_q
    labels = l
    params = p
    label_mapping = l_map
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
    processes = 4

    preload_amount = max(1, params.maximum_preload)
    max_jobs = max(1, params.maximum_preload // 2)
    chunk_size = processes * 30
    batches = math.ceil(len(samples) / batch_size)
    use_pool = True
    global item_c
    try:
        count = 0

        logger.info(
            "%s preloader got %s batches for epoch %s mem %s",
            name,
            batches,
            epoch,
            psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
        )
        total = 0
        # Once process_batch starts to back up
        loaded_up_to = 0
        total_samples = len(samples)
        while len(samples) > 0:
            next_load = samples[: batch_size * preload_amount]

            start = time.time()
            logger.info(
                "%s preloader memory %s",
                name,
                psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            )
            # next_load = batches[:preload_amount]

            logger.info(
                "%s preloader loading %s - %s  / %s",
                name,
                loaded_up_to,
                loaded_up_to + len(next_load),
                total_samples,
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
            num_batches = math.ceil(len(next_load) / batch_size)
            start = time.time()
            while len(next_load) > 0:
                segments = next_load[:batch_size]
                segment_data = []
                for i, seg in enumerate(segments):
                    if seg[0] in segment_db:
                        segment_data.append((seg[1], seg[2], segment_db[seg[0]]))
                        segment_db[seg[0]] = None
                if not use_pool:
                    preprocessed = loadbatch(
                        labels, segment_data, params, label_mapping, logger
                    )
                    put_with_timeout(train_queue, preprocessed, 10, "preloader")
                    # train_queue.append(preprocessed)
                    item_c += 1
                else:
                    data.append(segment_data)
                next_load = next_load[batch_size:]
            logger.debug(
                "processing %s mem %s",
                len(data),
                psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            )
            # tracemalloc.start()
            # snapshot1 = tracemalloc.take_snapshot()

            #     for segment_data in data:
            #         preprocessed = loadbatch(
            #             labels, segment_data, params, label_mapping, logger
            #         )
            #         put_with_timeout(train_queue, preprocessed, 10, "preloader")
            #         # train_queue.append(preprocessed)
            #         item_c += 1
            # else:
            if use_pool:

                with ProcessPool(max_workers=processes) as pool:

                    results = pool.map(process_batch, data, chunksize=5)
                    for res in results:
                        put_with_timeout(train_queue, res, 10, "preloader", log_q)
                    item_c += 1
                # del pool

            results = None
            segment_db = None
            data = None
            samples = samples[batch_size * preload_amount :]
            next_load = None
            segment_data = None
            logger.info(
                "%s preloader loaded up to %s time per batch %s qsize %s",
                name,
                loaded_up_to,
                time.time() - start,
                train_queue.qsize(),
            )
            # snapshot2 = tracemalloc.take_snapshot()
            # top_stats = snapshot2.compare_to(snapshot1, "lineno")

            # logger.info("[ Top 10 differences ]")
            # for stat in top_stats[:10]:
            #     logger.info("%s", stat)
            total += 1
            # while item_c > 0:
            #     logger.info(
            #         "waiting for items %s mem %s",
            #         item_c,
            #         psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
            #     )
            #     time.sleep(5)

        del batches
        # gc.collect()
        logger.info("%s preloader loaded epoch %s batches", name, epoch)

        # break
    except Exception as inst:
        logger.error("%s preloader epoch %s error %s", name, epoch, inst, exc_info=True)


def init_process(l, p, map, log_q):
    raise "EX"
    global labels, params, label_mapping, logger
    labels = l
    params = p
    label_mapping = map
    worker_configurer(log_q)
    logger = logging.getLogger(f"Pool-worker")
    print("INIT PROCDSS")


#
#
def process_batch(segment_data):
    # runs through loaded frames and applies appropriate prperocessing and then sends them to queue for training
    # try:
    # init_logging()
    global labels, params, label_mapping, logger_q
    logger = logging.getLogger(f"process_batch")

    try:
        preprocessed = loadbatch(labels, segment_data, params, label_mapping, logger)
    except Exception as e:
        print("Error processing batch", e)
        print("EXCEPTION", e)
        # self.loggererror("Error processing batch ", exc_info=True)
        return None
    return preprocessed


# Found hanging problems with blocking forever so using this as workaround
# keeps trying to put data in queue until complete
def put_with_timeout(queue, data, timeout, name=None, sleep_time=10, log_q=None):
    while True:
        try:
            queue.put(data, block=True, timeout=timeout)
            break
        except (Full):
            if log_q:
                log_q.debug("%s cant put cause full", name)
            print("%s cant put cause full", name)
            time.sleep(sleep_time)
        except Exception as e:
            if log_q:
                log_q.error(
                    "%s put error %s t/o %s sleep %s",
                    name,
                    e,
                    timeout,
                    sleep_time,
                    exc_info=True,
                )
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
    return 0
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
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        size += sum(
            [get_size(v, f"{name}.{k}", seen, depth + 1) for k, v in obj.items()]
        )
        size += sum([get_size(k, f"{name}.{k}", seen, depth + 1) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):

        size += get_size(obj.__dict__, f"{name}.dict", seen, depth + 1)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        # print("iter??")

        size += sum([get_size(i, f"{name}.iter", seen, depth + 1) for i in obj])
    if size * 0.000001 > 0.1 and depth <= 2:
        print(name, " size ", size * 0.000001, "MB")
    return size
