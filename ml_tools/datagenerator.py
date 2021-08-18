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
from collections import Counter
from queue import Empty, Full
import threading
from ml_tools.logs import init_logging
import psutil
import os

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
        self.preload = params.get("preload", False)

        use_threads = False

        self.segments = []
        # load epoch
        self.epoch_labels = []
        self.epoch_data = []
        if self.preload:
            if use_threads:
                # This will be slower, but use less memory
                # Queue is also extremely slow so should be change to deque
                self.epoch_queue = multiprocessing.Queue()
                self.train_queue = multiprocessing.Queue(self.params.maximum_preload)
                self.preloader_thread = threading.Thread(
                    target=preloader,
                    args=(
                        self.train_queue,
                        self.epoch_queue,
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
                multiprocessing.set_start_method("spawn", force=True)
                self.epoch_queue = multiprocessing.Queue()
                # m = multiprocessing.Manager()
                self.train_queue = multiprocessing.Queue(self.params.maximum_preload)
                # self.train_queue = multiprocessing.Queue(self.params.maximum_preload)
                self.preloader_thread = multiprocessing.Process(
                    target=preloader,
                    args=(
                        self.train_queue,
                        self.epoch_queue,
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

        self.load_next_epoch()
        logging.info(
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
            logging.info("stopping %s", self.dataset.name)
            self.epoch_queue.put("STOP")
            del self.train_queue
            self.train_queue = None
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
        if index == len(self) // 2 and self.eager_load:
            logging.debug(
                "%s on epoch %s index % s eager loading next epoch data",
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

                X, y, y_original = self.get_item(index)
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

    def get_item(self, index):
        if self.preload:
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
            segments, track_frames = load_batch_frames(
                self.dataset.numpy_data,
                batch_segments,
                self.dataset.segments_by_id,
                self.dataset.name,
            )
            segments = segments[0]
            segment_data = []
            for seg in segments:
                frame_data = get_cached_frames(track_frames, seg)
                segment_data.append(frame_data)
            return loadbatch(
                self.labels,
                segments,
                segment_data,
                self.params,
                self.dataset.label_mapping,
            )

    def load_next_epoch(self):
        if self.loaded_epochs >= self.epochs:
            return
        self.epoch_labels.append([])
        logging.info(
            "%s loading epoch %s shuffling %s",
            self.dataset.name,
            self.loaded_epochs,
            self.shuffle,
        )
        self.samples = self.dataset.epoch_samples(
            cap_samples=self.cap_samples,
            replace=False,
            random=self.randomize_epoch,
            cap_at=self.cap_at,
            label_cap=self.label_cap,
        )
        self.samples = np.uint32([sample.id for sample in self.samples])

        if self.shuffle:
            np.random.shuffle(self.samples)

        # first epoch needs sample size set, others set at epoch end
        if self.cur_epoch == 0:
            self.sample_size = len(self.samples)

        if self.preload:
            batches = []
            for index in range(len(self)):
                samples = self.samples[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]
                batches.append(samples)
            if len(batches) > 0:
                self.epoch_queue.put((self.loaded_epochs + 1, batches))

            self.dataset.segments = []
            # lower memory since these are already in processes
            gc.collect()
        self.epoch_samples.append(len(self.samples))
        self.loaded_epochs += 1

    def reload_samples(self):
        logging.debug("%s reloading samples", self.dataset.name)
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
        if not self.eager_load:
            self.load_next_epoch()
        self.sample_size = self.epoch_samples[self.cur_epoch]
        batches = len(self)
        self.epoch_data[self.cur_epoch] = None
        last_stats = self.epoch_label_count()
        logging.info(
            "%s epoch ended %s",
            self.dataset.name,
            last_stats,
        )

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
        if f_i not in track_frames:
            logging.warn("Caanot not load %s frame %s", sample, f_i)
            continue
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
        if params.use_segments:
            data = preprocess_movement(
                frame_data,
                params.square_width,
                params.frame_size,
                red_type=params.red_type,
                green_type=params.green_type,
                blue_type=params.blue_type,
                preprocess_fn=params.model_preprocess,
                augment=params.augment,
                reference_level=sample.frame_temp_median,
                sample=sample,
                keep_edge=params.keep_edge,
            )
            if data is not None:
                mvm.append(sample.movement_data)
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
        # just for testing
        xnan = np.any(np.isnan(data))
        if xnan:
            logging.error(
                "got  nan for x samples %s augment %s indices %s",
                sample,
                params.augment,
                sample.frame_indices,
            )

        y_original.append(sample.label)
        X[data_i] = data
        y[data_i] = labels.index(label)
        data_i += 1
    # remove data that was null
    X = X[:data_i]
    y = y[:data_i]
    if len(X) == 0:
        logging.warn("Empty length of x")
    assert len(X) == len(y)
    if to_categorical:
        y = keras.utils.to_categorical(y, num_classes=len(labels))
    total_time = time.time() - start

    if params.mvm:
        return [np.array(X), np.array(mvm)], y, y_original
    return np.array(X), y, y_original


def load_from_numpy(numpy_meta, frames_by_track, tracks, name):
    start = time.time()
    count = 0
    try:
        with numpy_meta as f:
            for _, frame_indices, u_id, regions_by_frames in tracks:
                numpy_info = numpy_meta.track_info[u_id]
                frames_to_i = numpy_info["frames"]
                track_data = frames_by_track.setdefault(u_id, {})
                f.seek(numpy_info["data"])
                frames = np.load(f, allow_pickle=True)
                thermals = frames[0]
                filtered = frames[1]
                for frame_i in frame_indices:
                    count += 1
                    thermal = thermals[frames_to_i[frame_i]]
                    filter = filtered[frames_to_i[frame_i]]
                    frame = Frame.from_channels(
                        [thermal, filter],
                        [TrackChannels.thermal, TrackChannels.filtered],
                        frame_i,
                        flow_clipped=True,
                    )
                    track_data[frame_i] = frame
                    frame.region = regions_by_frames[frame_i]

            logging.debug(
                "%s time to load %s frames %s",
                name,
                count,
                time.time() - start,
            )
    except:
        logging.error("%s error loading numpy file", name, exc_info=True)
    return frames_by_track


def load_batch_frames(
    numpy_meta,
    batches_by_id,
    segments_by_id,
    name,
):
    track_frames = {}
    # loads batches from numpy file, by increasing file location, into supplied track_frames dictionary
    # returns loaded batches as segments
    all_batches = []
    data_by_track = {}
    for batch in batches_by_id:
        batch_segments = []
        for s_id in batch:
            segment = segments_by_id[s_id]
            batch_segments.append(segment)
            track_segments = data_by_track.setdefault(
                segment.unique_track_id,
                (
                    numpy_meta.track_info[segment.unique_track_id]["data"],
                    [],
                    segment.unique_track_id,
                    {},
                ),
            )
            regions_by_frames = track_segments[3]
            for region in segment.track_bounds:
                regions_by_frames[region.frame_number] = region
            track_segments[1].extend(segment.frame_indices)
        all_batches.append(batch_segments)
    # sort by position in file
    track_segments = sorted(
        data_by_track.values(),
        key=lambda track_segment: track_segment[0],
    )
    logging.debug("%s loading tracks from numpy file", name)
    load_from_numpy(numpy_meta, track_frames, track_segments, name)

    return all_batches, track_frames


train_queue = None

jobs = 0


def preloader(
    q,
    epoch_queue,
    labels,
    name,
    db,
    segments_by_id,
    params,
    label_mapping,
    numpy_meta,
):
    global train_queue, jobs
    train_queue = q
    init_logging()
    """add a segment into buffer"""
    logging.info(
        " -started async fetcher for %s augment=%s numpyfile %s preload amount %s mem %s",
        name,
        params.augment,
        numpy_meta.filename,
        params.maximum_preload,
        psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
    )
    epoch = 0

    # this does the data pre processing
    p_list = []
    processes = 1
    if name == "train":
        processes = 4

    pool = multiprocessing.Pool(
        processes,
        init_process,
        (labels, params, label_mapping),
        maxtasksperchild=100,
    )
    chunk_size = 50
    preload_amount = max(1, params.maximum_preload)
    while True:
        try:
            item = get_with_timeout(epoch_queue, 1, f"epoch_queue preloader {name}")
        except:
            logging.error("%s preloader epoch %s error", name, epoch, exc_info=True)
            return

        if item == "STOP":
            logging.info("%s preloader received stop", name)
            pool.terminate()
            return
        try:
            epoch, batches = item
            count = 0

            logging.debug(
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
                logging.debug(
                    "%s preloader memory %s",
                    name,
                    psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
                )

                while jobs > (params.maximum_preload - preload_amount // 2):
                    logging.debug("%s waiting for jobs to complete %s", name, jobs)
                    time.sleep(5)
                next_load = batches[:preload_amount]

                logging.debug(
                    "%s preloader loading %s - %s ",
                    name,
                    loaded_up_to,
                    loaded_up_to + len(next_load),
                )
                loaded_up_to = loaded_up_to + len(next_load)

                batch_data, track_frames = load_batch_frames(
                    numpy_meta,
                    next_load,
                    segments_by_id,
                    name,
                )

                data = []
                for batch_i, segments in enumerate(batch_data):
                    start = time.time()
                    segment_data = [None] * len(segments)
                    for i, seg in enumerate(segments):
                        frame_data = get_cached_frames(track_frames, seg)
                        segment_data[i] = frame_data
                    data.append((segments, segment_data))
                    batch_data[batch_i] = None
                    if len(data) > chunk_size or batch_i == (len(batch_data) - 1):
                        pool.map_async(process_batch, data, callback=processed_data)
                        jobs += len(data)
                        del data
                        data = []
                del batch_data
                del track_frames
                del batches[:preload_amount]
                gc.collect()
                logging.debug(
                    "%s preloader loaded  up to %s",
                    name,
                    loaded_up_to,
                )
                total += 1

            gc.collect()
            logging.info("%s preloader loaded epoch %s batches", name, epoch)
        except Exception as inst:
            logging.error(
                "%s preloader epoch %s error %s", name, epoch, inst, exc_info=True
            )


def processed_data(results):
    global jobs
    for result in results:
        put_with_timeout(
            train_queue,
            result,
            1,
            f"train_queue-process_batches",
        )
        del result
    jobs -= len(results)
    del results


LOG_EVERY = 25
labels = None
params = None
label_mapping = None


def init_process(l, p, map):
    global labels, params, label_mapping
    labels = l
    params = p
    label_mapping = map


def process_batch(arguments):
    # runs through loaded frames and applies appropriate prperocessing and then sends them to queue for training
    # try:

    global labels, params, label_mapping
    segments, segment_data = arguments
    preprocessed = loadbatch(labels, segments, segment_data, params, label_mapping)
    return preprocessed


# Found hanging problems with blocking forever so using this as workaround
# keeps trying to put data in queue until complete
def put_with_timeout(queue, data, timeout, name=None, sleep_time=10):
    while True:
        try:
            queue.put(data, block=True, timeout=timeout)
            break
        except (Full):
            logging.debug("%s cant put cause full", name)
            time.sleep(sleep_time)
        except Exception as e:
            logging.error(
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
            logging.debug("%s cant get cause empty", name)
            time.sleep(sleep_time)
        except Exception as e:
            logging.error(
                "%s get error %s t/o %s sleep %s",
                name,
                e,
                timeout,
                sleep_time,
                exc_info=True,
            )
            raise e
