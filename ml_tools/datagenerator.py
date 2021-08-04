from collections import deque
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
            self.epoch_queue = Queue()
            self.train_queue = Queue(self.params.maximum_preload)
        else:
            self.epoch_queue = multiprocessing.Manager().Queue()
            self.train_queue = multiprocessing.Manager().Queue(
                self.params.maximum_preload
            )

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
        if not self.epoch_queue:
            return
        logging.info("stopping %s", self.dataset.name)
        self.epoch_queue.put("STOP")
        time.sleep(10)
        if hasattr(self.preloader_thread, "terminate"):
            self.preloader_thread.terminate()
        del self.train_queue
        del self.preloader_thread
        del self.epoch_queue
        self.epoch_queue = None
        self.train_queue = None

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
        # logging.info("%s requesting %s", self.dataset.name, index)
        if index == 0 and len(self.epoch_data) > self.cur_epoch:
            # when tensorflow uses model.fit it requests index 0 twice
            X, y = self.epoch_data[self.cur_epoch]
        else:
            try:
                X, y, y_original = get_with_timeout(
                    self.train_queue,
                    30,
                    f"train_queue get_item {self.dataset.name}",
                    sleep_time=1,
                )
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
            self.epoch_queue.put((self.loaded_epochs + 1, batches))

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
                self.epoch_queue.put((self.loaded_epochs + 1, batches))

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
        logging.info(
            "epoch ended for %s %s",
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
    assert len(X) == len(y)
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
    logging.info("%s loading tracks from numpy file", name)
    try:
        with numpy_meta as f:
            load_from_numpy(f, track_frames, track_segments, channels, name)
    except:
        logging.error("%s error loading numpy file", name, exc_info=True)
    return all_batches


def preloader(
    train_queue,
    epoch_queue,
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
    batch_q = multiprocessing.Manager().Queue(1)

    # this thread does the data pre processing
    p_list = []
    processes = 1
    if name == "train":
        processes = 2
    for i in range(processes):
        p_preprocess = multiprocessing.Process(
            target=process_batches,
            args=(
                batch_q,
                train_queue,
                labels,
                params,
                label_mapping,
                name,
            ),
        )
        p_list.append(p_preprocess)
        p_preprocess.start()
    while True:
        try:
            item = get_with_timeout(epoch_queue, 1, f"epoch_queue preloader {name}")
        except:
            logging.error("%s preloader epoch %s error", name, epoch, exc_info=True)
            return

        if item == "STOP":
            for i in range(processes):
                batch_q.put("STOP")
            logging.info("%s preloader received stop", name)
            break
        try:
            epoch, batches = item
            logging.info(
                "%s preloader got %s batches for epoch %s",
                name,
                len(batches),
                epoch,
            )
            total = 0

            preload_amount = min(
                100, params.maximum_preload // 2
            )  # Once process_batch starts to back up
            loaded_up_to = 0
            while loaded_up_to < len(batches):
                if loaded_up_to < len(batches):
                    next_load = batches[loaded_up_to : loaded_up_to + preload_amount]
                    logging.info(
                        "%s preloader loading more data, have segments: %s  loading %s - %s of %s qsize %s ",
                        name,
                        batch_q.qsize(),
                        loaded_up_to,
                        loaded_up_to + len(next_load),
                        len(batches),
                        train_queue.qsize(),
                    )
                    batch_data = load_batch_frames(
                        track_frames,
                        numpy_meta,
                        next_load,
                        segments_by_id,
                        channels,
                        name,
                    )
                    logging.info(
                        "%s preloader got batch frames %s", name, len(batch_data)
                    )
                    loaded_batches = []
                    for segments in batch_data:
                        segment_data = []
                        for seg in segments:
                            frame_data = get_cached_frames(track_frames, seg)
                            segment_data.append(frame_data)
                        loaded_batches.append((segments, segment_data))
                    # this will block if process batches isn't ready for more
                    put_with_timeout(
                        batch_q,
                        (epoch, loaded_batches),
                        1,
                        f"prealoder batch_q {name}",
                    )
                    # put all at once save queue overheader
                    del track_frames
                    track_frames = {}
                    gc.collect()
                    loaded_up_to = loaded_up_to + len(next_load)
                    logging.info(
                        "%s preloader loaded more data qsize is %s loaded up to %s",
                        name,
                        batch_q.qsize(),
                        loaded_up_to,
                    )
                    total += 1
                else:
                    logging.info(
                        "%s preloader preloaded maximum frames batch qsize %s, loaded up to %s",
                        name,
                        batch_q.qsize(),
                        loaded_up_to,
                    )
                    time.sleep(2)

            logging.info("%s preloader loaded epoch %s", name, epoch)
        except Exception as inst:
            logging.error(
                "%s preloader epoch %s error %s", name, epoch, inst, exc_info=True
            )
            pass


def process_batches(batch_queue, train_queue, labels, params, label_mapping, name):
    # runs through loaded frames and applies appropriate prperocessing and then sends them to queue for training
    total = 0
    epoch = 0
    chunk_size = 40
    while True:
        batches = None
        batches = get_with_timeout(
            batch_queue, 1, f"batch_queue process_batches {name}"
        )
        if batches == "STOP":
            logging.info("%s process_batches thread received stop", name)
            return

        if epoch != batches[0]:
            epoch = batches[0]
            logging.info("%s process_batches loading new epoch %s", name, epoch)
            total = 0
        batches = batches[1]
        chunks = math.ceil(len(batches) / chunk_size)
        for batch_i in range(chunks):
            start = batch_i * chunk_size
            chunk = batches[start : start + chunk_size]

            for i in range(len(chunk)):
                segments, data = chunk[i]
                batch_data = loadbatch(labels, segments, data, params, label_mapping)
                chunk[i] = batch_data
            logging.info(
                "%s process_batches - epoch %s preprocessed %s range %s - %s",
                name,
                epoch,
                total,
                start,
                start + chunk_size,
            )
            for batch_data in chunk:
                try:
                    put_with_timeout(
                        train_queue,
                        batch_data,
                        1,
                        f"train_queue-process_batches {name}",
                    )

                    total += 1
                except Exception as e:
                    logging.error(
                        "%s process_batches batch Put error",
                        name,
                        exc_info=True,
                    )
                    raise e


# Found hanging problems with blocking forever so using this as workaround
# keeps trying to put data in queue until complete
def put_with_timeout(queue, data, timeout, name=None, sleep_time=10):
    while True:
        try:
            queue.put(data, block=True, timeout=timeout)
            break
        except (Full):
            logging.info("%s cant put cause full", name)
            time.sleep(sleep_time)
            pass
        except Exception as e:
            raise e


# keeps trying to get data in queue until complete
def get_with_timeout(queue, timeout, name=None, sleep_time=10):
    while True:
        try:
            queue_data = queue.get(block=True, timeout=timeout)
            break
        except (Empty):
            logging.info("%s cant get cause empty", name)
            time.sleep(sleep_time)
            pass
        except Exception as e:
            raise e
    return queue_data
