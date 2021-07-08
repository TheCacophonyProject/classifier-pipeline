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
from ml_tools.dataset import TrackChannels
import queue
from ml_tools.frame import Frame

FRAMES_PER_SECOND = 9


class GeneartorParams:
    def __init__(self, output_dim, params):
        self.augment = params.get("augment", False)
        self.use_dots = params.get("use_dots", False)
        self.use_movement = params.get("use_movement")
        self.model_preprocess = params.get("model_preprocess")
        self.channel = params.get("channel")
        self.square_width = params.get("square_width", 5)
        self.output_dim = output_dim
        self.mvm = params.get("mvm", False)
        self.type = params.get("type", 1)
        self.segment_type = params.get("segment_type", 1)
        self.load_threads = params.get("load_threads", 2)
        self.keep_edge = params.get("keep_edge", False)


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
        self.preload = params.get("preload", True)
        self.epochs = params.get("epochs", 10)
        self.samples = None
        self.keep_epoch = params.get("keep_epoch")
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
        self.epoch_data = []
        self.epoch_stats = []
        self.epoch_labels = []
        if self.preload:
            self.load_queue = multiprocessing.Queue()
        if self.preload:
            self.preloader_queue = multiprocessing.Queue(params.get("buffer_size", 128))
        print("buffer is", params.get("buffer_size", 128))
        self.segments = None
        self.segments = []
        # load epoch
        self.load_next_epoch()
        self.epoch_stats.append({})
        self.epoch_data.append(([None] * len(self), [None] * len(self)))
        self.epoch_labels.append([None] * len(self))
        self.preloader_threads = []
        if self.preload:

            self.preloader_threads = [
                multiprocessing.Process(
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
                        dataset.numpy_file,
                    ),
                )
                for _ in range(self.params.load_threads)
            ]
            for thread in self.preloader_threads:
                thread.start()
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
        if not self.preload or not self.load_queue:
            return
        logging.info("stopping %s", self.dataset.name)
        for thread in self.preloader_threads:
            if hasattr(thread, "terminate"):
                thread.terminate()
            else:
                thread.exit()
        del self.preloader_queue
        del self.preloader_threads
        del self.load_queue
        self.load_queue = None
        self.preloader_queue = None

    def get_epoch_labels(self, epoch=-1):
        return self.epoch_labels[epoch]

    def get_epoch_predictions(self, epoch=-1):
        if self.epoch_data is not None:
            return self.epoch_data[epoch](0)
        return None

    def __len__(self):
        "Denotes the number of batches per epoch"

        return int(math.ceil(self.sample_size / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        start = time.time()
        if index == len(self) - 1:
            logging.info(
                "%s on epoch %s index % s loading next epoch data",
                self.dataset.name,
                self.loaded_epochs,
                index,
            )
            self.load_next_epoch()
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
                while True:
                    try:
                        X, y, y_original = self.preloader_queue.get(
                            block=True, timeout=30
                        )

                        break
                    except (queue.Empty):
                        logging.debug("%s Preload queue is empty", self.dataset.name)
                    except Exception as inst:
                        logging.error(
                            "%s error getting preloaded data %s",
                            self.dataset.name,
                            inst,
                            exc_info=True,
                        )
                        pass
                # X = process_frames(X, self.params)
            else:
                X, y, y_original = self.loadbatch(index)

            if self.epoch_data[self.cur_epoch][0][index] is None:
                epoch_stats = self.epoch_stats[self.cur_epoch]
                out_y = np.argmax(y, axis=1)
                indices, counts = np.unique(out_y, return_counts=True)
                for lbl in y_original:
                    epoch_stats.setdefault(lbl, 0)
                    epoch_stats[lbl] += 1
                # for i, label_index in enumerate(indices):
                #     label = self.labels[label_index]
                #     count = counts[i]
                #     epoch_stats.setdefault(label, 0)
                #     epoch_stats[label] += count
        # always keep a copy of epoch data
        if index == 0 or (self.keep_epoch and self.use_previous_epoch is None):
            self.epoch_data[self.cur_epoch][0][index] = X
            self.epoch_data[self.cur_epoch][1][index] = y
        self.epoch_labels[self.cur_epoch][index] = y
        # can start loading next epoch of training before validation
        # if (index + 1) == len(self):
        #     self.load_next_epoch(True)
        logging.info(
            "%s requsting index %s took %s q has %s",
            self.dataset.name,
            index,
            time.time() - start,
            self.preloader_queue.qsize(),
        )

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
            logging.info("%s loading epoch %s", self.dataset.name, self.loaded_epochs)
            # self.dataset.recalculate_segments(segment_type=self.params.segment_type)
            self.samples = self.dataset.epoch_samples(
                cap_samples=self.cap_samples,
                replace=False,
                random=self.randomize_epoch,
                cap_at=self.cap_at,
                label_cap=self.label_cap,
            )
            holdout_cameras = []

            self.samples = [sample.id for sample in self.samples]
            if self.shuffle:
                np.random.shuffle(self.samples)

            if self.cur_epoch == 0:
                self.sample_size = len(self.samples)
        if self.preload:
            batches_per_process = int(math.ceil(len(self) / self.params.load_threads))
            batches = []
            index = 0
            logging.info(
                "%s num of batches %s bathes per process %s",
                self.dataset.name,
                len(self),
                batches_per_process,
            )
            for i in range(self.params.load_threads):
                batches = []
                for _ in range(batches_per_process):
                    if index >= len(self):
                        break
                    samples = self.samples[
                        index * self.batch_size : (index + 1) * self.batch_size
                    ]
                    index += 1
                    batches.append(samples)
                    # if self.params.load_threads == 1 and len(batches) > 500:
                    #     pickled_batches = pickle.dumps(
                    #         (self.loaded_epochs + 1, batches)
                    #     )
                    #     self.load_queue.put(pickled_batches)
                    #     print(self.dataset.name, "adding", len(batches))
                    #     batches = []

                print(
                    self.dataset.name, "adding", len(batches), self.load_queue.qsize()
                )
                if len(batches) > 0:
                    pickled_batches = pickle.dumps((self.loaded_epochs + 1, batches))
                    self.load_queue.put(pickled_batches)
            # del self.samples[:]
            self.dataset.segments = []
            self.epoch_samples.append(len(self.samples))
            self.samples = np.empty(len(self.samples))
            gc.collect()
        self.loaded_epochs += 1

    def reload_samples(self):

        if self.shuffle:
            np.random.shuffle(self.samples)
        if self.preload:
            for index in range(len(self)):
                samples = self.samples[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]
                pickled_samples = pickle.dumps((self.loaded_epochs + 1, samples))
                self.load_queue.put(pickled_samples)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        # self.load_next_epoch(reuse=True)
        self.sample_size = self.epoch_samples[-1]
        logging.info(
            "%s setting sample size from %s and last epoch was %s",
            self.dataset.name,
            len(self.epoch_samples),
            self.cur_epoch,
        )
        batches = len(self)
        if not self.keep_epoch:
            # zero last epoch
            self.epoch_data[-1] = ([None] * batches, [None] * batches)

        last_stats = self.epoch_stats[self.cur_epoch]
        if len(self.epoch_stats) < self.epochs:
            self.epoch_stats.append({})
            self.epoch_data.append(([None] * batches, [None] * batches))
            self.epoch_labels.append([None] * batches)

        logging.info("epoch ended for %s %s", self.dataset.name, last_stats)
        self.cur_epoch += 1


def loadbatch(labels, db, samples, params, mapped_labels):
    start = time.time()
    # samples = self.samples[index * self.batch_size : (index + 1) * self.batch_size]
    X, y, y_orig = _data(labels, db, samples, params, mapped_labels)
    logging.debug("%s  Time to get data %s", "NULL", time.time() - start)
    return X, y, y_orig


def get_cached_frames(db, sample):
    track_frames = db[sample.unique_track_id]
    frames = []
    for f_i in sample.frame_indices:
        frames.append(track_frames[f_i].copy())
    return frames


#
#
# def _batch_frames(labels, db, samples, params, mapped_labels, to_categorical=True):
#     "Generates data containing batch_size samples"
#     # Initialization
#     start = time.time()
#     X = []
#     y = np.empty((len(samples)), dtype=int)
#     data_i = 0
#     y_original = []
#     for sample in samples:
#         label = mapped_labels[sample.label]
#         if label not in labels:
#             continue
#         channels = [TrackChannels.thermal, TrackChannels.filtered]
#         if params.type == 3:
#             channels.append(TrackChannels.flow)
#
#         try:
#             # frame_data = get_frames(db, sample, channels)
#             frame_data = get_cached_frames(db, sample)
#         except Exception as inst:
#
#             logging.error("Error fetching sample %s %s", sample, inst, exc_info=True)
#             continue
#         if len(frame_data) < 5:
#             logging.error("Not enough frame data for %s %s", sample, len(frame_data))
#             continue
#         y_original.append(sample.label)
#         X.append(frame_data)
#         y[data_i] = labels.index(label)
#         data_i += 1
#     X = X[:data_i]
#     y = y[:data_i]
#     if len(X) == 0:
#         logging.error("Empty length of x")
#
#     y = keras.utils.to_categorical(y, num_classes=len(labels))
#
#     return X, y, y_original


def get_frames(f, segment, channels):
    frames = []
    for frame_i in segment.frame_indices:
        frame_info = segment.track_info[frame_i]
        data = []
        for channel in channels:
            f.seek(frame_info[channel])
            channel_data = np.load(f)
            data.append(channel_data)

        frame = Frame.from_channel(data, channels, frame_i, flow_clipped=True)
        frame.region = tools.Rectangle.from_ltrb(*segment.track_bounds[frame_i])
        frames.append(frame)
    return frames


def process_frames(batch_data, params):
    X = np.empty(
        (
            len(batch_data),
            *params.output_dim,
        )
    )
    data_i = 0
    for frame_data in batch_data:
        # repeat some frames if need be
        while len(frame_data) < params.square_width ** 2:
            missing = params.square_width ** 2 - len(frame_data)
            indices = np.arange(len(frame_data))
            np.random.shuffle(indices)
            for frame_i in indices[:missing]:
                frame_data.append(frame_data[frame_i].copy())
        ref = None
        frame_data = sorted(frame_data, key=lambda frame_data: frame_data.frame_number)
        #
        # for frame in frame_data:
        #     ref.append(sample.frame_temp_median[frame.frame_number])
        data = preprocess_movement(
            None,
            frame_data,
            params.square_width,
            None,
            params.channel,
            preprocess_fn=params.model_preprocess,
            augment=params.augment,
            use_dots=params.use_dots,
            reference_level=ref,
            # sample=sample,
            # overlay=overlay,
            type=params.type,
            keep_edge=params.keep_edge,
        )
        X[data_i] = data
        data_i += 1
    return X


def _data(labels, db, samples, params, mapped_labels, to_categorical=True):
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
    total_db_time = 0
    for sample in samples:
        label = mapped_labels[sample.label]
        if label not in labels:
            continue
        if params.use_movement:
            try:
                data_time = time.time()
                channels = [TrackChannels.thermal, TrackChannels.filtered]
                if params.type == 3:
                    channels.append(TrackChannels.flow)
                frame_data = get_cached_frames(db, sample)
                # frame_data = get_frames(db, sample, channels)
                total_db_time += time.time() - data_time
                # frame_data = dataset.fetch_random_sample(sample)
                overlay = None
                #
                # overlay = dataset.db.get_overlay(
                #     sample.track.clip_id, sample.track.track_id
                # )

            except Exception as inst:
                logging.error(
                    "Error fetching sample %s %s", sample, inst, exc_info=True
                )
                continue

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
                use_dots=params.use_dots,
                reference_level=ref,
                sample=sample,
                overlay=overlay,
                type=params.type,
                keep_edge=params.keep_edge,
            )
            if data is not None:
                mvm.append(sample.movement_data)
        else:
            try:
                frame = dataset.fetch_sample(sample)

            except Exception as inst:
                logging.error("Error fetching samples %s %s", sample, inst)
                continue

            frame = preprocess_frame(
                frame,
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
    # logging.info(
    #     "%s took %s to load db out of total %s",
    #     params.augment,
    #     total_db_time,
    #     total_time,
    # )
    if params.mvm:
        return [np.array(X), np.array(mvm)], y, y_original
    return np.array(X), y, y_original


#
# # continue to read examples until queue is full
# def preloader(q, load_queue, labels, name, db, params, label_mapping):
#     """add a segment into buffer"""
#     logging.info(
#         " -started async fetcher for %s augment=%s",
#         name,883392:   INFO GOT X 4

#         params.augment,
#     )
#     while True:
#         try:
#             if not q.full():
#                 samples = pickle.loads(load_queue.get())
#                 # if not isinstance(samples, tuple):
#                 #     dataset = samples
#                 #     logging.info(
#                 #         " -Updated dataset %s augment=%s",
#                 #         dataset.name,
#                 #         params.augment,
#                 #     )
#                 #     time.sleep(3)  # hack to make sure others get dataset
#                 #     samples = pickle.loads(load_queue.get())
#                 if not isinstance(samples, tuple):
#                     raise Exception("Samples isn't a list", samples)
#                 # datagen.loaded_epochs = samples[0]
#                 data = []
#                 for segment in samples[1]:
#                     if params.use_movement:
#
#                         data.append(segment)
#                         logging.debug(
#                             "adding sample %s %s %s",
#                             segment.id,
#                             data[-1].label,
#                             data[-1].frame_indices,
#                         )
#                     else:
#                         data.append(dataset.frames_by_id[sample_id])
#
#                 q.put(loadbatch(labels, db, data, params, label_mapping))
#
#             else:
#                 logging.debug("Quue is full for %s", name)
#                 time.sleep(0.1)
#         except Exception as e:
#             logging.info("Queue %s error %s ", name, e)
#             raise e


def get_batch_frames(f, frames_by_track, tracks, channels, name):
    start = time.time()
    count = 0
    for track_info, frame_indices, u_id, regions_by_frames in tracks:
        # frames = track[1]
        frame_indices.sort()
        track_data = frames_by_track.setdefault(u_id, {})

        for frame_i in frame_indices:
            if frame_i in track_data:
                continue
            count += 1
            frame_info = track_info[frame_i]
            data = []
            for channel in channels:
                f.seek(frame_info[channel])
                channel_data = np.load(f)
                data.append(channel_data)

            frame = Frame.from_channel(data, channels, frame_i, flow_clipped=True)
            track_data[frame_i] = frame
            frame.region = tools.Rectangle.from_ltrb(*regions_by_frames[frame_i])
    logging.info("%s time to load %s frames %s", name, count, time.time() - start)
    return frames_by_track


def load_batch_frames(
    track_frames,
    seg_data,
    track_seg_count,
    numpyfile,
    batches,
    segments_by_id,
    channels,
    name,
):
    with open(numpyfile, "rb") as f:
        data_by_track = {}
        for batch in batches:
            for s_id in batch:
                segment = segments_by_id[s_id]
                seg_data.append(segments_by_id[s_id])
                track_segments = data_by_track.setdefault(
                    segment.unique_track_id,
                    (segment.track_info, [], segment.unique_track_id, {}),
                )
                regions_by_frames = track_segments[3]
                regions_by_frames.update(segment.track_bounds)
                track_segments[1].extend(segment.frame_indices)

                if segment.unique_track_id in track_seg_count:
                    track_seg_count[segment.unique_track_id] += 1
                else:
                    track_seg_count[segment.unique_track_id] = 1
        # sort by position in file
        track_segments = sorted(
            data_by_track.values(),
            key=lambda track_segment: next(iter(track_segment[0].values()))[
                TrackChannels.thermal
            ],
        )
        track_frames = get_batch_frames(f, track_frames, track_segments, channels, name)
    return track_frames, seg_data, track_seg_count


# continue to read examples until queue is full
def preloader(
    q, load_queue, labels, name, db, segments_by_id, params, label_mapping, numpyfile
):
    """add a segment into buffer"""
    logging.info(
        " -started async fetcher for %s augment=%s numpyfile %s",
        name,
        params.augment,
        numpyfile,
    )
    channels = [TrackChannels.thermal, TrackChannels.filtered]
    if params.type == 3:
        channels.append(TrackChannels.flow)

    track_data = {}

    epoch = 0
    seg_data = []
    track_frames = {}
    track_seg_count = {}
    while True:
        try:
            item = load_queue.get(block=True, timeout=30)
            batches = pickle.loads(item)
            # datagen.loaded_epochs = samples[0]
            epoch = batches[0]
            logging.info(
                "%s Preloader got (%s) batches for epoch %s",
                name,
                len(batches[1]),
                epoch,
            )
            total = 0

            memory_batches = 500
            load_more_at = 32 * memory_batches / 2
            loaded_up_to = 0
            for i, batch in enumerate(batches[1]):
                if len(seg_data) < load_more_at and loaded_up_to < len(batches[1]):

                    next_load = batches[1][loaded_up_to : loaded_up_to + memory_batches]
                    logging.info(
                        "%s loading more data %s have %s loading %s of %s qsize %s ",
                        name,
                        len(seg_data),
                        loaded_up_to,
                        len(next_load),
                        len(batches[1]),
                        q.qsize(),
                    )
                    track_frames, seg_data, track_seg_count = load_batch_frames(
                        track_frames,
                        seg_data,
                        track_seg_count,
                        numpyfile,
                        next_load,
                        segments_by_id,
                        channels,
                        name,
                    )
                    loaded_up_to = i + len(next_load)
                    logging.info(
                        "%s loaded more data %s loaded up to %s",
                        name,
                        len(seg_data),
                        loaded_up_to,
                    )

                data = seg_data[: len(batch)]
                batch_data = loadbatch(
                    labels, track_frames, data, params, label_mapping
                )
                while True:
                    try:
                        q.put(batch_data, block=True, timeout=30)
                        break
                    except (queue.Full):
                        logging.debug("%s Batch Queue full epoch %s", name, epoch)
                    except Exception as e:
                        logging.error(
                            "%s - %s batch %s Put error %s",
                            epoch,
                            name,
                            i,
                            e,
                            exc_info=True,
                        )
                for seg in data:
                    track_seg_count[seg.unique_track_id] -= 1
                    if track_seg_count[seg.unique_track_id] <= 0:
                        del track_frames[seg.unique_track_id]
                        del track_seg_count[seg.unique_track_id]

                total += 1
                logging.debug(
                    "%s put %s out of %s %s",
                    total,
                    name,
                    len(batches[1]),
                    q.qsize(),
                )
                seg_data = seg_data[len(data) :]
        except (queue.Empty):
            logging.debug("%s Samples Queue empty epoch %s", name, epoch)

        except Exception as inst:
            logging.error("%s epoch %s error %s", name, epoch, inst, exc_info=True)
            pass
