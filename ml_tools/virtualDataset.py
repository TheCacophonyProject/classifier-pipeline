"""

Handles streaming of a large segment dataset from disk.

Segment information is loaded into memory, but segment data is loads as needed by VirtualDataset.next_batch

"""

import os.path

import queue
import threading
import multiprocessing

import time
import numpy as np
import pickle
from ml_tools import tools
import scipy
import random

from bisect import bisect

class SegmentInfo():

    def __init__(self, source, offset, index, weight, tag, average_mass):
        # source file this segment comes from
        self.source = source

        # first frame of this segment
        self.offset = offset

        # relative weight of the segment (higher is sampled more often)
        self.weight = weight

        # index of the segment, i.e. 1 = the 2nd segment in the track.
        self.index = index

        self.tag = tag.lower()

        self.average_mass = average_mass

    @property
    def save_path(self):
        base_name = os.path.splitext(os.path.basename(self.source))[0]
        return os.path.join('c:\\cac', 'dataset', base_name + '-' + str(self.offset) + ".dat")

    def __str__(self):
        return "offset:{0} weight:{1:.1f}".format(self.offset, self.weight)

def overlap_ratio(rect1, rect2):
    """
    Compute the area overlap between rect1 and rect2.  Return overlap as ratio of area of first rect.
    :param rect1: (left,top,right,bottom)
    :param rect2: (left,top,right,bottom)
    :return: area overlaped
    """
    left1, top1, right1, bottom1 = rect1
    left2, top2, right2, bottom2 = rect2
    x_overlap = max(0, min(right1, right2) - max(left1, left2))
    y_overlap = max(0, min(bottom1, bottom2) - max(top1, top2))
    return (x_overlap * y_overlap) / (abs(right1-left1)*abs(bottom1-top1))


class TrackInfo():
    """ Stores information about a track for virtual database. """

    # segments are 3 seconds long
    SEGMENT_WIDTH = 9 * 3

    def __init__(self, source):
        self.source = source
        self.stats = None
        self.segments = None
        self.weight = None

    def __str__(self):
        return "{0} segments:{1} weight:{2:.1f}".format(self.source, len(self.segments), self.weight)

    def get_segments_and_stats(self, min_segment_mass=None, max_segment_mass=None, max_cropping=None, segment_spacing = 9,
                               labels_to_ignore_mass_threshold = None):
        """
        Gets track statistics and segments for track.
        Returns number of tracks loaded, and number filtered
        """

        stats_filename = os.path.splitext(self.source)[0] + ".txt"

        if not os.path.exists(stats_filename):
            raise Exception("Stats file not found for track {0}".format(self.source))

        stats = tools.load_track_stats(stats_filename)

        label = stats['tag']

        mass_history = stats['mass_history']
        bounds_history = stats['bounds_history']
        crop_history = [1-overlap_ratio(bounds, (0,0,180,120)) for bounds in bounds_history]

        segment_offsets = []
        segment_masses = []

        discarded_segments = 0

        for i in range(len(mass_history) // segment_spacing):

            segment_start = i * segment_spacing

            segment_frames = len(mass_history[segment_start:segment_start + TrackInfo.SEGMENT_WIDTH])
            segment_mass = sum(mass_history[segment_start:segment_start + TrackInfo.SEGMENT_WIDTH]) / segment_frames
            segment_min_mass = min(mass_history[segment_start:segment_start + TrackInfo.SEGMENT_WIDTH])
            segment_crop = sum(crop_history[segment_start:segment_start + TrackInfo.SEGMENT_WIDTH]) / segment_frames

            # might have a short segment on the last iteration, so discard it here
            if segment_frames != TrackInfo.SEGMENT_WIDTH:
                continue

            if labels_to_ignore_mass_threshold is None or label not in labels_to_ignore_mass_threshold:

                # if any frame has less than half the required mass then remove the entire segment.
                if min_segment_mass is not None and segment_min_mass < min_segment_mass / 2:
                    discarded_segments += 1
                    continue

                # make sure segment has enough mass to be useful
                if min_segment_mass is not None and segment_mass < min_segment_mass:
                    discarded_segments += 1
                    continue

                if max_segment_mass is not None and segment_mass > max_segment_mass:
                    discarded_segments += 1
                    continue

                # check how cropped each segment is
                if max_cropping is not None and segment_crop > max_cropping:
                    discarded_segments += 1
                    continue

            segment_offsets.append(segment_start)
            segment_masses.append(segment_mass)

        self.stats = stats

        # first we choose a track to use, weighted by the square root of the number of segments.
        # this means that longer tracks will not dominate the smaller ones.
        self.weight = (len(segment_offsets) ** 0.5)

        self.segments = []
        for index, offset, mass in zip(range(len(segment_offsets)), segment_offsets, segment_masses):
            self.segments.append(
                SegmentInfo(self.source, offset, index, self.weight / len(segment_offsets), label, mass))

        return len(segment_offsets), discarded_segments


# continue to read examples until queue is full
def preloader(q, dataset, buffer_size):
    """ add a segment into buffer """
    print("Async loader pool starting")
    while not dataset.shutdown_worker_threads:
        if q.qsize() < buffer_size:
            q.put(dataset.next_batch(1, disable_async=True))
        else:
            time.sleep(0.01)

class VirtualDataset():
    """ A virtual dataset that streams examples from disk rather than storing the entire dataset in memory. """

    def __init__(self, name="Dataset"):
        """ Create the virutal dataset """

        # tracks by source name
        self.tracks = {}

        # each segment must have an average active pixel count at least equal to this
        self.min_segment_mass = 10
        self.max_segment_mass = None
        # a list of labels that should ignore the mass thresholds
        self.ignore_segment_mass_thresholds = set()

        # spacing in frames between segments
        self.segment_spacing = 9

        # Array containing all segments in this dataset
        self.segments = []

        # Cumulative probability distribution for segments.  Allows for super fast weighted random sampling.
        self.segment_cdf = []

        self.name = name

        self.classes = []
        self.class_index = {}

        # A counter for how many segments where disgarded due to low mass
        self.disgarded_segment_count = 0

        # this allows for some minipulateion of data during the sampling stage.
        self.enable_augmentation = False

        self.preloader_queue = None
        self.preloader_threads = None

        self.reloader_stop_flag = False

        self.shutdown_worker_threads = False

        self.normalisation_constants = []


    @property
    def rows(self):
        # we enflate the rows in single frame mode so that training lasts longer.  We randomly sample the frames when
        # calling next batch to 10 segments will actaully be 10 * TrackInfo.SEGMENT_WIDTH frames.
        return len(self.segments)

    """
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['preload_queue']
        del d['preloader_pool']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.preloader_pool = None
        self.preloader_queue = None
    """

    def start_async_load(self, buffer_size = 64):
        """
        Starts async load process.
        """

        WORKER_THREADS = 2

        print("Starting async fetcher")
        self.preloader_queue = queue.Queue()
        self.preloader_threads = [threading.Thread(target=preloader, args=(self.preloader_queue, self, buffer_size)) for _ in range(WORKER_THREADS)]
        self.reloader_stop_flag = False
        for thread in self.preloader_threads:
            thread.start()

    def stop_async_load(self, buffer_size = 64):
        """
        Stops async worker thread.
        """
        if self.preloader_threads is not None:
            self.reloader_stop_flag = True
            for thread in self.preloader_threads:
                thread.join()

    def next_batch(self, n, disable_async=False):
        """
        Returns a batch of n items (X, y) from dataset.
        If enabled augmentation will be applied.
        """
        # if async is enabled use it.
        if not disable_async and self.preloader_queue is not None:
            # get samples from queue
            batch_X = []
            batch_y = []
            for _ in range(n):
                X, y = self.preloader_queue.get(block = True, timeout=4.0)
                batch_X.append(X[0])
                batch_y.append(y[0])

            return np.asarray(batch_X), np.asarray(batch_y)

        batch_X = []
        batch_y = []
        for i in range(n):
            index =  self.sample()
            sample_data = self.fetch_segment(self.segments[index], augment=self.enable_augmentation)
            sample_tag = self.segments[index].tag
            class_index = self.class_index[sample_tag]

            batch_X.append(sample_data)
            batch_y.append(class_index)

        X = np.asarray(batch_X)
        y = np.asarray(batch_y)

        return (X, y)

    def get_segment_by_name(self, name):
        """ returns segment header by name.  Peforms a linear scan, so very slow. """
        for segment in self.segments:
            if segment.source == name:
                return segment
        return None

    def add_tracks(self, new_tracks):
        """ Add tracks to data sets track list.  Returns number of segments added. """

        new_segments = 0

        for track_name in new_tracks:

            # this track is already in the dataset.
            if track_name in self.tracks:
                continue

            track = TrackInfo(track_name)
            _, segments_disgarded = track.get_segments_and_stats(
                min_segment_mass=self.min_segment_mass, max_segment_mass=self.max_segment_mass,
                segment_spacing=self.segment_spacing,
                labels_to_ignore_mass_threshold=self.ignore_segment_mass_thresholds
            )
            self.disgarded_segment_count += segments_disgarded

            self.tracks[track_name] = track

            # create segment references
            self.segments += track.segments
            new_segments += len(track.segments)

        self.rebuild_cdf()

        self.classes = list(set(segment.tag for segment in self.segments))
        self.classes.sort()
        for index, class_name in enumerate(self.classes):
            self.class_index[class_name] = index

        return new_segments

    def rebuild_cdf(self):
        """ Calculates the CDF used for fast random sampling """
        self.segment_cdf = []
        prob = 0
        for segment in self.segments:
            prob += segment.weight
            self.segment_cdf.append(prob)
        normalizer = self.segment_cdf[-1]
        self.segment_cdf = [x / normalizer for x in self.segment_cdf]

    def fetch_segment(self, segment: SegmentInfo, include_padding=False, normalise=True, augment=False):
        """ Fetch a single segments from disk from the prewritten segment cache. """
        if os.path.exists(segment.save_path):
            data = pickle.load(open(segment.save_path, 'rb'))
            if augment: data = self.apply_augmentation(data)
            if normalise: data = self.apply_normalisation(data)
            return data if include_padding else data[:27]

    def get_raw_track_segments(self, track: TrackInfo, padding = 0):
        """
        Fetch all segments for a track from disk.  These segments will be unnormalised
        :param track:
        :param padding: number of additional frames to try and add at the end of each segment.
        :return:
        """

        # first we need to load the track
        try:
            save_file = pickle.load(open(track.source, 'rb'))
        except Exception as e:
            # pass the error on
            raise Exception("Error loading '{0}': {1}".format(track.source, e))

        # next fetch the frames
        frames = np.asarray(save_file['frames'], dtype=np.float32)  # 32bit for preprocessing
        filtered = np.asarray(save_file['filtered_frames'], dtype=np.float32)  # 32bit for preprocessing
        flow = np.asarray(save_file['flow_frames'], dtype=np.float32)  # 32bit for preprocessing
        mask = np.asarray(save_file['mask_frames'], dtype=np.float32)  # 32bit for preprocessing

        frame_count, width, height = frames.shape

        segment_data_list = []
        for segment in track.segments:
            segment_width = len(frames[segment.offset:segment.offset + TrackInfo.SEGMENT_WIDTH + padding])
            segment_data = np.zeros((segment_width, width, height, 5), dtype=np.float16)
            segment_data[:, :, :, 0] = frames[segment.offset:segment.offset + TrackInfo.SEGMENT_WIDTH + padding]
            segment_data[:, :, :, 1] = filtered[segment.offset:segment.offset + TrackInfo.SEGMENT_WIDTH + padding]
            segment_data[:, :, :, 2:3+1] = flow[segment.offset:segment.offset + TrackInfo.SEGMENT_WIDTH + padding]
            segment_data[:, :, :, 4] = mask[segment.offset:segment.offset + TrackInfo.SEGMENT_WIDTH + padding]
            segment_data_list.append(segment_data)

        return segment_data_list

    def fetch_all(self, augment=False, normalise=True):
        """ Fetches all segments from data set, returns (X, y) """

        data = np.zeros((len(self.segments), TrackInfo.SEGMENT_WIDTH, 64, 64, 5), dtype=np.float16)
        labels = []

        for i, segment in enumerate(self.segments):
            data[i] = self.fetch_segment(segment, augment=augment, normalise=normalise)
            labels.append(self.class_index[segment.tag])

        return (data, np.asarray(labels))

    def write_out_segments(self, padding=9, overwrite=False):
        """
        Writes all segments to disk in an easily to load form.
        :param padding: addtional frames to add at the end of the segment.  Allows for jittering the first frame.
        :param overwrite:
        :return:
        """
        for track_name, track in self.tracks.items():

            segments = self.tracks[track_name].segments
            required_segments = set()

            # check if we can skip loading this track
            if overwrite:
                required_segments = set(segments)
            else:
                for segment in segments:
                    if not os.path.exists(segment.save_path):
                        required_segments.add(segment)

            if len(required_segments) == 0:
                continue

            segments_data = self.get_raw_track_segments(track, padding)
            for segment_info, segment_data in zip(self.tracks[track_name].segments, segments_data):
                path = os.path.dirname(segment_info.save_path)
                if segment_info in required_segments:
                    if not os.path.exists(path):
                        os.makedirs(path)
                    pickle.dump(segment_data, open(segment_info.save_path, 'wb'))

    def get_class_weight(self, class_name):
        """ Returns the total weight for all segments of given class. """
        return sum(segment.weight for segment in self.segments if segment.tag == class_name)

    def get_class_segments_count(self, class_name):
        """ Returns the total weight for all segments of given class. """
        return len(self.get_class_segments(class_name))

    def get_class_segments(self, class_name):
        """ Returns the total weight for all segments of given class. """
        return [segment for segment in self.segments if segment.tag == class_name]

    def weight_balance(self, weight_modifiers=None):
        """
        Adjusts weights so that every class is evenly represented.
        :param weight_modifiers: if specified is a dictionary mapping from class name to weight modifier,
            where < 1 sampled less frequently, and > 1 is sampled more frequently.
        :return:
        """

        class_weight = {}
        mean_class_weight = 0

        for class_name in self.classes:
            class_weight[class_name] = self.get_class_weight(class_name)
            mean_class_weight += class_weight[class_name] / len(self.classes)

        scale_factor = {}
        for class_name in self.classes:
            modifier = 1.0 if weight_modifiers is None else weight_modifiers.get(class_name, 1.0)
            scale_factor[class_name] = mean_class_weight / class_weight[class_name] * modifier

        for segment in self.segments:
            segment.weight *= scale_factor[segment.tag]

        self.rebuild_cdf()

    def resample_balance(self, max_ratio=2.0, weight_modifiers=None):
        """ Removes segments until all classes are with given ratio """

        class_count = {}
        min_class_count = 9999999

        for class_name in self.classes:
            class_count[class_name] = self.get_class_segments_count(class_name)
            modifier = 1.0 if weight_modifiers is None else weight_modifiers.get(class_name, 1.0)
            if modifier > 0:
                min_class_count = min(min_class_count, class_count[class_name] / modifier)

        max_class_count = int(min_class_count * max_ratio)

        new_segments = []

        for class_name in self.classes:
            segments = self.get_class_segments(class_name)
            if len(segments) > max_class_count:
                # resample down
                modifier = 1.0 if weight_modifiers is None else weight_modifiers.get(class_name, 1.0)
                segments = np.random.choice(segments, int(max_class_count * modifier), replace=False).tolist()
            new_segments += segments

        self.segments = new_segments

        segment_set = set(self.segments)

        # remove segments from tracks
        for track_name, track in self.tracks.items():
            segments = track.segments
            segments = [segment for segment in segments if (segment in segment_set)]
            track.segments = segments

        self.rebuild_cdf()

    def apply_normalisation(self, data):
        # apply normalisation
        for channel in range(5):
            # note we switch up to float32 here to get enough precision for the preprocessing.
            offset, scale, power = self.normalisation_constants[channel]
            frame = np.float32(data[:, :, :, channel])
            frame = (frame + offset)
            if power != 1:
                frame = np.power(np.abs(frame), power) * np.sign(frame)
            frame = frame / scale
            data[:, :, :, channel] = frame

        return data

    def apply_augmentation(self, segment):
        """ Applies a random augmentation to the segment. """

        # jitter first frame
        extra_frames = len(segment) - 27
        offset = random.randint(0, extra_frames)
        segment = segment[offset:offset+27]

        mask = segment[:, :, :, 4]

        # apply scaling
        if random.randint(0, 1) == 0:
            av_mass = np.sum(mask) / len(mask)
            scale_options = []
            if av_mass > 50: scale_options.append('down')
            if av_mass < 100: scale_options.append('up')
            scale_method = np.random.choice(scale_options)

            up_scale = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
            down_scale = [0.25, 0.33, 0.5, 0.75]

            if scale_method == 'up':
                scale = np.random.choice(up_scale)
            else:
                scale = np.random.choice(down_scale)

            for i in range(27):
                segment[i,:,:] = tools.clipped_zoom(np.float32(segment[i,:,:]), scale, order=1)

        # introduce noise onto non masked area.
        # this doesn't work so well as we need to also add noise during testing, which I don't want to do.
        # it's also quite slow
        """
        noise = np.random.normal(loc=0.0, scale=1, size=[27, 64, 64])
        noise = noise * (1 - mask)
        segment[:, :, :, 1] += noise
        segment[:, :, :, 2:4] += noise[:, :, :, np.newaxis] * 0.2
        """

        if random.randint(0,1) == 0:
            return np.flip(segment, axis = 2)
        else:
            return segment

    def sample(self):
        """
        Randomly sample a segment from dataset.
        :return: returns index of sample
        """
        roll = random.random()
        index = bisect(self.segment_cdf, roll)
        return index
