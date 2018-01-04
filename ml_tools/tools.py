"""
Helper functions for classification of the tracks extracted from CPTV videos
"""

import os.path
import PIL as pillow
import numpy as np
import random
import pickle
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import itertools
import gzip
import json
import dateutil
import binascii
import time
import datetime
import glob
import cv2
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
import subprocess
import scipy

EPISON = 1e-5

# the coldest value to display when rendering previews
TEMPERATURE_MIN = 2800
TEMPERATURE_MAX = 4200


class Rectangle:
    """ Defines a rectangle by the topleft point and width / height. """
    def __init__(self, topleft_x, topleft_y, width, height):
        """ Defines new rectangle. """
        self.x = topleft_x
        self.y = topleft_y
        self.width = width
        self.height = height

    def copy(self):
        return Rectangle(self.x, self.y, self.width, self.height)

    @property
    def mid_x(self):
        return self.x + self.width / 2

    @property
    def mid_y(self):
        return self.y + self.height / 2

    @property
    def left(self):
        return self.x

    @property
    def top(self):
        return self.y

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    def overlap_area(self, other):
        """ Compute the area overlap between this rectangle and another. """
        x_overlap = max(0, min(self.right, other.right) - max(self.left, other.left))
        y_overlap = max(0, min(self.bottom, other.bottom) - max(self.top, other.top))
        return x_overlap * y_overlap

    @property
    def area(self):
        return self.width * self.height

    def __repr__(self):
        return "({0},{1},{2},{3})".format(self.left, self.top, self.right, self.bottom)

    def __str__(self):
        return "<({0},{1})-{2}x{3}>".format(self.x, self.y, self.width, self.height)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
            # Let the base class default method raise the TypeError
        if isinstance(obj, Rectangle):
            return int(obj.left), int(obj.top), int(obj.right), int(obj.bottom)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def purge(dir, pattern):
    for f in glob.glob(os.path.join(dir, pattern)):
        os.remove(os.path.join(dir, f))

def find_file(root, filename):
    """
    Finds a file in root folder, or any subfolders.
    :param root: root folder to search file
    :param filename: exact time of file to look for
    :return: returns full path to file or None if not found.
    """
    for root, dir, files in os.walk(root):
        if filename in files:
            return os.path.join(root, filename)
    return None


def get_ffmpeg_command(filename, width, height, quality=21):
    if os.name == 'nt':
        FFMPEG_BIN = "ffmpeg.exe"  # on Windows
    else:
        FFMPEG_BIN = "ffmpeg"  # on Linux ans Mac OS

    command = [
        FFMPEG_BIN,
        '-y',  # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', str(width) + 'x' + str(height),  # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '9',  # frames per second
        '-i', '-',  # The imput comes from a pipe
        '-an',  # Tells FFMPEG not to expect any audio
        '-vcodec', 'libx264',
        '-tune', 'grain',  # good for keepinn the grain in our videos
        '-crf', str(quality),  # quality, lower is better
        '-pix_fmt', 'yuv420p',  # window thumbnails require yuv420p for some reason
        filename
    ]
    return command


def stream_mpeg(filename, frame_generator):
    """
    Saves a
    Saves a sequence of rgb image frames as an MPEG video.
    :param filename: output filename
    :param frame_generator: generator that produces numpy array of shape [height, width, 3] of type uint8
    """

    first_frame = next(frame_generator)
    height, width, channels = first_frame.shape

    command = get_ffmpeg_command(filename, width, height)

    # write out the data.
    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        for frame in frame_generator:
            process.stdin.write(frame.tostring())
            process.stdin.flush()

        process.stdin.close()
        process.stderr.close()
        process.stdout.close()
        return_code = process.wait(timeout=60)
        if return_code != 0:
            raise Exception("FFMPEG failed with error {}".format(return_code))
    except Exception as e:
        print("Failed to write MPEG:", e)
        if process is not None:
            print("out:", process.stdout)
            print("error:", process.stderr)


def write_mpeg(filename, frames):
    """
    Saves a sequence of rgb image frames as an MPEG video.
    :param filename: output filename
    :param frames: numpy array of shape [frame, height, width, 3] of type uint8
    """

    # we may have passed a list of frames, if so convert to a 3d array.
    frames = np.asarray(frames, np.uint8)

    if frames is None or len(frames) == 0:
        # empty video
        return

    frame_count, height, width, channels = frames.shape

    command = get_ffmpeg_command(filename, width, height)

    # write out the data.
    try:
        process = None
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=frames.tostring())
        process.check_returncode()
    except Exception as e:
        print("Failed to write MPEG:", e)
        if process is not None:
            print("out:", process.stdout)
            print("error:", process.stderr)

def load_colormap(filename):
    """ Loads a custom colormap used for creating MPEG previews of tracks. """

    if not os.path.exists(filename):
        return

    return pickle.load(open(filename, 'rb'))

def convert_heat_to_img(frame, colormap, temp_min = 2800, temp_max = 4200):
    """
    Converts a frame in float32 format to a PIL image in in uint8 format.
    :param frame: the numpy frame contining heat values to convert
    :param colormap: an optional colormap to use, if none is provided then tracker.colormap is used.
    :return: a pillow Image containing a colorised heatmap
    """
    # normalise
    frame = (frame - temp_min) / (temp_max - temp_min)
    colorized = np.uint8(255.0*colormap(frame))
    img = pillow.Image.fromarray(colorized[:,:,:3]) #ignore alpha
    return img


def most_common(lst):
    return max(set(lst), key=lst.count)

def is_gz_file(filepath):
    """ returns if file is a gzip file or not"""
    with open(filepath, 'rb') as test_f:
        return binascii.hexlify(test_f.read(2)) == b'1f8b'


def normalise(x):
    # we apply a small epison so we don't divide by zero.
    return (x - np.mean(x)) / (EPISON + np.std(x))


def load_tracker_stats(filename):
    """
    Loads a stats file for a processed clip.
    :param filename: full path and filename to stats file
    :return: returns the stats file
    """
    with open(filename, 'r') as t:
        # add in some metadata stats
        stats = json.load(t)

    stats['date_time'] = dateutil.parser.parse(stats['date_time'])

    return stats


def load_clip_metadata(filename):
    """
    Loads a metadata file for a clip.
    :param filename: full path and filename to stats file
    :return: returns the stats file
    """
    with open(filename, 'r') as t:
        # add in some metadata stats
        stats = json.load(t)

    stats['recordingDateTime'] = dateutil.parser.parse(stats['recordingDateTime'])

    return stats


def load_track_stats(filename):
    """
    Loads a stats file for a processed track.
    :param filename: full path and filename to stats file
    :return: returns the stats file
    """

    with open(filename, 'r') as t:
        # add in some metadata stats
        stats = json.load(t)

    stats['timestamp'] = dateutil.parser.parse(stats['timestamp'])
    return stats


def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    print("Creating new GPU session with memory growth enabled.")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8 # save some ram for other applications.

    session = tf.Session(config=config)

    return session


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def compute_saliency_map(X_in, y_in, model):
    """
    Compute a class saliency map for segment using the model for image X and label y.
    :param X_in: segment of shape [frames, height, width, channels]
    :param y_in: label index
    :param model: the model to use classify the segment
    :return: the saliency map of shape [frames, height, width]
    """

    correct_scores = tf.gather_nd(model.pred,
                                  tf.stack((tf.range(X_in.shape[0], dtype="int64"), model.y), axis=1))

    # normalise thermal and filtered channels
    for i in [0, 1]:
        X_in[:, :, :, :, i] = normalise(X_in[:, :, :, :, i])

    feed_dict = {
        model.X: X_in,
        model.y: y_in,
    }

    grads = tf.abs(tf.gradients(correct_scores, model.X)[0])
    saliency = model.sess.run([grads], feed_dict=feed_dict)[0]

    return saliency


def show_saliency_map(model, X_in, y_in):
    """
    Plots saliency map for single segment via pyplot.
    :param model: the model to use to generate the saliency maps
    :param X_in: segment of shape [frames, height, width, channels]
    :param y_in: label index
    """

    X = np.asarray(X_in, dtype=np.float32)
    y = np.asarray(y_in, dtype=np.int32)

    saliency = compute_saliency_map(X[np.newaxis, :, :, :, :], y[np.newaxis], model)[0]
    saliency = saliency / (np.max(saliency) + 1e-8)

    cols = X.shape[0]

    rows = 8

    for frame in range(cols):

        # plot original image

        # plot all 4 channels plus original
        for channel in range(4):

            plt.subplot(rows, cols, (cols * channel) + frame + 1)
            frame_data = X[frame, :, :, channel].astype('float32')
            if channel in [2, 3]:
                # for motion vectors it's better to use magnitude when drawing them
                frame_data = np.abs(frame_data)

            plt.imshow(frame_data, aspect='auto', vmin=-1, vmax=10)
            plt.axis('off')

            plt.subplot(rows, cols, (cols * (channel + 4)) + frame + 1)
            plt.imshow(saliency[frame, :, :, channel], vmin=0.0, vmax=0.7, cmap=plt.cm.hot, aspect='auto')
            plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.gcf().set_size_inches(cols * 3, rows * 3)
    plt.show()

def show_segment(X_in):
    """
    Displays all channels of a segment
    :param X_in: segment of shape [frames, height, width, channels]
    """

    X = np.asarray(X_in, dtype=np.float32)

    cols = X.shape[0]

    rows = 4

    for frame in range(cols):

        # plot original image

        # plot all 4 channels plus original
        for channel in range(4):

            plt.subplot(rows, cols, (cols * channel) + frame + 1)
            frame_data = X[frame, :, :, channel].astype('float32')
            if channel in [2, 3]:
                # for motion vectors it's better to use magnitude when drawing them
                frame_data = np.abs(frame_data) * 5

            plt.imshow(frame_data, aspect='auto', vmin=-10, vmax=10)
            plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.gcf().set_size_inches(cols * 3, rows * 3)
    plt.show()


def get_image_subsection(image, bounds, window_size, boundary_value=None):
    """
    Returns a subsection of the original image bounded by bounds.
    Area outside of frame will be filled with boundary_value.  If None the median value will be used.
    """

    # todo: rewrite this using opencv's built in method
    # cropping method.  just center on the bounds center and take a section there.

    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]

    # for some reason I write this to only work with even window sizes?
    window_half_width, window_half_height = window_size[0] // 2, window_size[1] // 2
    window_size = (window_half_width * 2, window_half_height * 2)
    image_height, image_width, channels = image.shape

    # find how many pixels we need to pad by
    padding = (max(window_size)//2)+1

    midx = int(bounds.mid_x + padding)
    midy = int(bounds.mid_y + padding)

    if boundary_value is None: boundary_value = np.median(image)

    # note, we take the median of all channels, should really be on a per channel basis.
    enlarged_frame = np.ones([image_height + padding*2, image_width + padding*2, channels], dtype=np.float16) * boundary_value
    enlarged_frame[padding:-padding,padding:-padding] = image

    sub_section = enlarged_frame[midy-window_half_width:midy+window_half_width, midx-window_half_height:midx+window_half_height]

    width, height, channels = sub_section.shape
    if int(width) != window_size[0] or int(height) != window_size[1]:
        print("Warning: subsection wrong size. Expected {} but found {}".format(window_size,(width, height)))

    if channels == 1:
        sub_section = sub_section[:,:,0]

    return sub_section

def to_HWC(data):
    """ converts from CHW format to HWC format. """
    return np.transpose(data, axes=(1, 2, 0))

def to_CHW(data):
    """ converts from HWC format to CHW format. """
    return np.transpose(data, axes=(2, 0, 1))


def zoom_image(img, scale, pad_with_min=False, channels_first=False):
    """
    Zooms into or out of the center of the image.  The dimensions are left unchanged, and either padding is added, or
    cropping is performed.
    :param img: image to process of shape [height, width, channels]
    :param scale: how much to scale image
    :param pad_with_min: if true shrunk images will pad with the channels min value (otherwise 0 is used)
    :param channels_first: if true uses [channels, height, width] format
    :return: the new image
    """

    if scale == 1:
        return img

    if channels_first:
        img = to_HWC(img)
    width, height, channels = img.shape
    if scale < 1:
        # scale down and pad
        new_height, new_width = int(height * scale), int(width * scale)

        # note:
        # cv2.INTER_AREA would be better, but sometimes bugs out for certian scales, not sure why.
        res = cv2.resize(np.float32(img), (new_height, new_width), interpolation=cv2.INTER_LINEAR)
        pad_width = (width - new_width) / 2
        pad_height = (height - new_height) / 2
        if pad_with_min:
            min_values = [np.min(res[:, :, i]) for i in range(channels)]
            img = np.ones([width, height, channels], dtype=np.float32) * min_values
        else:
            img = np.zeros([width, height, channels], dtype=np.float32)

        img[int(pad_height):int(pad_height) + new_height, int(pad_width):int(pad_width) + new_width, :] = res
    else:
        # crop and scale up
        crop_height, crop_width = int(height / scale), int(width / scale)
        inset_width = int((width - crop_width) / 2)
        inset_height = int((height - crop_height) / 2)
        crop = img[inset_height:inset_height + crop_height, inset_width:inset_width + crop_width]
        img = cv2.resize(np.float32(crop), dsize=(height, width), interpolation=cv2.INTER_CUBIC)

    if channels_first:
        img = to_CHW(img)

    return img

def read_track_files(track_folder, min_tracks = 50, ignore_classes = ['false-positive'], track_filter = None):
    """ Read in the tracks files from folder. Returns tupple containing list of class names and dictionary mapping from
        class name to list of racks."""

    # gather up all the tracks we can find and show how many of each class exist

    folders = [os.path.join(track_folder, f) for f in os.listdir(track_folder) if os.path.isdir(os.path.join(track_folder, f)) \
               and f not in ignore_classes]

    class_tracks = {}
    classes = []

    num_filtered = {}

    for folder in folders:

        class_name = os.path.basename(folder)

        if class_name in ignore_classes:
            continue

        trk_files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() == '.trk']

        if track_filter:
            filtered_files = []
            for track in trk_files:
                stats_filename = os.path.splitext(track)[0]+".txt"
                if os.path.exists(stats_filename):
                    stats = load_track_stats(stats_filename)
                else:
                    continue
                # stub: some data is excluded from track stats and only found in the clip stats, so we read it in here.
                base_name = os.path.splitext(track)[0]
                clip_stats_filename = base_name[:base_name.rfind('-')] + ".txt"
                if os.path.exists(clip_stats_filename):
                    clip_stats = load_tracker_stats(clip_stats_filename)
                    stats['source'] = base_name
                    stats['event'] = clip_stats.get('event','none')
                else:
                    continue
                if track_filter(stats):
                    filtered_files.append(track)

        else:
            filtered_files = trk_files

        num_filtered[class_name] = len(trk_files) - len(filtered_files)

        if len(filtered_files) < min_tracks:
            print("Warning, too few tracks ({1}) to process for class {0}".format(class_name, len(filtered_files)))
            continue

        class_tracks[class_name] = filtered_files
        classes.append(class_name)

    for class_name in classes:
        filter_string = "" if num_filtered[class_name] == 0 else "({0} filtered)".format(num_filtered[class_name])
        print("{0:<10} {1} tracks {2}".format(class_name, len(class_tracks[class_name]), filter_string))

    return classes, class_tracks

def get_classification_info(model, batch_X, batch_y):
    """
    Gets classification predictions for a batch then returns a list of classification errors as well as some statistics.
    """

    incorrectly_classified_segments = []

    predictions = model.classify_batch(batch_X)

    pred_class = []
    true_class = []
    confidences = []

    for prediction, X, correct_label in zip(predictions, batch_X, batch_y):
        predicted_label = np.argmax(prediction)
        predicted_confidence = prediction[predicted_label]
        if correct_label != predicted_label:
            incorrectly_classified_segments.append(X)
        pred_class.append(predicted_label)
        true_class.append(correct_label)
        confidences.append(predicted_confidence)

    return incorrectly_classified_segments, pred_class, true_class, confidences

def show_confusion_matrix(pred_class, true_class, classes, normalize = True):
    """ Display a confusion matrix from list of results. """

    cm = confusion_matrix([classes[class_num] for class_num in pred_class],
                          [classes[class_num] for class_num in true_class], labels=classes)

    # normalise matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)

    plt.title("Classification Confusion Matrix")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_xticklabels([''] + classes, rotation=45)
    ax.set_yticklabels([''] + classes)
    ax.xaxis.set_tick_params(labeltop='off', labelbottom='on')
    plt.show()


# todo: change this to classify track.
def classify_segment(model, segment, verbose = False):
    """ Loop through frames in the segment, classifying them, then output a probability distribution of the class
        of this segment """
    frames = segment.shape[0]

    prediction_history = []
    confidence_history = []
    confidence = np.ones([len(model.num_classes)]) * 0.05

    # todo: could be much faster if we simply batch them.

    for i in range(frames):

        feed_dict = {model.X: segment[i:i+1]}

        prediction = model.pred_out.eval(feed_dict = feed_dict, session = model.sess)[0]

        # bayesian update
        # this responds much quicker to strong changes in evidence, which is more helpful I think.
        confidence = (confidence * prediction) / (confidence * prediction + (confidence * (1.0-prediction)))

        confidence = np.clip(confidence, 0.05, 0.95)

        confidence_history.append(confidence)
        prediction_history.append(prediction)

    image = np.asarray(confidence_history)
    image[image < 0.5] = 0
    if verbose:
        print(image.min(), image.max())
        print(prediction)
        plt.imshow(image, aspect = "auto", vmin=0, vmax = 1.0)
        plt.show()
    # divide by temperature to smooth out confidence (i.e. decrease confidence when there are competing categories)
    predicted_class = softmax(np.sum(confidence_history, axis = 0)/10.0)
    return predicted_class


def blosc_opts(complevel=9, complib='blosc:lz4', shuffle=True):
    """ Gets params to pass for blosc compression.  Requires tables to be imported. """
    shuffle = 2 if shuffle == 'bit' else 1 if shuffle else 0
    compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
    complib = ['blosc:' + c for c in compressors].index(complib)
    args = {
        'compression': 32001,
        'compression_opts': (0, 0, 0, 0, complevel, shuffle, complib)
    }
    if shuffle:
        args['shuffle'] = False
    return args


blosc_zstd = blosc_opts(complevel=9, complib='blosc:zstd', shuffle=True)

color_dict = {'red': ((0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (1.0, 1.0, 1.0)),

              'green': ((0.0, 1.0, 1.0),
                        (0.5, 0.0, 0.0),
                        (1.0, 0.5, 0.8)),

              'blue': ((0.0, 0.3, 0.3),
                       (0.5, 0.0, 0.0),
                       (1.0, 0.1, 0.1))
              }
cm_blue_red = LinearSegmentedColormap('BlueRed2', color_dict)
