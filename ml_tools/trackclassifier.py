"""
Module classify a tracking window based on a 3 second segment.
"""


import os
import math

# disable tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import json

class TrackSegment:
    """ A short segment of a track. """
    def __init__(self):
        self.data = np.zeros((27, 64, 64, 5), dtype=np.float16)

    def append_thermal_frame(self, frame):
        """ Appends another thermal frame to the 27 frame buffer and shuffles other frames down. """

        self.data[0:-1] = self.data[1:]

        # clear out all 5 channels of the frame
        self.data[-1,:,:,:] = 0

        # copy across thermal part
        self.data[-1, :, :, 0] = frame

    def append_frame(self, frame):
        """
        Appends another 5 channel frame to the 27 frame buffer and shuffles other frames down.
        Channels are thermal, filtered, u, v, mask (where u,v are per pixel motion)
        :param frame: numpy array of shape [64, 64, 5]
        :return:
        """

        self.data[0:-1] = self.data[1:]

        # copy frames across
        self.data[-1, :, :, :] = frame

    def copy(self):
        """
        Deep copy of segment and it's data.
        :return: a deep copy if this segment.
        """
        new_segment = TrackSegment()
        new_segment.data = self.data[:, :, :, :]


class TrackClassifier:
    """ Classifies tracking segments. """
    def __init__(self, model_path, disable_GPU=True):
        """
        Create a new track classifier.
        :param model_path: the path to load the model from
        :param disable_GPU: defaults to on because it takes quite a lot of GPU memory and is not any faster for single
            segment classification
        """

        global tf
        import tensorflow as tf

        # TensorFlow session:

        # GPU is not as necessary if we are just classifying 1 segment at a time.
        if disable_GPU:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
            config = tf.ConfigProto(
                device_count={'GPU': 0},
                gpu_options=gpu_options
            )
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.8  # save some ram for other applications.

        self.sess = tf.Session(config=config)

        # predicton node
        self.prediction = None

        # list of classes
        self.classes = None

        # constants to use to normalise data.
        self.normalisation_constants = None

        self.load_model(model_path)


    def normalise(self, data):
        """
        Normalises a track segment according to pre-defined normalisation parameters.
        :param data: numpy array of shape (27, 64, 64, 5)
        :return: normalised segment data
        """

        # copy data, and make sure it's in a high enough precision to perform normalisation
        data = np.float32(data)

        for channel, (offset, scale, power) in enumerate(self.normalisation_constants):
            slice = data[:, :, :, channel]
            slice = slice + offset
            if power != 1:
                slice = np.power(np.abs(slice), power) * np.sign(slice)
            slice = slice / scale
            data[:, :, :, channel] = slice

        return data

    def load_model(self, model_path):
        """ Loads a pre-trained model from disk. """

        saver = tf.train.import_meta_graph(model_path+'.meta', clear_devices=True)
        saver.restore(self.sess, model_path)

        # get prediction node
        graph = tf.get_default_graph()
        self.prediction = graph.get_tensor_by_name('prediction:0')

        # get additonal stats
        stats = json.load(open(model_path+".txt",'r'))
        self.classes = stats['classes']

        self.normalisation_constants = stats['normalisation']

    def predict(self, segment):
        """
        Returns the models prediction of what animal is contained in this segment.
        :param segment: numpy array of shape [27,64,64,5]
        :return: probability distribution for each class.
        """
        return self.predict_batch([segment])[0]

    def predict_batch(self, segments):
        """
        Returns the models prediction of what animal is contained in each segment.
        :param segment: numpy array of shape [27,64,64,5]
        :return: probability distribution for each class.
        """

        result = []

        X = np.zeros((len(segments), 27, 64, 64, 5),dtype=np.float32)

        for i, segment in enumerate(segments):
            X[i, :, :, :, :] = self.normalise(segment.data)

        batches = int(math.ceil(len(X)/32))
        for i in range(batches):
            batch = X[i*32:(i+1)*32]

            feed_dict = {"X:0": batch}
            predictions = self.prediction.eval(feed_dict, session=self.sess)
            for j in range(len(predictions)):
                result.append(predictions[j])

        return result
