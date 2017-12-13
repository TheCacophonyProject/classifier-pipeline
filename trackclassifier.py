"""
Module classify a tracking window based on a 3 second segment.
"""

import os

# disable logging
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


class TrackClassifier:
    """ Classifies tracking segments. """
    def __init__(self, model_path):

        global tf
        import tensorflow as tf

        # TensorFlow session:
        # note we disable the GPU, it won't be needed as classification is very quick anyway.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
        config = tf.ConfigProto(
            device_count={'GPU': 0},
            gpu_options=gpu_options
        )

        self.sess = tf.Session(config=config)

        # predicton node
        self.prediction = None

        # list of classes
        self.classes = None


        self.load_model(model_path)



    def load_model(self, model_path):
        """ Loads a pre-trained model from disk. """

        saver = tf.train.import_meta_graph(model_path+'.meta', clear_devices=True)
        saver.restore(self.sess, model_path)

        # get prediction node
        self.prediction = tf.get_collection('predict_op')[0]

        # get addtional stats
        stats = json.load(open(model_path+".txt",'r'))
        self.classes = stats['classes']

    def predict(self, segment):
        """
        Returns the models prediction of what animal is contained in this segment.
        :param segment: the segment to classify
        :return: probability distribution for each class.
        """

        feed_dict = {"X:0": segment.data[np.newaxis,:,:,:,:]}
        result = self.prediction.eval(feed_dict, session=self.sess)[0]
        return result

