"""
Script to train model for classifying animals from thermal footage.
"""

import os.path
import numpy as np
import random
import pickle
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import itertools
from importlib import reload
from ml_tools import tools
from tensorflow import contrib
import json
from ml_tools.model import Model

import logging

"""
Things to try
Apparently recomendedation is 1e-8?? might try that
Gradient clipping on LSTM is a good idea, set to 10
Deeper
Batch norm after every weight layer?
Try 10 frames?
Do some performance analysis... could optimise this
"""

#------------------------------------------------------------
# Helper functions
#------------------------------------------------------------

def prod(data):
    x = 1
    for value in data:
        x *= int(value)
    return x

#------------------------------------------------------------

class Estimator():

    # todo: these should be in some settings file
    MODEL_NAME = "DEEP"
    MODEL_DESCRIPTION = "CNN + LSTM"

    MAX_EPOCHS = 5

    BATCH_SIZE = 16
    BATCH_NORM = True
    LEARNING_RATE = 1e-4
    LEARNING_RATE_DECAY = 1.0
    L2_REG = 0.01
    LABEL_SMOOTHING = 0.1
    LSTM_UNITS = 128
    USE_PEEPHOLES = False # these don't really help.
    AUGMENTATION = True
    NOTES = "#search-lr-1e5#"

    def get_hyper_parameter_string(self):
        """ Converts hyperparmeters into a string. """
        return "{}_epoch={}_bs={}_bn={}_lr={}_lrd={}_l2reg={}_ls={}_h={}_aug={}".format(
            self.NOTES, self.MAX_EPOCHS, self.BATCH_SIZE, self.BATCH_NORM, self.LEARNING_RATE, self.LEARNING_RATE_DECAY,
            self.L2_REG, self.LABEL_SMOOTHING, self.LSTM_UNITS, self.AUGMENTATION
        )

    def __init__(self):

        self.train = None
        self.validation = None
        self.test = None

        # tensorflow placeholders
        self.keep_prob = None

    @property
    def datasets(self):
        return (self.train, self.validation, self.test)

    @property
    def labels(self):
        return self.train.labels

    def import_dataset(self, base_path, force_normalisation_constants=None):
        """
        Import dataset from basepath.
        :param base_path:
        :param force_normalisation_constants: If defined uses these normalisation constants rather than those
            saved with the dataset.
        :return:
        """
        self.train, self.validation, self.test = pickle.load(open(os.path.join(base_path, "datasets.dat"),'rb'))

        # augmentation really helps with reducing over-fitting, but test set should be fixed so we don't apply it there.
        self.train.enable_augmentation = self.AUGMENTATION
        self.validation.enable_augmentation = False
        self.test.enable_augmentation = False

        logging.info("Training segments: {0:.1f}k".format(self.train.rows/1000))
        logging.info("Validation segments: {0:.1f}k".format(self.validation.rows/1000))
        logging.info("Test segments: {0:.1f}k".format(self.test.rows/1000))
        logging.info("Labels: {}".format(self.train.labels))

        label_strings = [",".join(self.train.labels), ",".join(self.test.labels), ",".join(self.validation.labels)]
        assert len(set(label_strings)) == 1, 'dataset labels do not match.'

        if force_normalisation_constants:
            print("Using custom normalisation constants.")
            for dataset in self.datasets:
                dataset.normalisation_constants = force_normalisation_constants

        self.test.load_all()

    def _conv_layer(self, name, input_layer, filters, kernal_size, conv_stride=1, pool_stride=1):

        layer = tf.layers.conv2d(inputs=input_layer, filters=filters, kernel_size=kernal_size,
                                 strides=(conv_stride, conv_stride),
                                 padding="same", activation=None, data_format='channels_first',
                                 name=name+"_conv")

        tf.summary.histogram(name+'/layeractivations', layer)

        layer = tf.nn.relu(layer)

        if self.BATCH_NORM:
            layer = tf.layers.batch_normalization(
                layer, axis=1, fused=True,
                training=self.training,
                #renorm=True,
                name=name + "/batchnorm"

            )
            moving_mean = tf.contrib.framework.get_variables(name + '/batchnorm/moving_mean')[0]
            moving_variance = tf.contrib.framework.get_variables(name + '/batchnorm/moving_variance')[0]

            tf.summary.histogram(name+'/batchnorm/mean', moving_mean)
            tf.summary.histogram(name+'/batchnorm/var', moving_variance)


        if pool_stride != 1:
            layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[pool_stride, pool_stride],
                                            strides=pool_stride, data_format='channels_first',
                                            name=name + "_max_pool"
                                            )
        return layer

    def build_model(self):
        ####################################
        # CNN + LSTM
        # based on https://arxiv.org/pdf/1507.06527.pdf
        ####################################

        tf.reset_default_graph()

        # Define our model

        self.X = tf.placeholder(tf.float32, [None, 27, 5, 48, 48], name='X')

        self.y = tf.placeholder(tf.int64, [None], name='y')

        # Split up input

        # default keep_probability to 1.0 if not specified
        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0, tf.float32), [], name='keep_prob')
        self.training = tf.placeholder_with_default(tf.constant(False, tf.bool), [], name='training')

        # first put all frames in batch into one line sequence
        X_reshaped = tf.reshape(self.X, [-1, 5, 48, 48])

        # save distribution of inputs
        for channel in range(5):
            tf.summary.histogram('inputs/'+str(channel), X_reshaped[:, channel])

        layer = self._conv_layer('filtered/1',X_reshaped[:, 2:4, :, :], 64, [3, 3], pool_stride=2)
        layer = self._conv_layer('filtered/2',layer, 64, [3, 3], pool_stride=2)
        layer = self._conv_layer('filtered/3',layer, 96, [3, 3], pool_stride=2)
        layer = self._conv_layer('filtered/4',layer, 128, [3, 3], pool_stride=2)
        layer = self._conv_layer('filtered/5',layer, 128, [3, 3], pool_stride=1)

        filtered_conv = layer

        """
        layer = self._conv_layer('motion/1',X_reshaped[:, 2:4, :, :], 64, [3, 3], pool_stride=2)
        layer = self._conv_layer('motion/2',layer, 64, [3, 3], pool_stride=2)
        layer = self._conv_layer('motion/3',layer, 96, [3, 3], pool_stride=2)
        layer = self._conv_layer('motion/4',layer, 128, [3, 3], pool_stride=2)
        layer = self._conv_layer('motion/5',layer, 128, [3, 3], pool_stride=1)

        motion_conv = layer
        """

        """
        c1 = self._conv_layer(X_reshaped[:, :, :, 1:2], 32, [8, 8], pool_stride=4)
        c2 = self._conv_layer(c1, 48, [4, 4], pool_stride=2)
        c3 = self._conv_layer(c2, 64, [3, 3], pool_stride=1)

        filtered_conv = c3

        c1 = self._conv_layer(X_reshaped[:, :, :, 2:4], 32, [8, 8], pool_stride=4)
        c2 = self._conv_layer(c1, 48, [4, 4], pool_stride=2)
        c3 = self._conv_layer(c2, 64, [3, 3], pool_stride=1)
        

        motion_conv = c3
        
        """

        #print("convolution output shape: ", filtered_conv.shape, motion_conv.shape)

        # reshape back into segments

        flat1 = tf.reshape(filtered_conv, [-1, 27, prod(filtered_conv.shape[1:])])
        #flat2 = tf.reshape(motion_conv, [-1, 27, prod(motion_conv.shape[1:])])

        # no motion
        flat = tf.concat((flat1,), axis=2)

        #print('Flat', flat.shape, 'from', flat1.shape, ',', flat2.shape)

        # the LSTM expects an array of 27 examples
        sequences = tf.unstack(flat, 27, 1)

        print('Sequences', len(sequences), sequences[0].shape)

        # run the LSTM
        lstm_cell_fw = tf.contrib.rnn.LSTMCell(
            num_units=self.LSTM_UNITS, use_peepholes=self.USE_PEEPHOLES, )
        lstm_cell_bk = tf.contrib.rnn.LSTMCell(
            num_units=self.LSTM_UNITS, use_peepholes=self.USE_PEEPHOLES)

        # lstm_outputs, lstm_states = tf.contrib.rnn.static_rnn(lstm_cell, sequences, dtype = 'float32')
        lstm_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
            lstm_cell_fw, lstm_cell_bk, sequences,
            dtype=tf.float32)

        print("LSTM outputs:", len(lstm_outputs))

        # just need the last output
        lstm_output = lstm_outputs[-1]

        # print("Final output shape:",lstm_output.shape)
        print("lstm output shape:", lstm_output.shape)

        # skip dense layer... might be needed for more complex things, but need more data to train.
        h1 = tf.nn.dropout(lstm_output, keep_prob=self.keep_prob)

        # dense layer2
        logits = tf.layers.dense(inputs=h1, units=len(self.labels), activation=None, name='logits')

        # prediction with softmax
        class_out = tf.argmax(logits, axis=1, name='class_out')
        pred = tf.nn.softmax(logits, name='prediction')

        correct_prediction = tf.equal(class_out, self.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name='accuracy')

        with tf.variable_scope('logits', reuse=True):
            h2_weights = tf.get_variable('kernel')

        reg_loss = (tf.nn.l2_loss(h2_weights) * self.L2_REG)
        loss = tf.add(
            tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(self.y, len(self.labels)), logits=logits,
                                            label_smoothing=self.LABEL_SMOOTHING), reg_loss,
            name='loss')

        # setup our training loss
        epoch_steps = self.train.rows
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.LEARNING_RATE, global_step, epoch_steps, self.LEARNING_RATE_DECAY,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, name='train_op')

        """
        # clip grads
        grad_vars = optimizer.compute_gradients(loss=loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        capped_grads = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in grad_vars]
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(capped_grads, name='train_op')
            
        # show grads
        for grad, var in grad_vars:
            tf.summary.histogram(var.name + '/gradient', grad)
        for grad, var in capped_grads:
            tf.summary.histogram(var.name + '/clipped_gradient', grad)
        """

        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), name='train_op')

        # show grads
        for grad, clipped, var in zip(gradients, clipped_gradients, variables):
            tf.summary.histogram(var.name + '/gradient', grad)
            tf.summary.histogram(var.name + '/clipped_gradient', clipped)
        """

        # define our model
        self.model = Model(self.datasets, self.X, self.y, self.keep_prob, pred, accuracy, loss, train_op, self.labels)
        self.model.batch_size = self.BATCH_SIZE
        self.model.name = self.MODEL_NAME + "_" + self.get_hyper_parameter_string()
        self.model.training = self.training

    def start_async_load(self):
        self.train.start_async_load(256)
        self.validation.start_async_load(256)

    def stop_async(self):
        self.train.stop_async_load()
        self.validation.stop_async_load()

    def train_model(self, max_epochs=10.0, stop_after_no_improvement=None, stop_after_decline=None, log_dir = None):
        print("{0:.1f}K training examples".format(self.train.rows / 1000))
        self.model.train_model(max_epochs, keep_prob=0.4, stop_after_no_improvement=stop_after_no_improvement,
                               stop_after_decline=stop_after_decline, log_dir=log_dir)

    def save_model(self):
        """ Saves a copy of the current model. """
        score_part = "{:.3f}".format(self.model.eval_score)
        while len(score_part) < 3:
            score_part = score_part + "0"

        saver = tf.train.Saver()
        save_filename = os.path.join("./models/", self.MODEL_NAME + '-' + score_part)
        print("Saving", save_filename)
        saver.save(self.model.sess, save_filename)

        # save some additional data
        model_stats = {}
        model_stats['name'] = self.MODEL_NAME
        model_stats['description'] = self.MODEL_DESCRIPTION
        model_stats['notes'] = ""
        model_stats['classes'] = self.labels
        model_stats['score'] = self.model.eval_score
        model_stats['normalisation'] = self.train.normalisation_constants

        json.dump(model_stats, open(save_filename + ".txt", 'w'), indent=4)


def main():

    logging.basicConfig(level=0)
    tf.logging.set_verbosity(3)

    normalisation_constants = [
        [3200, 200],
        [5, 20],
        [0, 1],
        [0, 1],
        [0, 1]
    ]

    # try learning rates for 10 epochs
    #learning_rates = [
    #    1e-2,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6
    #]

    learning_rates = [1e-4]

    log_folder = 'c:/cac/search_lr/'

    for learning_rate in learning_rates:

        print("-"*60)
        print("STARTING JOB learning_rate="+str(learning_rate))
        print("-" * 60)

        sess = tools.get_session()

        estimator = Estimator()
        estimator.import_dataset("c://cac//kea", force_normalisation_constants=normalisation_constants)

        estimator.MODEL_NAME = "flow/squareroot more jitter no flow scale/"
        estimator.LEARNING_RATE = learning_rate
        estimator.LEARNING_RATE_DECAY = 1.0
        estimator.NOTES = str(learning_rate)

        estimator.MAX_EPOCHS = 10

        estimator.build_model()

        try:
            estimator.start_async_load()
            estimator.train_model(
                max_epochs=estimator.MAX_EPOCHS, stop_after_no_improvement=None, stop_after_decline=None,
                log_dir=log_folder)
            estimator.save_model()
        finally:
            estimator.stop_async()
            estimator.model.sess.close()
            sess.close()



if __name__ == "__main__":
    # execute only if run as a script
    main()