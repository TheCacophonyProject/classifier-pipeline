"""
Author: Matthew Aitchison
Date: Jan 2018
"""

import tensorflow as tf
import numpy as np

import logging

from ml_tools.model import Model
from ml_tools import tools


class Model_CRNN(Model):
    """
    Convolutional neural net model feeding into an LSTM
    """

    MODEL_NAME = "DEEP"
    MODEL_DESCRIPTION = "CNN + LSTM"

    DEFAULT_PARAMS = {

        # training params
        'batch_size': 32,
        'learning_rate': 1e-4,
        'learning_rate_decay': 1.0,
        'l2_reg': 0,
        'label_smoothing': 0.1,
        'keep_prob': 0.5,

        # model params
        'batch_norm': True,
        'lstm_units': 384,

        # augmentation
        'augmentation': True,
        'filter_threshold': 20,
        'filter_noise': 1.0,
        'scale_frequency': 0.5
    }

    def __init__(self, labels):
        """
        Initialise the model
        :param labels: number of labels for model to predict
        """
        super().__init__()
        self.params.update(self.DEFAULT_PARAMS)
        self._build_model(labels)

    def conv_layer(self, name, input_layer, filters, kernal_size, conv_stride=1, pool_stride=1, disable_norm=False):
        """ Adds a convolutional layer to the model. """
        tf.summary.histogram(name + '/input', input_layer)
        layer = tf.layers.conv2d(inputs=input_layer, filters=filters, kernel_size=kernal_size,
                                 strides=(conv_stride, conv_stride),
                                 padding="same", activation=None,
                                 name=name + '/conv')

        tf.summary.histogram(name + '/conv_output', layer)

        layer = tf.nn.relu(layer, name=name + '/relu')

        tf.summary.histogram(name + '/activations', layer)

        if self.params['batch_norm'] and not disable_norm:
            layer = tf.layers.batch_normalization(
                layer, fused=True,
                training=self.is_training,
                name=name + "/batchnorm"

            )
            moving_mean = tf.contrib.framework.get_variables(name + '/batchnorm/moving_mean')[0]
            moving_variance = tf.contrib.framework.get_variables(name + '/batchnorm/moving_variance')[0]

            tf.summary.histogram(name + '/batchnorm/mean', moving_mean)
            tf.summary.histogram(name + '/batchnorm/var', moving_variance)

        if pool_stride != 1:
            layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[pool_stride, pool_stride],
                                            strides=pool_stride,
                                            name=name + "/max_pool"
                                            )
        return layer

    def _build_model(self, label_count):
        ####################################
        # CNN + LSTM
        # based on https://arxiv.org/pdf/1507.06527.pdf
        ####################################

        # Setup placeholders
        self.X = tf.placeholder(tf.float32, [None, 27, 5, 48, 48], name='X')
        self.y = tf.placeholder(tf.int64, [None], name='y')

        # Create some placeholder varaibles with defaults if not specified
        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0, tf.float32), [], name='keep_prob')
        self.is_training = tf.placeholder_with_default(tf.constant(False, tf.bool), [], name='training')
        self.global_step = tf.placeholder_with_default(tf.constant(0, tf.int32), [], name='global_step')

        # First put all frames in batch into one line sequence, this is required for convolutions.
        # note: we also switch to BHWC format, which is not great, but is required for CPU processing for some reason.
        X = tf.transpose(self.X,(0, 1, 3, 4, 2))
        X = tf.reshape(self.X, [-1, 48, 48, 5])

        # save distribution of inputs
        for channel in range(5):
            tf.summary.histogram('inputs/'+str(channel), X[:, :, :, channel])

        layer = self.conv_layer('filtered/1',X[:, :, :, 1:1+1], 48, [3, 3], pool_stride=2, disable_norm=True)
        layer = self.conv_layer('filtered/2',layer, 64, [3, 3], pool_stride=2)
        layer = self.conv_layer('filtered/3',layer, 96, [3, 3], pool_stride=2)
        layer = self.conv_layer('filtered/4',layer, 128, [3, 3], pool_stride=2)
        layer = self.conv_layer('filtered/5',layer, 128, [3, 3], pool_stride=1)

        filtered_conv = layer

        layer = self.conv_layer('motion/1',X[:, :, :, 2:3+1], 48, [3, 3], pool_stride=2, disable_norm=True)
        layer = self.conv_layer('motion/2',layer, 64, [3, 3], pool_stride=2)
        layer = self.conv_layer('motion/3',layer, 96, [3, 3], pool_stride=2)
        layer = self.conv_layer('motion/4',layer, 128, [3, 3], pool_stride=2)
        layer = self.conv_layer('motion/5',layer, 128, [3, 3], pool_stride=1)

        motion_conv = layer

        logging.info("Convolution output shape: {} {}".format(filtered_conv.shape, motion_conv.shape))

        # reshape back into segments

        filtered_out = tf.reshape(filtered_conv, [-1, 27, tools.product(filtered_conv.shape[1:])], name='filtered/out')
        motion_out = tf.reshape(motion_conv, [-1, 27, tools.product(motion_conv.shape[1:])], name='motion/out')

        out = tf.concat((filtered_out, motion_out), axis=2, name='out')

        logging.info('Output shape {} from {}, {}'.format(out.shape, filtered_out.shape, motion_out.shape))

        # the LSTM expects an array of 27 examples
        sequences = tf.unstack(out, 27, 1)

        logging.info('Sequences'.format(len(sequences), sequences[0].shape))

        # run the LSTM
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params['lstm_units'])

        dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob, dtype=np.float32)

        lstm_outputs, lstm_state = tf.nn.static_rnn(
            cell=dropout, inputs=sequences, sequence_length=[len(sequences)]*self.params['batch_size'],
            dtype=tf.float32, scope='lstm')

        # just need the last output
        lstm_output = lstm_outputs[-1]

        # print("Final output shape:",lstm_output.shape)
        logging.info("lstm output shape: {} x {}".format(len(lstm_outputs),lstm_output.shape))

        # dense layer on top of convolutional output mapping to class labels.
        logits = tf.layers.dense(inputs=lstm_output, units=label_count, activation=None, name='logits')


        # loss

        softmax_loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=tf.one_hot(self.y, label_count),
                    logits=logits, label_smoothing=self.params['label_smoothing'],
                    scope='loss')

        if self.params['l2_reg'] != 0:
            with tf.variable_scope('logits', reuse=True):
                logit_weights = tf.get_variable('kernel')

            reg_loss = (tf.nn.l2_loss(logit_weights, name='loss/reg') * self.params['l2_reg'])
            loss = tf.add(
                softmax_loss, reg_loss, name='loss'
            )
        else:
            loss = softmax_loss

        class_out = tf.argmax(logits, axis=1, name='class_out')
        correct_prediction = tf.equal(class_out, self.y)
        pred = tf.nn.softmax(logits, name='prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name='accuracy')

        # setup our training loss
        if self.params['learning_rate_decay'] != 1.0:
            learning_rate = tf.train.exponential_decay(self.params['learning_rate'], self.global_step, 1000,
                                                   self.params['learning_rate_decay'],
                                                   staircase=True)
            tf.summary.scalar('params/learning_rate', learning_rate)
        else:
            learning_rate = 1.0

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, name='train_op')

        # attach nodes
        self.set_ops(pred=pred,accuracy=accuracy, loss=loss, train_op=train_op)


