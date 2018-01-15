"""
Author: Matthew Aitchison
Date: Jan 2018
"""

import logging

import numpy as np
import tensorflow as tf

from ml_tools import tools
from ml_tools.model import Model


class ModelCRNN(Model):
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
        'thermal_threshold': 10,
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
        conv = tf.layers.conv2d(inputs=input_layer, filters=filters, kernel_size=kernal_size,
                                 strides=(conv_stride, conv_stride),
                                 padding="same", activation=None,
                                 name=name + '/conv')

        conv_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name + '/conv/kernel')[0]

        tf.summary.histogram(name + '/conv_output', conv)
        tf.summary.histogram(name + '/weights', conv_weights)

        activation = tf.nn.relu(conv, name=name + '/relu')
        tf.summary.histogram(name + '/activations', activation)

        if self.params['batch_norm'] and not disable_norm:
            out = tf.layers.batch_normalization(
                activation, fused=True,
                training=self.is_training,
                name=name + "/batchnorm"
            )

            moving_mean = tf.contrib.framework.get_variables(name + '/batchnorm/moving_mean')[0]
            moving_variance = tf.contrib.framework.get_variables(name + '/batchnorm/moving_variance')[0]

            tf.summary.histogram(name + '/batchnorm/mean', moving_mean)
            tf.summary.histogram(name + '/batchnorm/var', moving_variance)
            tf.summary.histogram(name + '/norm_output', out)
        else:
            out = activation

        if pool_stride != 1:
            out = tf.layers.max_pooling2d(inputs=out, pool_size=[pool_stride, pool_stride],
                                            strides=pool_stride,
                                            name=name + "/max_pool"
                                            )
        return out

    def create_summaries(self, name, var):
        """
        Creates TensorFlow summaries for given tensor
        :param name: the namespace for the summaries
        :param var: the tensor
        """
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def _build_model(self, label_count):
        ####################################
        # CNN + LSTM
        # based on https://arxiv.org/pdf/1507.06527.pdf
        ####################################
        
        # dimensions are documents as follows
        # B batch size
        # F frames per segment
        # C channels 
        # H frame height
        # W frame width

        # Setup placeholders
        self.X = tf.placeholder(tf.float32, [None, None, 5, 48, 48], name='X')          # [B, F, C, H, W]
        self.y = tf.placeholder(tf.int64, [None], name='y')
        frame_count = tf.shape(self.X)[1]
        batch_size = tf.shape(self.X)[0]

        # State input allows for processing longer sequences
        zero_state = tf.zeros(shape=[batch_size, 384, 2], dtype=tf.float32)
        self.state_in = tf.placeholder_with_default(input=zero_state, shape=[None, 384, 2], name='state_in')

        # Create some placeholder varaibles with defaults if not specified
        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0, tf.float32), [], name='keep_prob')
        self.is_training = tf.placeholder_with_default(tf.constant(False, tf.bool), [], name='training')
        self.global_step = tf.placeholder_with_default(tf.constant(0, tf.int32), [], name='global_step')

        # First put all frames in batch into one line sequence, this is required for convolutions.
        # note: we also switch to BHWC format, which is not great, but is required for CPU processing for some reason.
        X = self.X                              #[B, F, C, H, W]
        X = tf.transpose(X, (0, 1, 3, 4, 2))    #[B, F, H, W, C]
        X = tf.reshape(X, [-1, 48, 48, 5])      #[B*F, 48, 48, 5]

        # save distribution of inputs
        for channel in range(5):
            tf.summary.histogram('inputs/' + str(channel), X[:, :, :, channel])
            # just record the final frame, as that is often the most important.
            tf.summary.image('input/' + str(channel), X[-2:-1, :, :, channel:channel+1], max_outputs=1)

        layer = X[:, :, :, 0:0 + 1]
        layer = self.conv_layer('filtered/1', layer, 64, [3, 3], pool_stride=2)
        layer = self.conv_layer('filtered/2', layer, 64, [3, 3], pool_stride=2)
        layer = self.conv_layer('filtered/3', layer, 96, [3, 3], pool_stride=2)
        layer = self.conv_layer('filtered/4', layer, 128, [3, 3], pool_stride=2)
        layer = self.conv_layer('filtered/5', layer, 128, [3, 3], pool_stride=1)

        filtered_conv = layer

        layer = X[:, :, :, 2:3 + 1]
        layer = self.conv_layer('motion/1', layer, 64, [3, 3], pool_stride=2)
        layer = self.conv_layer('motion/2', layer, 64, [3, 3], pool_stride=2)
        layer = self.conv_layer('motion/3', layer, 96, [3, 3], pool_stride=2)
        layer = self.conv_layer('motion/4', layer, 128, [3, 3], pool_stride=2)
        layer = self.conv_layer('motion/5', layer, 128, [3, 3], pool_stride=1)

        motion_conv = layer

        logging.info("Convolution output shape: {} {}".format(filtered_conv.shape, motion_conv.shape))

        # reshape back into segments
        filtered_out = tf.reshape(filtered_conv, [-1, frame_count, tools.product(filtered_conv.shape[1:])], name='filtered/out')
        motion_out = tf.reshape(motion_conv, [-1, frame_count, tools.product(motion_conv.shape[1:])], name='motion/out')

        out = tf.concat((filtered_out, motion_out), axis=2, name='out')

        logging.info('Output shape {} from {}, {}'.format(out.shape, filtered_out.shape, motion_out.shape))

        # run the LSTM
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params['lstm_units'])

        dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob, dtype=np.float32)

        init_state = tf.nn.rnn_cell.LSTMStateTuple(self.state_in[:,:,0], self.state_in[:,:,1])

        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(
            cell=dropout, inputs=out,
            initial_state=init_state,
            swap_memory=True,
            dtype=tf.float32,
            scope='lstm'
        )

        lstm_state_1, lstm_state_2 = lstm_states

        # just need the last output
        lstm_output = lstm_outputs[:,-1]
        lstm_state = tf.stack([lstm_state_1, lstm_state_2], axis=2)

        logging.info("lstm output shape: {} x {}".format(lstm_outputs.shape[1], lstm_output.shape))
        logging.info("lstm state shape: {}".format(lstm_state.shape))

        # dense layer on top of convolutional output mapping to class labels.
        logits = tf.layers.dense(inputs=lstm_output, units=label_count, activation=None, name='logits')
        tf.summary.histogram('weights/logits', logits)

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
            # just relabel the loss node
            loss = tf.identity(softmax_loss, 'loss')

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
            learning_rate = self.params['learning_rate']

        # 1e-6 because our data is a bit non normal.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, name='train_op')
            # get gradients
            # note: I can't write out the grads because of problems with NaN
            # his is very concerning as it implies we have a critical problem with training.  Maybe I should try
            # clipping gradients at something very high, say 100?
            #grads = optimizer.compute_gradients(loss)
            #for index, grad in enumerate(grads):
            #    self.create_summaries("grads/{}".format(grads[index][1].name.split(':')[0]), grads[index])

        # attach nodes
        self.set_ops(pred=pred, accuracy=accuracy, loss=loss, train_op=train_op)
        self.state_out = tf.identity(lstm_state, 'state_out')
