"""
Author: Matthew Aitchison
Date: Jan 2018
"""

import logging

import numpy as np
import tensorflow as tf

from ml_tools import tools
from ml_tools.model import Model


class ConvModel(Model):
    """
    Base class for convolutional models.
    """

    def conv_layer(
        self,
        name,
        input_layer,
        filters,
        kernal_size,
        conv_stride=1,
        pool_stride=1,
        disable_norm=False,
    ):
        """ Adds a convolutional layer to the model. """

        tf.compat.v1.summary.histogram(name + "/input", input_layer)
        conv = tf.compat.v1.layers.conv2d(
            inputs=input_layer,
            filters=filters,
            kernel_size=kernal_size,
            strides=(conv_stride, conv_stride),
            padding="same",
            activation=None,
            name=name + "/conv",
        )

        conv_weights = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, name + "/conv/kernel"
        )[0]

        tf.compat.v1.summary.histogram(name + "/conv_output", conv)
        tf.compat.v1.summary.histogram(name + "/weights", conv_weights)

        activation = tf.nn.relu(conv, name=name + "/relu")
        tf.compat.v1.summary.histogram(name + "/activations", activation)

        if self.params["batch_norm"] and not disable_norm:
            out = tf.compat.v1.layers.batch_normalization(
                activation,
                fused=True,
                training=self.is_training,
                name=name + "/batchnorm",
            )
            moving_mean = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                scope=name + "/batchnorm/moving_mean",
            )[0]

            moving_variance = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                scope=name + "/batchnorm/moving_variance",
            )[0]

            tf.compat.v1.summary.histogram(name + "/batchnorm/mean", moving_mean)
            tf.compat.v1.summary.histogram(name + "/batchnorm/var", moving_variance)
            tf.compat.v1.summary.histogram(name + "/norm_output", out)
        else:
            out = activation

        if pool_stride != 1:
            out = tf.compat.v1.layers.max_pooling2d(
                inputs=out,
                pool_size=[pool_stride, pool_stride],
                strides=pool_stride,
                name=name + "/max_pool",
            )
        return out

    def process_inputs(self):
        """ process input channels, returns thermal, flow, mask, """

        # Setup placeholders
        self.X = tf.compat.v1.placeholder(
            tf.float32, [None, 5, 48, 48], name="X"
        )  # [B, F, C, H, W]
        self.y = tf.compat.v1.placeholder(tf.int64, [None], name="y")
        batch_size = tf.shape(input=self.X)[0]

        # State input allows for processing longer sequences
        zero_state = tf.zeros(
            shape=[batch_size, self.params["lstm_units"], 2], dtype=tf.float32
        )
        self.state_in = tf.compat.v1.placeholder_with_default(
            input=zero_state,
            shape=[None, self.params["lstm_units"], 2],
            name="state_in",
        )

        # Create some placeholder variables with defaults if not specified
        self.keep_prob = tf.compat.v1.placeholder_with_default(
            tf.constant(1.0, tf.float32), [], name="keep_prob"
        )
        self.is_training = tf.compat.v1.placeholder_with_default(
            tf.constant(False, tf.bool), [], name="training"
        )
        self.global_step = tf.compat.v1.placeholder_with_default(
            tf.constant(0, tf.int32), [], name="global_step"
        )

        # Apply pre-processing
        X = self.X  # [B, F, C, H, W]

        # normalise the thermal
        # the idea here is to apply sqrt to any values over 100 so that we reduce the effect of very strong values.
        thermal = X[ :, 0 : 0 + 1]
        
        #         (20,27, 5, 48, 48)
        # thermal = X[:, :, 0 : 0 + 1]
# (?, 48, 48, 1)

# (?, 5, 48, 48)
# (?, 1, 48, 48)

        AUTO_NORM_THERMAL = False
        THERMAL_ROLLOFF = 400

        if AUTO_NORM_THERMAL:
            thermal = thermal - tf.reduce_mean(
                input_tensor=thermal, axis=(3, 4), keepdims=True
            )  # center data

            signs = tf.sign(thermal)
            abs = tf.abs(thermal)
            thermal = (
                tf.minimum(tf.sqrt(abs / THERMAL_ROLLOFF) * THERMAL_ROLLOFF, abs)
                * signs
            )  # curve off the really strong values
            thermal = thermal - tf.reduce_mean(
                input_tensor=thermal, axis=(3, 4), keepdims=True
            )  # center data
            thermal = thermal / tf.sqrt(
                tf.reduce_mean(
                    input_tensor=tf.square(thermal), axis=(3, 4), keepdims=True
                )
            )
            # theshold out the background, not sure this is a great idea.  1.5 keeps std approx 1.
            # relu_threshold = +0.1
            # thermal = (tf.nn.relu(thermal - relu_threshold) + relu_threshold) * 1.5
        else:
            # signs = tf.sign(thermal)
            abs = tf.abs(thermal)
            thermal = (
                tf.minimum(tf.sqrt(abs / THERMAL_ROLLOFF) * THERMAL_ROLLOFF, abs)
                # * signs
            )  # curve off the really strong values

            thermal = (
                tf.nn.relu(thermal - self.params["thermal_threshold"])
                + self.params["thermal_threshold"]
            )

            thermal = thermal / 40

        # normalise the flow
        # horizontal and vertical flow have different normalisation constants
        flow = X[ :, 2 : 3 + 1]
        flow = (
            flow
            * np.asarray([2.5, 5])[np.newaxis, :, np.newaxis, np.newaxis]
        )

        # grab the mask
        mask = X[ :, 4 : 4 + 1]

        # tap the outputs
        tf.identity(thermal, "thermal_out")
        tf.identity(flow, "flow_out")
        tf.identity(mask, "mask_out")

        # First put all frames in batch into one line sequence, this is required for convolutions.
        # note: we also switch to BHWC format, which is not great, but is required for CPU processing for some reason.
        thermal = tf.transpose(a=thermal, perm=(0, 2, 3, 1))  # [B, F, H, W, 1]
        flow = tf.transpose(a=flow, perm=( 0, 2, 3, 1))  # [B, F, H, W, 2]

        thermal = tf.reshape(thermal, [-1, 48, 48, 1])  # [B*F, 48, 48, 1]
        flow = tf.reshape(flow, [-1, 48, 48, 2])  # [B*F, 48, 48, 2]

        mask = tf.reshape(mask, [-1, 48, 48, 1])  # [B*F, 48, 48, 1]

        # save distribution of inputs
        self.save_input_summary(thermal, "inputs/thermal", 3)
        self.save_input_summary(flow[:, :, :, 0 : 0 + 1], "inputs/flow/h", 3)
        self.save_input_summary(flow[:, :, :, 1 : 1 + 1], "inputs/flow/v", 3)
        self.save_input_summary(mask, "inputs/mask", 1)
        return thermal, flow, mask

    def setup_novelty(self, logits, hidden):
        """ Creates nodes required for novelty"""

        # samples is [1000, C]
        # logits is [N, C]
        # delta is [N, 1000, C]
        # distances is [N, 1000]

        _, label_count = logits.shape
        _, hidden_count = hidden.shape

        sample_logits = self.create_writable_variable(
            "sample_logits", [1000, label_count]
        )
        self.create_writable_variable("sample_hidden", [1000, hidden_count])

        novelty_threshold = self.create_writable_variable("novelty_threshold", [])
        novelty_scale = self.create_writable_variable("novelty_scale", [])

        delta = tf.expand_dims(logits, axis=1) - tf.expand_dims(sample_logits, axis=0)
        squared_distances = tf.reduce_sum(input_tensor=tf.square(delta), axis=2)
        min_distance = tf.sqrt(
            tf.reduce_min(input_tensor=squared_distances, axis=1),
            name="novelty_distance",
        )
        novelty = tf.sigmoid(
            (min_distance - novelty_threshold) / novelty_scale, "novelty"
        )

        return novelty

    def setup_optimizer(self, loss):
        # setup our training loss
        if self.params["learning_rate_decay"] != 1.0:
            learning_rate = tf.compat.v1.train.exponential_decay(
                self.params["learning_rate"],
                self.global_step,
                1000,
                self.params["learning_rate_decay"],
                staircase=True,
            )
            tf.compat.v1.summary.scalar("params/learning_rate", learning_rate)
        else:
            learning_rate = self.params["learning_rate"]

        # setup optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate, name="AdamO"
        )
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, name="train_op")
            # get gradients
            # note: I can't write out the grads because of problems with NaN
            # his is very concerning as it implies we have a critical problem with training.  Maybe I should try
            # clipping gradients at something very high, say 100?
            # grads = optimizer.compute_gradients(loss)
            # for index, grad in enumerate(grads):
            #    self.create_summaries("grads/{}".format(grads[index][1].name.split(':')[0]), grads[index])


class ModelCRNN_HQ(ConvModel):
    """
    Convolutional neural net model feeding into an LSTM
    """

    MODEL_NAME = "model_hq"
    MODEL_DESCRIPTION = "CNN + LSTM"

    DEFAULT_PARAMS = {
        # training params
        "batch_size": 16,
        "learning_rate": 1e-4,
        "learning_rate_decay": 1.0,
        "l2_reg": 0.0,
        "label_smoothing": 0.1,
        "keep_prob": 0.5,
        # model params
        "batch_norm": True,
        "lstm_units": 512,
        "enable_flow": True,
        # augmentation
        "augmentation": True,
        "thermal_threshold": 10,
        "scale_frequency": 0.5,
    }

    def __init__(self, labels, train_config, **kwargs):
        """
        Initialise the model
        :param labels: number of labels for model to predict
        """
        super().__init__(train_config=train_config)
        self.params.update(self.DEFAULT_PARAMS)
        self.params.update(kwargs)
        self._build_model(labels)

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

        thermal, flow, mask = self.process_inputs()

        frame_count = tf.shape(input=self.X)[1]

        # -------------------------------------
        # run the Convolutions

        layer = thermal

        layer = self.conv_layer("thermal/1", layer, 64, [3, 3], pool_stride=2)
        layer = self.conv_layer("thermal/2", layer, 64, [3, 3], pool_stride=2)
        layer = self.conv_layer("thermal/3", layer, 96, [3, 3], pool_stride=2)
        layer = self.conv_layer("thermal/4", layer, 128, [3, 3], pool_stride=2)
        layer = self.conv_layer("thermal/5", layer, 128, [3, 3], pool_stride=1)

        filtered_conv = layer
        filtered_out = tf.reshape(
            filtered_conv,
            [-1, frame_count, tools.product(filtered_conv.shape[1:])],
            name="thermal/out",
        )
        logging.info("Thermal convolution output shape: {}".format(filtered_conv.shape))
        logging.info("filtered_out convolution output shape: {}".format(filtered_out.shape))

        if self.params["enable_flow"]:
            # integrate thermal and flow into a 3 channel layer
            layer = tf.concat((thermal, flow), axis=3)
            layer = self.conv_layer("motion/1", layer, 64, [3, 3], pool_stride=2)
            layer = self.conv_layer("motion/2", layer, 64, [3, 3], pool_stride=2)
            layer = self.conv_layer("motion/3", layer, 96, [3, 3], pool_stride=2)
            layer = self.conv_layer("motion/4", layer, 128, [3, 3], pool_stride=2)
            layer = self.conv_layer("motion/5", layer, 128, [3, 3], pool_stride=1)

            motion_conv = layer
            motion_out = tf.reshape(
                motion_conv,
                [-1, frame_count, tools.product(motion_conv.shape[1:])],
                name="motion/out",
            )
            logging.info(
                "Motion convolution output shape: {}".format(motion_conv.shape)
            )

            out = tf.concat((filtered_out, motion_out), axis=2, name="out")
        else:
            out = tf.concat((filtered_out,), axis=2, name="out")

        logging.info("Output shape {}".format(out.shape))

        # -------------------------------------
        # run the LSTM
        memory_output, memory_state = self._build_memory(out)
        if self.params["l2_reg"] > 0:
            regularizer = tf.keras.regularizers.l2(l=0.5 * (self.params["l2_reg"]))
        else:
            regularizer = None

        # dense hidden layer
        dense = tf.compat.v1.layers.dense(
            inputs=memory_output,
            units=384,
            activation=tf.nn.relu,
            name="hidden",
            kernel_regularizer=regularizer,
        )

        dense = tf.nn.dropout(dense, rate=1 - (self.keep_prob))
        # dense layer on top of convolutional output mapping to class labels.
        logits = tf.compat.v1.layers.dense(
            inputs=dense,
            units=label_count,
            activation=None,
            name="logits",
            kernel_regularizer=regularizer,
        )

        tf.compat.v1.summary.histogram("weights/dense", dense)
        tf.compat.v1.summary.histogram("weights/logits", logits)

        # loss
        softmax_loss = tf.compat.v1.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(self.y, label_count),
            logits=logits,
            label_smoothing=self.params["label_smoothing"],
            scope="softmax_loss",
        )

        if self.params["l2_reg"] != 0:
            reg_loss = tf.compat.v1.losses.get_regularization_loss()
            loss = tf.add(softmax_loss, reg_loss, name="loss")
            tf.compat.v1.summary.scalar("loss/reg", reg_loss)
            tf.compat.v1.summary.scalar("loss/softmax", softmax_loss)
        else:
            # just relabel the loss node
            loss = tf.identity(softmax_loss, name="loss")

        class_out = tf.argmax(input=logits, axis=1, name="class_out")
        correct_prediction = tf.equal(class_out, self.y)
        pred = tf.nn.softmax(logits, name="prediction")
        accuracy = tf.reduce_mean(
            input_tensor=tf.cast(correct_prediction, dtype=tf.float32), name="accuracy"
        )

        # -------------------------------------
        # novelty

        self.setup_novelty(logits, memory_output)
        self.setup_optimizer(loss)

        # make reference to special nodes
        tf.identity(memory_state, "state_out")
        tf.identity(dense, "hidden_out")
        tf.identity(logits, "logits_out")

        self.attach_nodes()


class ModelCRNN_LQ(ConvModel):
    """
    Convolutional neural net model feeding into an LSTM

    Lower quality, but faster model
    Uses 256 LSTM units and conv stride instead of max pool
    Uses less filters

    Trains on GPU at around 5ms / segment as apposed to 16ms for the high quality model.
    """

    MODEL_NAME = "model_lq"
    MODEL_DESCRIPTION = "CNN + LSTM"

    DEFAULT_PARAMS = {
        # training params
        "batch_size": 16,
        "learning_rate": 1e-4,
        "learning_rate_decay": 1.0,
        "l2_reg": 0,
        "label_smoothing": 0.1,
        "keep_prob": 0.2,
        # model params
        "batch_norm": True,
        "lstm_units": 256,
        "enable_flow": True,
        # augmentation
        "augmentation": True,
        "thermal_threshold": 10,
        "scale_frequency": 0.5,
    }

    def __init__(self, labels, train_config, **kwargs):
        """
        Initialise the model
        :param labels: number of labels for model to predict
        """
        super().__init__(train_config=train_config)
        self.params.update(self.DEFAULT_PARAMS)
        self.params.update(kwargs)
        self._build_model(labels)

    def _build_model(self, label_count):
        label_count = 2
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

        thermal, flow, mask = self.process_inputs()
        frame_count = tf.shape(input=self.X)[0]
        # -------------------------------------
        # run the Convolutions

        layer = thermal
        layer = self.conv_layer("thermal/1", layer, 32, [3, 3], conv_stride=2)

        layer = self.conv_layer("thermal/2", layer, 48, [3, 3], conv_stride=2)
        layer = self.conv_layer("thermal/3", layer, 64, [3, 3], conv_stride=2)
        layer = self.conv_layer("thermal/4", layer, 64, [3, 3], conv_stride=2)
        layer = self.conv_layer("thermal/5", layer, 64, [3, 3], conv_stride=1)

        filtered_conv = layer
        logging.info("Thermal convolution output shape: {}".format(filtered_conv.shape))
        filtered_out = tf.reshape(
            filtered_conv,
            [ frame_count, tools.product(filtered_conv.shape[1:])],
            name="thermal/out",
        )
        logging.info("filtered_out convolution output shape: {}".format(filtered_out.shape))

        if self.params["enable_flow"]:
            # integrate thermal and flow into a 3 channel layer
            layer = tf.concat((thermal, flow), axis=3)
            layer = self.conv_layer("motion/1", layer, 32, [3, 3], conv_stride=2)
            layer = self.conv_layer("motion/2", layer, 48, [3, 3], conv_stride=2)
            layer = self.conv_layer("motion/3", layer, 64, [3, 3], conv_stride=2)
            layer = self.conv_layer("motion/4", layer, 64, [3, 3], conv_stride=2)
            layer = self.conv_layer("motion/5", layer, 64, [3, 3], conv_stride=1)

            motion_conv = layer
            logging.info(
                "Motion convolution output shape: {}".format(motion_conv.shape)
            )
            motion_out = tf.reshape(
                motion_conv,
                [-1, frame_count, tools.product(motion_conv.shape[1:])],
                name="motion/out",
            )

            out = tf.concat((filtered_out, motion_out), axis=2, name="out")
        else:
            out = tf.concat((filtered_out,), axis=2, name="out")

        # INFO logging.info("Output shape {}".format(out.shape))
        # INFO Output shape (None, None, 576)
        # INFO memory_output shape: None x (None, 576)
        # INFO lstm state shape: (None, 576, 2)
        # INFO memory_output output shape: (None, 576)
        # INFO memory_state output shape: (None, 576, 2)

        # -------------------------------------
        # add short term memory (GRU / LSTM)
        memory_output, memory_state = self._build_memory(out)
        
        # memory_output = out
        # tf.reshape(out,[-1,576])

        # memory_state =  tf.stack([memory_output, memory_output], axis=2)
        logging.info(
            "memory_output output shape: {}".format(memory_output.shape)
        )
        logging.info(
            "memory_state output shape: {}".format(memory_state.shape)
        )
        # -------------------------------------
        # dense / logits

        # dense layer on top of convolutional output mapping to class labels.
        logits = tf.compat.v1.layers.dense(
            inputs=memory_output, units=label_count, activation=None, name="logits"
        )
        logging.info(
            "logits output shape: {}".format(logits.shape)
        )
        tf.compat.v1.summary.histogram("weights/logits", logits)
        softmax_loss = tf.compat.v1.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(self.y, label_count),
            logits=logits,
            label_smoothing=self.params["label_smoothing"],
            scope="softmax_loss",
        )
        if self.params["l2_reg"] != 0:
            with tf.compat.v1.variable_scope("logits", reuse=True):
                logit_weights = tf.compat.v1.get_variable("kernel")

            reg_loss = (
                tf.nn.l2_loss(logit_weights, name="loss/reg") * self.params["l2_reg"]
            )
            loss = tf.add(softmax_loss, reg_loss, name="loss")
            tf.compat.v1.summary.scalar("loss/reg", reg_loss)
            tf.compat.v1.summary.scalar("loss/softmax", softmax_loss)
        else:
            # just relabel the loss node
            loss = tf.identity(softmax_loss, "loss")

        class_out = tf.argmax(input=logits, axis=1, name="class_out")
        correct_prediction = tf.equal(class_out, self.y)
        pred = tf.nn.softmax(logits, name="prediction")
        accuracy = tf.reduce_mean(
            input_tensor=tf.cast(correct_prediction, dtype=tf.float32), name="accuracy"
        )

        self.setup_novelty(logits, memory_output)
        self.setup_optimizer(loss)

        # make reference to special nodes
        tf.identity(memory_state, "state_out")
        tf.identity(memory_output, "hidden_out")
        tf.identity(logits, "logits_out")
        self.attach_nodes()
