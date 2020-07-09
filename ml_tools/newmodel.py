import tensorflow as tf
import pickle
import logging
from tensorboard.plugins.hparams import api as hp

from collections import namedtuple
from ml_tools.datagenerator import DataGenerator
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import json
from ml_tools.dataset import Preprocessor

#
HP_DENSE_SIZES = hp.HParam(
    "dense_sizes",
    hp.Discrete(["1024 1024 1024 1024 512", "1024 512", "512", "128", "64"]),
)

HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([32]))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "sgd"]))
HP_LEARNING_RATE = hp.HParam(
    "learning_rate", hp.Discrete([0.0001, 0.001, 0.01, 0.1, 1.0])
)

METRIC_ACCURACY = "accuracy"


class NewModel:
    """ Defines a deep learning model """

    MODEL_NAME = "new model"
    MODEL_DESCRIPTION = "Pre trained resnet"
    VERSION = "0.3.0"

    def __init__(self, train_config=None, labels=None, preserve_labels=False):
        self.log_base = os.path.join(train_config.train_dir, "logs")
        self.log_dir = self.log_base
        os.makedirs(self.log_base, exist_ok=True)
        self.checkpoint_folder = os.path.join(train_config.train_dir, "checkpoints")

        self.model = None
        self.datasets = None
        # namedtuple("Datasets", "train, validation, test")
        # dictionary containing current hyper parameters
        self.params = {
            # augmentation
            "augmentation": True,
            "thermal_threshold": 10,
            "scale_frequency": 0.5,
            # dropout
            "keep_prob": 0.5,
            # training
            "batch_size": 16,
            "use_filtered": True,
        }
        self.params.update(train_config.hyper_params)
        self.labels = labels
        self.preserve_labels = preserve_labels
        self.pretrained_model = self.params.get("model", "resnetv2")
        self.preprocess_fn = None

    def base_model(self, input_shape):
        if self.pretrained_model == "resnet":
            return (
                tf.keras.applications.ResNet50(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif self.pretrained_model == "resnetv2":
            return (
                tf.keras.applications.ResNet50V2(
                    weights="imagenet", include_top=False, input_shape=input_shape
                ),
                tf.keras.applications.resnet_v2.preprocess_input,
            )
        elif self.pretrained_model == "resnet152":
            return (
                tf.keras.applications.ResNet152(
                    weights="imagenet", include_top=False, input_shape=input_shape
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif self.pretrained_model == "vgg16":
            return (
                tf.keras.applications.VGG16(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.vgg16.preprocess_input,
            )
        elif self.pretrained_model == "vgg19":
            return (
                tf.keras.applications.VGG19(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.vgg19.preprocess_input,
            )
        elif self.pretrained_model == "mobilenet":
            return (
                tf.keras.applications.MobileNetV2(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.mobilenet_v2.preprocess_input,
            )
        elif self.pretrained_model == "densenet121":
            return (
                tf.keras.applications.DenseNet121(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.densenet.preprocess_input,
            )
        elif self.pretrained_model == "inceptionresnetv2":
            return (
                tf.keras.applications.InceptionResNetV2(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.inception_resnet_v2.preprocess_input,
            )

        raise "Could not find model" + self.pretrained_model

    def build_model(self, dense_sizes=[1024]):
        # note the model already applies batch_norm
        inputs = tf.keras.Input(shape=(48, 48, 3))

        base_model, preprocess = self.base_model((48, 48, 3))
        self.preprocess_fn = preprocess

        base_model.trainable = False
        x = base_model(inputs, training=False)  # IMPORTANT

        if self.params["lstm"]:
            # not tested
            self.add_lstm(base_model)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)

            for i in dense_sizes:
                x = tf.keras.layers.Dense(i, activation="relu")(x)
            preds = tf.keras.layers.Dense(len(self.labels), activation="softmax")(x)
            self.model = tf.keras.models.Model(inputs, outputs=preds)

        self.model.summary()

        self.model.compile(
            optimizer=self.optimizer(), loss=self.loss(), metrics=["accuracy"],
        )

    def build_model_mobile(self):
        # note the model already applies batch_norm
        IMG_SHAPE = (48, 48, 3)

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1)(x)
        self.model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

        self.model.compile(
            optimizer=self.optimizer(),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def loss(self):
        softmax = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=self.params["label_smoothing"],
        )
        return softmax

    def optimizer(self):
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
            learning_rate = self.params["learning_rate"]  # setup optimizer
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        return optimizer

    def load_weights(self, file):
        if not self.model:
            self.build_model()
        self.model.load_weights(file)
        self.load_meta(os.path.basename(file))

    def load_model(self, dir):
        self.model = tf.keras.models.load_model(dir)
        self.load_meta(dir)

    def load_meta(self, dir):
        meta = json.load(open(os.path.join(dir, "metadata.txt"), "r"))
        self.params = meta["hyperparams"]
        self.labels = meta["labels"]

    def save(self, run_name=MODEL_NAME):
        # create a save point
        self.model.save(
            os.path.join(self.checkpoint_folder, run_name), save_format="tf"
        )

        model_stats = {}
        model_stats["name"] = self.MODEL_NAME
        model_stats["description"] = self.MODEL_DESCRIPTION
        model_stats["labels"] = self.labels
        model_stats["hyperparams"] = self.params
        model_stats["training_date"] = str(time.time())
        model_stats["version"] = self.VERSION
        json.dump(
            model_stats,
            open(os.path.join(self.checkpoint_folder, run_name, "metadata.txt"), "w"),
            indent=4,
        )

    def close(self):
        pass

    def train_model(self, epochs, run_name):
        self.log_dir = os.path.join(self.log_base, run_name)
        os.makedirs(self.log_base, exist_ok=True)
        if not self.model:
            self.build_model()
        train = DataGenerator(
            self.datasets.train,
            len(self.datasets.train.labels),
            batch_size=self.params.get("batch_size", 32),
            lstm=self.params.get("lstm", False),
            use_thermal=self.params.get("use_thermal", False),
            use_filtered=self.params.get("use_filtered", False),
            shuffle=True,
            preprocess_fn=self.preprocess_fn,
        )
        validate = DataGenerator(
            self.datasets.validation,
            len(self.datasets.train.labels),
            batch_size=self.params.get("batch_size", 32),
            lstm=self.params.get("lstm", False),
            use_thermal=self.params.get("use_thermal", False),
            use_filtered=self.params.get("use_filtered", False),
            shuffle=True,
            preprocess_fn=self.preprocess_fn,
        )
        history = self.model.fit(
            train,
            validation_data=validate,
            epochs=10,
            shuffle=False,
            callbacks=[tf.keras.callbacks.TensorBoard(self.log_dir)],  # log metrics
        )

        self.save(run_name)
        for key, value in history.history.items():
            plt.figure()
            plt.plot(value, label="Training {}".format(key))
            plt.ylabel("{}".format(key))
            plt.title("Training {}".format(key))
            plt.savefig("{}.png".format(key))

    def preprocess(self, frame):
        thermal_reference = np.median(frame[0])
        frames = Preprocessor.apply([frame], [thermal_reference], default_inset=0)
        return frames[0]

    def classify_frame(self, frame):
        data_i = 1
        if self.params["use_thermal"]:
            data_i = 0
        frame = [
            frame[data_i, :, :],
            frame[data_i, :, :],
            frame[data_i, :, :],
        ]

        frame = np.transpose(frame, (1, 2, 0))
        frame = frame[
            np.newaxis,
        ]
        output = self.model.predict(frame)
        return output[0]

    def rebalance(self, train_cap=1000, validate_cap=500, exclude=[]):
        # set samples of each label to have a maximum cap, and exclude labels
        self.datasets.train.rebalance(train_cap, exclude)
        self.datasets.validation.rebalance(validate_cap, exclude)
        self.set_labels()

    def set_labels(self):
        # preserve label order if needed, this should be used when retraining
        # on a model already trained with our data
        if self.labels is None or self.preserve_labels == False:
            self.labels = self.datasets.train.labels
        else:
            for label in self.datasets.train.labels:
                if label not in self.labels:
                    self.labels.append(label)
            self.datasets.train.labels = label

    def import_dataset(self, dataset_filename, ignore_labels=None):
        """
        Import dataset.
        :param dataset_filename: path and filename of the dataset
        :param ignore_labels: (optional) these labels will be removed from the dataset.
        :return:
        """

        self.datasets = namedtuple("Datasets", "train, validation, test")
        datasets = pickle.load(open(dataset_filename, "rb"))
        self.datasets.train, self.datasets.validation, self.datasets.test = datasets
        self.labels = self.datasets.train.labels

        # augmentation really helps with reducing over-fitting, but test set should be fixed so we don't apply it there.
        self.datasets.train.enable_augmentation = self.params["augmentation"]
        self.datasets.validation.enable_augmentation = False
        self.datasets.test.enable_augmentation = False
        for dataset in datasets:
            if ignore_labels:
                for label in ignore_labels:
                    dataset.remove_label(label)

        logging.info(
            "Training frames: {0:.1f}k".format(self.datasets.train.rows / 1000)
        )
        logging.info(
            "Validation frames: {0:.1f}k".format(self.datasets.validation.rows / 1000)
        )
        logging.info("Test segments: {0:.1f}k".format(self.datasets.test.rows / 1000))
        logging.info("Labels: {}".format(self.labels))

    #  was some problem with batch norm in old tensorflows
    # when training accuracy would be high, but when evaluating on same data
    # it would be low, this tests that, all printed accuracies should be close
    def learning_phase_test(self, dataset):
        self.save()
        dataset.shuffle = False
        _, accuracy = self.model.evaluate(dataset)
        print("dynamic", accuracy)
        tf.keras.backend.set_learning_phase(0)
        self.load_model(os.path.join(self.checkpoint_folder, "resnet50/"),)
        _, accuracy = self.model.evaluate(dataset)
        print("learning0", accuracy)

        tf.keras.backend.set_learning_phase(1)
        self.load_model(os.path.join(self.checkpoint_folder, "resnet50/"),)
        _, accuracy = self.model.evaluate(dataset)
        print("learning1", accuracy)

    # GRID SEARCH
    def train_test_model(self, hparams, log_dir, epochs=1):
        # if not self.model:
        dense_size = hparams[HP_DENSE_SIZES].split()
        for i, size in enumerate(dense_size):
            dense_size[i] = int(size)
        self.build_model(dense_sizes=dense_size)
        train = DataGenerator(
            self.datasets.train,
            len(self.datasets.train.labels),
            batch_size=hparams.get(HP_BATCH_SIZE, 32),
            lstm=self.params.get("lstm", False),
            use_thermal=self.params.get("use_thermal", False),
            use_filtered=self.params.get("use_filtered", False),
            shuffle=False,
            preprocess_fn=self.preprocess_fn,
        )
        validate = DataGenerator(
            self.datasets.validation,
            len(self.datasets.train.labels),
            batch_size=hparams.get(HP_BATCH_SIZE, 32),
            lstm=self.params.get("lstm", False),
            use_thermal=self.params.get("use_thermal", False),
            use_filtered=self.params.get("use_filtered", False),
            shuffle=False,
            preprocess_fn=self.preprocess_fn,
        )
        opt = None
        learning_rate = hparams[HP_LEARNING_RATE]
        if hparams[HP_OPTIMIZER] == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        self.model.compile(
            optimizer=opt, loss=self.loss(), metrics=["accuracy"],
        )
        history = self.model.fit(train, epochs=epochs,)

        _, accuracy = self.model.evaluate(validate)
        return accuracy

    def test_hparams(self):
        dir = self.log_dir + "/hparam_tuning"
        with tf.summary.create_file_writer(dir).as_default():
            hp.hparams_config(
                hparams=[HP_BATCH_SIZE, HP_LEARNING_RATE],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
            )
        session_num = 0

        for batch_size in HP_BATCH_SIZE.domain.values:
            for dense_size in HP_DENSE_SIZES.domain.values:
                for learning_rate in HP_LEARNING_RATE.domain.values:
                    for optimizer in HP_OPTIMIZER.domain.values:
                        hparams = {
                            HP_DENSE_SIZES: dense_size,
                            HP_BATCH_SIZE: batch_size,
                            HP_LEARNING_RATE: learning_rate,
                            HP_OPTIMIZER: optimizer,
                        }
                        run_name = "run-%d" % session_num
                        print("--- Starting trial: %s" % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        self.run(dir + "/" + run_name, hparams)
                        session_num += 1

    def run(self, log_dir, hparams):
        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = self.train_test_model(hparams, log_dir)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

    @property
    def hyperparams_string(self):
        """ Returns list of hyperparameters as a string. """
        return "\n".join(
            ["{}={}".format(param, value) for param, value in self.params.items()]
        )

    def add_lstm(self, base_model):
        model2 = tf.keras.models.Model(inputs=base_model.input, outputs=x)

        input_layer = tf.keras.Input(shape=(27, 48, 48, 3))
        curr_layer = tf.keras.layers.TimeDistributed(model2)(input_layer)
        curr_layer = tf.keras.layers.Reshape(target_shape=(27, 2048))(curr_layer)
        memory_output, memory_state = self.lstm(curr_layer)
        x = memory_output
        x = tf.keras.layers.Dense(self.params["lstm_units"], activation="relu")(x)

        tf.identity(memory_state, "state_out")

        preds = tf.keras.layers.Dense(
            len(self.datasets.train.labels), activation="softmax"
        )(x)

        self.model = tf.keras.models.Model(input_layer, preds)

        #
        #         encoded_frames = tf.keras.layers.TimeDistributed(self.model)(input_layer)
        #         encoded_sequence = LSTM(512)(encoded_frames)
        #
        # hidden_layer = Dense(1024, activation="relu")(encoded_sequence)
        # outputs = Dense(50, activation="softmax")(hidden_layer)
        # model = Model([inputs], outputs)

    def lstm(self, inputs):
        lstm_cell = tf.keras.layers.LSTMCell(
            self.params["lstm_units"], dropout=self.params["keep_prob"]
        )
        rnn = tf.keras.layers.RNN(
            lstm_cell,
            return_sequences=True,
            return_state=True,
            dtype=tf.float32,
            unroll=False,
        )
        # whole_seq_output, final_memory_state, final_carry_state = rnn(inputs)
        lstm_outputs, lstm_state_1, lstm_state_2 = rnn(inputs)

        lstm_output = tf.identity(lstm_outputs[:, -1], "lstm_out")
        lstm_state = tf.stack([lstm_state_1, lstm_state_2], axis=2)
        return lstm_output, lstm_state


def preprocess_resv2(data):
    data = data * 255.0
    data = tf.keras.applications.resnet_v2.preprocess_input(data, data_format=None)
    return data
