import tensorflow as tf
import pickle
import logging
from collections import namedtuple
from ml_tools.datagenerator import DataGenerator
import os
import matplotlib.pyplot as plt


class NewModel:
    """ Defines a deep learning model """

    MODEL_NAME = "new model"
    MODEL_DESCRIPTION = ""
    VERSION = "0.3.0"

    def __init__(self, datasets_filename=None, train_config=None, labels=None):
        self.log_dir = os.path.join(train_config.train_dir, "logs")
        self.checkpoint_folder = os.path.join(train_config.train_dir, "checkpoints")
        self.model = None
        self.datasets = namedtuple("Datasets", "train, validation, test")
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
        }
        self.params.update(train_config.hyper_params)
        self.labels = labels
        if datasets_filename:
            self.import_dataset(datasets_filename)
            self.labels = self.datasets.train.labels

        self.build_model()

    def build_model(self):
        # note the model already applies batch_norm
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=(48, 48, 3)
        )

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        if self.params["lstm"]:
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
        else:
            x = tf.keras.layers.Dense(1024, activation="relu")(x)
            x = tf.keras.layers.Dense(1024, activation="relu")(x)
            x = tf.keras.layers.Dense(1024, activation="relu")(x)
            x = tf.keras.layers.Dense(512, activation="relu")(x)
            preds = tf.keras.layers.Dense(len(self.labels), activation="softmax")(x)
            self.model = tf.keras.models.Model(inputs=base_model.input, outputs=preds)

        # print(self.model.summary())

        #
        #         encoded_frames = tf.keras.layers.TimeDistributed(self.model)(input_layer)
        #         encoded_sequence = LSTM(512)(encoded_frames)
        #
        # hidden_layer = Dense(1024, activation="relu")(encoded_sequence)
        # outputs = Dense(50, activation="softmax")(hidden_layer)
        # model = Model([inputs], outputs)
        self.model.compile(
            optimizer=self.optimizer(),
            loss=self.loss(),
            metrics=["accuracy", tf.keras.metrics.Precision()],
        )

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
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        return optimizer

    def load_model(self, file):
        self.model = tf.keras.models.load_model(file)
        # self.model.load_model(file)

    def save(self):
        # create a save point
        self.model.save(os.path.join(self.checkpoint_folder, "resnet50"))

    def close(self):
        pass

    def train_model(self, epochs, run_name):
        train = DataGenerator(
            self.datasets.train,
            len(self.datasets.train.labels),
            lstm=self.params.get("lstm", False),
            thermal_only=self.params.get("thermal_only", False),
        )
        validate = DataGenerator(
            self.datasets.validation,
            len(self.datasets.train.labels),
            lstm=self.params.get("lstm", False),
            thermal_only=self.params.get("thermal_only", False),
        )
        # test = DataGenerator(self.datasets.test, len(self.datasets.test.labels))

        history = self.model.fit(train, validation_data=validate, epochs=epochs)
        self.save()
        for key, value in history.history.items():
            plt.figure()
            plt.plot(value, label="Training {}".format(key))
            plt.ylabel("{}".format(key))
            plt.title("Training {}".format(key))
            plt.savefig("{}.png".format(key))

    def evaluate(self):
        # infer = self.model.signatures["serving_default"]
        #
        # labeling = infer(tf.constant(x))[self.model.output_names[0]]
        test = DataGenerator(self.datasets.test, len(self.datasets.train.labels))
        scalars = self.model.evaluate(test)
        # print(self.model.metrics_name)

        print(scalars)
        # acc = history.history["acc"]
        # loss = history.history["loss"]
        # print(acc)
        # print(loss)
        # plt.figure()
        # plt.plot(acc, label="Training Accuracy")
        # plt.ylabel("Accuracy")
        # plt.title("Training Accuracy")
        # plt.savefig("accuracy.png")
        # plt.figure()
        #
        # plt.plot(loss, label="Training Loss")
        # plt.ylabel("Loss")
        # plt.title("Training Loss")
        # plt.xlabel("epoch")
        # plt.savefig("loss.png")

    def import_dataset(self, dataset_filename, ignore_labels=None):
        """
        Import dataset.
        :param dataset_filename: path and filename of the dataset
        :param ignore_labels: (optional) these labels will be removed from the dataset.
        :return:
        """
        datasets = pickle.load(open(dataset_filename, "rb"))
        self.datasets.train, self.datasets.validation, self.datasets.test = datasets

        # augmentation really helps with reducing over-fitting, but test set should be fixed so we don't apply it there.
        self.datasets.train.enable_augmentation = self.params["augmentation"]
        self.datasets.train.scale_frequency = self.params["scale_frequency"]
        self.datasets.validation.enable_augmentation = False
        self.datasets.test.enable_augmentation = False
        for dataset in datasets:
            if ignore_labels:
                for label in ignore_labels:
                    dataset.remove_label(label)

        self.labels = self.datasets.train.labels.copy()

        logging.info(
            "Training segments: {0:.1f}k".format(self.datasets.train.rows / 1000)
        )
        logging.info(
            "Validation segments: {0:.1f}k".format(self.datasets.validation.rows / 1000)
        )
        logging.info("Test segments: {0:.1f}k".format(self.datasets.test.rows / 1000))
        logging.info("Labels: {}".format(self.datasets.train.labels))

        # assert set(self.datasets.train.labels).issubset(
        #     set(self.datasets.validation.labels)
        # )
        # assert set(self.datasets.train.labels).issubset(set(self.datasets.test.labels))
