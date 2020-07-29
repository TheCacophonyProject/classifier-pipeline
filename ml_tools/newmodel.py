import itertools
import io
from ml_tools.dataset import TrackChannels
import tensorflow as tf
import pickle
import logging
from tensorboard.plugins.hparams import api as hp

from collections import namedtuple
from ml_tools.datagenerator import DataGenerator, preprocess_frame
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import json
from ml_tools.dataset import Preprocessor
from classify.trackprediction import TrackPrediction
from sklearn.metrics import confusion_matrix

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

validate = None
model = None
file_writer_cm = None


class NewModel:
    """ Defines a deep learning model """

    # fig = plt.figure(figsize=(48, 48))
    # plt_i = 1
    MODEL_NAME = "new model"
    MODEL_DESCRIPTION = "Pre trained resnet"
    VERSION = "0.3.0"

    def __init__(self, train_config=None, labels=None, preserve_labels=False):

        self.frame_size = 48
        self.model = None
        self.datasets = None
        # namedtuple("Datasets", "train, validation, test")
        # dictionary containing current hyper parameters
        self.params = {
            # augmentation
            "base_training": False,
            "augmentation": True,
            "thermal_threshold": 10,
            "scale_frequency": 0.5,
            # dropout
            "keep_prob": 0.5,
            "lstm": False,
            # training
            "batch_size": 16,
        }
        if train_config:
            self.log_base = os.path.join(train_config.train_dir, "logs")
            self.log_dir = self.log_base
            os.makedirs(self.log_base, exist_ok=True)
            self.checkpoint_folder = os.path.join(train_config.train_dir, "checkpoints")
            self.params.update(train_config.hyper_params)
        self.labels = labels
        self.preserve_labels = preserve_labels
        self.pretrained_model = self.params.get("model", "resnetv2")
        self.preprocess_fn = None
        self.validate = None
        self.train = None

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

    def get_preprocess_fn(self):
        if self.pretrained_model == "resnet":
            return tf.keras.applications.resnet.preprocess_input

        elif self.pretrained_model == "resnetv2":
            return tf.keras.applications.resnet_v2.preprocess_input

        elif self.pretrained_model == "resnet152":
            return tf.keras.applications.resnet.preprocess_input

        elif self.pretrained_model == "vgg16":
            return tf.keras.applications.vgg16.preprocess_input

        elif self.pretrained_model == "vgg19":
            return tf.keras.applications.vgg19.preprocess_input

        elif self.pretrained_model == "mobilenet":
            return tf.keras.applications.mobilenet_v2.preprocess_input

        elif self.pretrained_model == "densenet121":
            return tf.keras.applications.densenet.preprocess_input

        elif self.pretrained_model == "inceptionresnetv2":
            return tf.keras.applications.inception_resnet_v2.preprocess_input

        return None

    def build_model(self, dense_sizes=[1024, 512]):
        # note the model already applies batch_norm
        inputs = tf.keras.Input(shape=(self.frame_size, self.frame_size, 3))

        base_model, preprocess = self.base_model((self.frame_size, self.frame_size, 3))
        self.preprocess_fn = preprocess

        x = base_model(inputs, training=self.params["base_training"])  # IMPORTANT

        if self.params["lstm"]:
            # not tested
            self.add_lstm(base_model)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)

            for i in dense_sizes:
                x = tf.keras.layers.Dense(i, activation="relu")(x)
            preds = tf.keras.layers.Dense(len(self.labels), activation="softmax")(x)
            self.model = tf.keras.models.Model(inputs, outputs=preds)
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name)

        if self.params.get("retrain_layer") is not None:
            for i, layer in enumerate(base_model.layers):
                print(i, layer.name)

                layer.trainable = i >= self.params["retrain_layer"]
        else:
            base_model.trainable = self.params["base_training"]

        self.model.summary()

        self.model.compile(
            optimizer=self.optimizer(), loss=self.loss(), metrics=["accuracy"],
        )
        global model
        model = self.model

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

    def load_weights(self, file, meta=True):
        if not self.model:
            self.build_model()
        self.model.load_weights(file)
        if meta:
            self.load_meta(os.path.basename(file))
        print("loading weights", file)

    def load_model(self, dir):
        self.model = tf.keras.models.load_model(dir)
        self.load_meta(dir)
        # base_model = tf.keras.applications.ResNet50V2(
        #     weights="imagenet", include_top=False
        # )
        # base_model.summary()
        # for i, layer in enumerate(base_model.layers):
        #     print(i, layer.name)

    def load_meta(self, dir):
        meta = json.load(open(os.path.join(dir, "metadata.txt"), "r"))
        self.params = meta["hyperparams"]
        self.labels = meta["labels"]
        self.pretrained_model = self.params.get("model", "resnetv2")
        self.preprocess_fn = self.get_preprocess_fn()
        self.frame_size = self.params.get("frame_size", 48)

    def save(self, run_name=MODEL_NAME, history=None, test_results=None):
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
        model_stats["frame_size"] = self.frame_size
        model_stats["model"] = self.pretrained_model

        if history:
            model_stats["history"] = history.history
        if test_accuracy:
            model_stats["test_loss"] = test_results[0]
            model_stats["test_acc"] = test_results[1]

        json.dump(
            model_stats,
            open(os.path.join(self.checkpoint_folder, run_name, "metadata.txt"), "w"),
            indent=4,
        )

    def close(self):
        if self.validate:
            self.validate.stop_load()
        if self.train:
            self.train.stop_load()

    def train_model(self, epochs, run_name):
        self.log_dir = os.path.join(self.log_base, run_name)
        os.makedirs(self.log_base, exist_ok=True)
        if not self.model:
            self.build_model()
        self.train = DataGenerator(
            self.datasets.train,
            self.datasets.train.labels,
            len(self.datasets.train.labels),
            batch_size=self.params.get("batch_size", 32),
            lstm=self.params.get("lstm", False),
            use_thermal=self.params.get("use_thermal", False),
            use_filtered=self.params.get("use_filtered", False),
            shuffle=True,
            model_preprocess=self.preprocess_fn,
            epochs=epochs,
            load_threads=4,
        )
        global validate
        self.validate = DataGenerator(
            self.datasets.validation,
            self.datasets.train.labels,
            len(self.datasets.train.labels),
            batch_size=self.params.get("batch_size", 32),
            lstm=self.params.get("lstm", False),
            use_thermal=self.params.get("use_thermal", False),
            use_filtered=self.params.get("use_filtered", False),
            shuffle=True,
            model_preprocess=self.preprocess_fn,
            epochs=epochs,
            load_threads=1,
        )
        validate = self.validate
        cm_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_confusion_matrix
        )
        global file_writer_cm
        file_writer_cm = tf.summary.create_file_writer(self.log_dir + "/cm")
        history = self.model.fit(
            self.train,
            validation_data=self.validate,
            epochs=epochs,
            shuffle=False,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    self.log_dir, write_graph=True, write_images=True
                ),
                cm_callback,
            ],  # log metrics
        )
        self.validate.stop_load()
        self.train.stop_load()
        test_accuracy = None
        if self.datasets.test:
            test = DataGenerator(
                self.datasets.test,
                self.datasets.train.labels,
                len(self.datasets.train.labels),
                batch_size=self.params.get("batch_size", 32),
                lstm=self.params.get("lstm", False),
                use_thermal=self.params.get("use_thermal", False),
                use_filtered=self.params.get("use_filtered", False),
                shuffle=False,
                model_preprocess=self.preprocess_fn,
                epochs=1,
                load_threads=1,
            )
            test_accuracy = self.model.evaluate(test)
            test.stop_load()

            logging.info("Test accuracy is %s", test_accuracy)
        self.save(run_name, history=history, test_results=test_accuracy)
        for key, value in history.history.items():
            plt.figure()
            plt.plot(value, label="Training {}".format(key))
            plt.ylabel("{}".format(key))
            plt.title("Training {}".format(key))
            plt.savefig("{}.png".format(key))

    def preprocess(self, frame, data):
        if self.use_thermal:
            channel = TrackChannels.thermal
        else:
            channel = TrackChannels.filtered
        data = data[channel]

        # normalizes data, constrast stretch good or bad?
        if self.augment:
            percent = random.randint(0, 2)
        else:
            percent = 0
        max = int(np.percentile(data, 100 - percent))
        min = int(np.percentile(data, percent))
        if max == min:
            logging.error(
                "frame max and min are the same clip %s track %s frame %s",
                frame.clip_id,
                frame.track_id,
                frame.frame_num,
            )
            return None

        data -= min
        data = data / (max - min)
        np.clip(data, a_min=0, a_max=None, out=data)

        data = data[np.newaxis, :]
        data = np.transpose(data, (1, 2, 0))
        data = np.repeat(data, 3, axis=2)
        return data

    def preprocess_old(self, frame):
        frame = [
            frame[0, :, :],
            frame[1, :, :],
            frame[4, :, :],
        ]

        frame = np.transpose(frame, (1, 2, 0))
        frame = frame[
            np.newaxis,
        ]

    def classify_frameold(self, frame):
        frame = [
            frame[0, :, :],
            frame[1, :, :],
            frame[4, :, :],
        ]

        frame = np.transpose(frame, (1, 2, 0))
        frame = frame[
            np.newaxis,
        ]
        # print(frame.shape)

        output = self.model.predict(frame)
        # print(output)
        return output[0]

    def classify_frame(self, frame, preprocess=True):

        if preprocess:
            frame = preprocess_frame(
                frame,
                (self.frame_size, self.frame_size, 3),
                self.params.get("use_thermal", True),
                augment=False,
                preprocess_fn=self.preprocess_fn,
            )
        # if NewModel.plt_i < 41:
        #     axes = NewModel.fig.add_subplot(4, 10, NewModel.plt_i)
        #     plt.imshow(tf.keras.preprocessing.image.array_to_img(frame))
        #
        #     NewModel.plt_i += 1
        output = self.model.predict(frame[np.newaxis, :])
        return output[0]

    def binarize(self):
        # set samples of each label to have a maximum cap, and exclude labels
        self.datasets.train.binarize(["wallaby"], lbl_one="Wallaby", lbl_two="Not")
        self.datasets.validation.binarize(["wallaby"], lbl_one="Wallaby", lbl_two="Not")
        self.datasets.test.binarize(["wallaby"], lbl_one="Wallaby", lbl_two="Not")

        self.set_labels()
        print(self.labels)

    def rebalance(self, train_cap=1000, validate_cap=500, exclude=[], update=True):
        # set samples of each label to have a maximum cap, and exclude labels
        self.datasets.train.rebalance(
            label_cap=train_cap, exclude=exclude, update=update
        )
        self.datasets.validation.rebalance(
            label_cap=validate_cap, exclude=exclude, update=update
        )
        self.datasets.test.rebalance(
            label_cap=validate_cap, exclude=exclude, update=update
        )
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
        self.datasets.train.set_read_only(True)
        self.datasets.validation.set_read_only(True)
        self.datasets.test.set_read_only(True)

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

        input_layer = tf.keras.Input(shape=(27, self.frame_size, self.frame_size, 3))
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

    def classify_track(self, track_id, data, keep_all=True):
        track_prediction = TrackPrediction(track_id, 0, keep_all)
        for i, frame in enumerate(data):
            prediction = self.classify_frame(frame)
            track_prediction.classified_frame(i, prediction, None)
        # plt.savefig("testimage.png")
        # plt.close(NewModel.fig)
        return track_prediction

    def evaluate(self, dataset):
        test = DataGenerator(
            dataset,
            self.labels,
            len(self.labels),
            batch_size=self.params.get("batch_size", 32),
            lstm=self.params.get("lstm", False),
            use_thermal=self.params.get("use_thermal", False),
            use_filtered=self.params.get("use_filtered", False),
            shuffle=False,
            model_preprocess=self.preprocess_fn,
            epochs=1,
            load_threads=1,
        )
        test_accuracy = self.model.evaluate(test)
        test.stop_load()

        logging.info("Test accuracy is %s", test_accuracy)


def plot_confusion_matrix(cm, class_names):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    x, y = validate.get_data()
    test_pred_raw = model.predict(x)
    test_pred = np.argmax(test_pred_raw, axis=1)
    # Calculate the confusion matrix.

    cm = confusion_matrix(y, test_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=validate.labels)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
