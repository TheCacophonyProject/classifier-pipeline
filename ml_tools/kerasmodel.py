import itertools
import io
import math
from ml_tools.dataset import TrackChannels
import tensorflow as tf
import pickle
import logging
from tensorboard.plugins.hparams import api as hp

from collections import namedtuple
from ml_tools.datagenerator import (
    DataGenerator,
    preprocess_frame,
    preprocess_lstm,
    saveimages,
    preprocess_movement,
)
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import json
from ml_tools.dataset import Preprocessor, filtered_is_valid
from classify.trackprediction import TrackPrediction
from sklearn.metrics import confusion_matrix

from ml_tools.hyperparams import HyperParams

#
HP_DENSE_SIZES = hp.HParam("dense_sizes", hp.Discrete(["1024 512"]),)
HP_TYPE = hp.HParam("type", hp.Discrete([14]))

HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([16, 32]))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam"]))
HP_LEARNING_RATE = hp.HParam("learning_rate", hp.Discrete([0.001, 0.0001]))
HP_EPSILON = hp.HParam("epislon", hp.Discrete([1e-7]))  # 1.0 and 0.1 for inception
HP_RETRAIN = hp.HParam("retrain_layer", hp.Discrete([-1]))
HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.2, 0.1, 0.0]))

METRIC_ACCURACY = "accuracy"
METRIC_LOSS = "loss"


class KerasModel:
    """ Defines a deep learning model """

    MODEL_NAME = "keras model"
    MODEL_DESCRIPTION = "Using pre trained keras application models"
    VERSION = "0.3.0"

    def __init__(self, train_config=None, labels=None, type=0):
        self.type = type
        self.model = None
        self.datasets = None
        # dictionary containing current hyper parameters
        self.params = HyperParams()
        if train_config:
            self.log_base = os.path.join(train_config.train_dir, "logs")
            self.log_dir = self.log_base
            os.makedirs(self.log_base, exist_ok=True)
            self.checkpoint_folder = os.path.join(train_config.train_dir, "checkpoints")
            self.params.update(train_config.hyper_params)
        self.labels = labels
        self.preprocess_fn = None
        self.validate = None
        self.train = None

    def base_model(self, input_shape):
        pretrained_model = self.params.model
        if pretrained_model == "resnet":
            return (
                tf.keras.applications.ResNet50(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif pretrained_model == "resnetv2":
            return (
                tf.keras.applications.ResNet50V2(
                    weights="imagenet", include_top=False, input_shape=input_shape
                ),
                tf.keras.applications.resnet_v2.preprocess_input,
            )
        elif pretrained_model == "resnet152":
            return (
                tf.keras.applications.ResNet152(
                    weights="imagenet", include_top=False, input_shape=input_shape
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif pretrained_model == "vgg16":
            return (
                tf.keras.applications.VGG16(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.vgg16.preprocess_input,
            )
        elif pretrained_model == "vgg19":
            return (
                tf.keras.applications.VGG19(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.vgg19.preprocess_input,
            )
        elif pretrained_model == "mobilenet":
            return (
                tf.keras.applications.MobileNetV2(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.mobilenet_v2.preprocess_input,
            )
        elif pretrained_model == "densenet121":
            return (
                tf.keras.applications.DenseNet121(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.densenet.preprocess_input,
            )
        elif pretrained_model == "inceptionresnetv2":
            return (
                tf.keras.applications.InceptionResNetV2(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.inception_resnet_v2.preprocess_input,
            )
        elif pretrained_model == "inceptionv3":
            return (
                tf.keras.applications.InceptionV3(
                    weights="imagenet", include_top=False, input_shape=input_shape,
                ),
                tf.keras.applications.inception_v3.preprocess_input,
            )
        raise Exception("Could not find model" + pretrained_model)

    def get_preprocess_fn(self):
        pretrained_model = self.params.model
        if pretrained_model == "resnet":
            return tf.keras.applications.resnet.preprocess_input

        elif pretrained_model == "resnetv2":
            return tf.keras.applications.resnet_v2.preprocess_input

        elif pretrained_model == "resnet152":
            return tf.keras.applications.resnet.preprocess_input

        elif pretrained_model == "vgg16":
            return tf.keras.applications.vgg16.preprocess_input

        elif pretrained_model == "vgg19":
            return tf.keras.applications.vgg19.preprocess_input

        elif pretrained_model == "mobilenet":
            return tf.keras.applications.mobilenet_v2.preprocess_input

        elif pretrained_model == "densenet121":
            return tf.keras.applications.densenet.preprocess_input

        elif pretrained_model == "inceptionresnetv2":
            return tf.keras.applications.inception_resnet_v2.preprocess_input
        elif pretrained_model == "inceptionv3":
            return tf.keras.applications.inception_v3.preprocess_input
        return None

    def build_model(self, dense_sizes=None, retrain_from=None, dropout=None):

        width = self.params.frame_size
        if self.params.use_movement:
            width = self.params.square_width * self.params.frame_size

        inputs = tf.keras.Input(shape=(width, width, 3), name="input")

        base_model, preprocess = self.base_model((width, width, 3))
        self.preprocess_fn = preprocess
        x = base_model(inputs, training=self.params.base_training)  # IMPORTANT

        if self.params.lstm:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            for i in dense_sizes:
                x = tf.keras.layers.Dense(i, activation="relu")(x)
            # gp not sure how many should be pre lstm, and how many post
            cnn = tf.keras.models.Model(inputs, outputs=x)

            self.model = self.add_lstm(cnn)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            for i in dense_sizes:
                x = tf.keras.layers.Dense(i, activation="relu")(x)
                if dropout:
                    x = tf.keras.layers.Dropout(dropout)(x)

            preds = tf.keras.layers.Dense(
                len(self.labels), activation="softmax", name="prediction"
            )(x)
            self.model = tf.keras.models.Model(inputs, outputs=preds)

        if retrain_from is None:
            retrain_from = self.params.retrain_layer
        if retrain_from:
            for i, layer in enumerate(base_model.layers):
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    # apparently this shouldn't matter as we set base_training = False
                    layer.trainable = False
                    logging.info("dont train %s %s", i, layer.name)
                else:
                    layer.trainable = i >= retrain_from
        else:
            base_model.trainable = self.params.base_training

        self.model.summary()

        self.model.compile(
            optimizer=self.optimizer(),
            loss=self.loss(),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
            ],
        )

    def loss(self):
        softmax = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=self.params.label_smoothing,
        )
        return softmax

    def optimizer(self):
        if self.params.learning_rate_decay != 1.0:
            learning_rate = tf.compat.v1.train.exponential_decay(
                self.params.learning_rate,
                self.global_step,
                1000,
                self.params["learning_rate_decay"],
                staircase=True,
            )
            tf.compat.v1.summary.scalar("params/learning_rate", learning_rate)
        else:
            learning_rate = self.params.learning_rate  # setup optimizer
        if learning_rate:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        return optimizer

    def load_weights(self, model_path, meta=True):
        logging.info("loading weights %s", model_path)
        dir = os.path.dirname(model_path)

        if meta:
            self.load_meta(dir)
        if not self.model:
            self.build_model(
                dense_sizes=self.params.dense_sizes,
                retrain_from=self.params.retrain_layer,
                dropout=self.params.dropout,
            )
        self.model.load_weights(dir + "/variables/variables")

    def load_model(self, model_path):
        logging.info("Loading %s", model_path)
        self.model = tf.keras.models.load_model(model_path)
        dir = os.path.dirname(model_path)
        self.load_meta(dir)
        self.model.summary()

    def load_meta(self, dir):
        meta = json.load(open(os.path.join(dir, "metadata.txt"), "r"))
        self.params = HyperParams()
        self.params.update(meta["hyperparams"])
        self.labels = meta["labels"]
        self.preprocess_fn = self.get_preprocess_fn()
        self.type = meta.get("type")

    def save(self, run_name=MODEL_NAME, history=None, test_results=None):
        # create a save point
        self.model.save(
            os.path.join(self.checkpoint_folder, run_name), save_format="tf"
        )
        self.save_metadata(run_name, history, test_results)

    def save_metadata(self, run_name=MODEL_NAME, history=None, test_results=None):
        #  save metadata
        model_stats = {}
        model_stats["name"] = self.MODEL_NAME
        model_stats["description"] = self.MODEL_DESCRIPTION
        model_stats["labels"] = self.labels
        model_stats["hyperparams"] = self.params
        model_stats["training_date"] = str(time.time())
        model_stats["version"] = self.VERSION
        model_stats["type"] = self.type
        if history:
            model_stats["history"] = history.history
        if test_results:
            model_stats["test_loss"] = test_results[0]
            model_stats["test_acc"] = test_results[1]

        run_dir = os.path.join(self.checkpoint_folder, run_name)
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        json.dump(
            model_stats, open(os.path.join(run_dir, "metadata.txt"), "w",), indent=4,
        )
        best_loss = os.path.join(run_dir, "val_loss")
        if os.path.exists(best_loss):
            json.dump(
                model_stats,
                open(os.path.join(best_loss, "metadata.txt"), "w",),
                indent=4,
            )
        best_acc = os.path.join(run_dir, "val_acc")
        if os.path.exists(best_acc):
            json.dump(
                model_stats,
                open(os.path.join(best_acc, "metadata.txt"), "w",),
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

        self.train = DataGenerator(
            self.datasets.train,
            self.labels,
            len(self.datasets.train.labels),
            batch_size=self.params.batch_size,
            lstm=self.params.lstm,
            buffer_size=self.params.buffer_size,
            use_thermal=self.params.use_thermal,
            use_filtered=self.params.use_filtered,
            shuffle=self.params.shuffle,
            model_preprocess=self.preprocess_fn,
            epochs=epochs,
            load_threads=self.params.train_load_threads,
            use_movement=self.params.use_movement,
            type=self.type,
            cap_at="bird",
            # label_cap=1000,
            square_width=self.params.square_width,
        )
        self.validate = DataGenerator(
            self.datasets.validation,
            self.labels,
            len(self.datasets.train.labels),
            batch_size=self.params.batch_size,
            lstm=self.params.lstm,
            buffer_size=self.params.buffer_size,
            use_thermal=self.params.use_thermal,
            use_filtered=self.params.use_filtered,
            shuffle=self.params.shuffle,
            model_preprocess=self.preprocess_fn,
            epochs=epochs,
            load_threads=1,
            use_movement=self.params.use_movement,
            type=self.type,
            cap_at="bird",
            square_width=self.params.square_width,
        )
        if not self.model:
            self.build_model(
                dense_sizes=self.params.dense_sizes,
                retrain_from=self.params.retrain_layer,
                dropout=self.params.dropout,
            )
        file_writer_cm = tf.summary.create_file_writer(self.log_dir + "/cm")
        cm_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: log_confusion_matrix(
                epoch, logs, self.model, self.validate, file_writer_cm
            )
        )
        checkpoints = self.checkpoints(run_name)

        self.save_metadata(run_name)

        history = self.model.fit(
            self.train,
            validation_data=self.validate,
            epochs=epochs,
            shuffle=False,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    self.log_dir, write_graph=True, write_images=True
                ),
                *checkpoints
                # cm_callback,
            ],  # log metrics
        )
        self.validate.stop_load()
        self.train.stop_load()
        test_accuracy = None
        if self.datasets.test and self.datasets.test.has_data():
            test = DataGenerator(
                self.datasets.test,
                self.datasets.train.labels,
                len(self.datasets.train.labels),
                batch_size=self.params.batch_size,
                lstm=self.params.lstm,
                use_thermal=self.params.use_thermal,
                use_filtered=self.params.use_filtered,
                use_movement=self.params.use_movement,
                shuffle=True,
                model_preprocess=self.preprocess_fn,
                epochs=1,
                load_threads=4,
                cap_samples=True,
                cap_at="bird",
                type=self.type,
                square_width=self.params.square_width,
            )
            test_accuracy = self.model.evaluate(test)
            test.stop_load()
            logging.info("Test accuracy is %s", test_accuracy)
        self.save(run_name, history=history, test_results=test_accuracy)

    def checkpoints(self, run_name):
        val_loss = os.path.join(self.checkpoint_folder, run_name, "val_loss")

        checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(
            val_loss,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
        )
        val_acc = os.path.join(self.checkpoint_folder, run_name, "val_acc")

        checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
            val_acc,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="max",
        )

        val_precision = os.path.join(self.checkpoint_folder, run_name, "val_recall")

        checkpoint_recall = tf.keras.callbacks.ModelCheckpoint(
            val_precision,
            monitor="val_recall",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="max",
        )
        return [checkpoint_acc, checkpoint_loss, checkpoint_recall]

    def classify_frames(self, data, preprocess=True, regions=None):
        predictions = []
        if self.params.use_thermal:
            channel = TrackChannels.thermal
        else:
            channel = TrackChannels.filtered

        filtered_data = []
        valid_indices = []
        for i, frame in enumerate(data):
            if filtered_is_valid(frame, ""):
                filtered_data.append((i, frame))
                valid_indices.append(i)

        frame_sample = valid_indices
        frame_sample.extend(valid_indices)
        frame_sample.extend(valid_indices)
        np.random.shuffle(frame_sample)
        if self.params.lstm:
            median = []
            for f in data:
                median.append(np.median(f[0]))

            data, _ = Preprocessor.apply(data, median, default_inset=0,)
            data = preprocess_lstm(
                data,
                (self.params.frame_size, self.params.frame_size, 3),
                channel,
                augment=False,
                preprocess_fn=self.preprocess_fn,
            )
            output = self.model.predict(data[np.newaxis, :])
            predictions.append(output[0])
        elif self.params.use_movement:
            frames_per_classify = self.params.square_width ** 2
            frames = len(filtered_data)

            n_squares = 3 * math.ceil(float(frames) / frames_per_classify)
            median = np.zeros((frames_per_classify))

            for i in range(n_squares):
                if self.type >= 4:
                    square_data = filtered_data
                    region_data = regions
                    seg_frames = frame_sample[:frames_per_classify]
                    if len(seg_frames) == 0:
                        break
                    # print("using", seg_frames)
                    segment = []
                    median = np.zeros((len(seg_frames)))
                    # update remaining
                    frame_sample = frame_sample[frames_per_classify:]
                    seg_frames.sort()
                    for i, frame_i in enumerate(seg_frames):
                        f = data[frame_i]
                        segment.append(f)
                        median[i] = np.median(f[0])
                else:
                    start = i * frames_per_classify
                    end = start + frames_per_classify
                    if end > len(data):
                        end = len(data)
                        start = len(data) - frames_per_classify

                    square_data = data[start:end]
                    region_data = regions[start:end]
                    segment = square_data
                    for i, f in enumerate(segment):
                        median[i] = np.median(f[0])
                frames = preprocess_movement(
                    square_data,
                    (segment, median),
                    self.params.square_width,
                    region_data,
                    channel,
                    self.preprocess_fn,
                    type=self.type,
                )
                if frames is None:
                    print("frames are none")
                    continue
                output = self.model.predict(frames[np.newaxis, :])
                predictions.append(output[0])
        return predictions

    def classify_frame(self, frame, preprocess=True):
        if self.params.use_thermal:
            channel = TrackChannels.thermal
        else:
            channel = TrackChannels.filtered
        if preprocess:
            frame = preprocess_frame(
                frame,
                (self.params.frame_size, self.params.frame_size, 3),
                channel,
                augment=False,
                preprocess_fn=self.preprocess_fn,
            )
        output = self.model.predict(frame[np.newaxis, :])
        return output[0]

    def regroup(
        self, groups, shuffle=True, random_segments=False,
    ):
        for fld in self.datasets._fields:
            dataset = getattr(self.datasets, fld)
            if random_segments:
                dataset.random_segments(require_movement=self.type == 14)
            dataset.regroup(groups, shuffle=shuffle)
        # set samples of each label to have a maximum cap, and exclude labels

        self.set_labels()

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
        self.labels = self.datasets.train.labels.copy()
        print("training with", self.labels)

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

        self.datasets.train.enable_augmentation = True
        self.datasets.validation.enable_augmentation = False
        self.datasets.test.enable_augmentation = False
        for dataset in datasets:
            dataset.set_read_only(True)
            dataset.use_segments = self.params.use_segments

            if ignore_labels:
                for label in ignore_labels:
                    dataset.remove_label(label)

        logging.info(
            "Training samples: {0:.1f}k".format(self.datasets.train.sample_count / 1000)
        )
        logging.info(
            "Validation samples: {0:.1f}k".format(
                self.datasets.validation.sample_count / 1000
            )
        )
        logging.info(
            "Test samples: {0:.1f}k".format(self.datasets.test.sample_count / 1000)
        )
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
    def train_test_model(self, hparams, log_dir, epochs=6):
        # if not self.model:
        dense_size = hparams[HP_DENSE_SIZES].split()
        retrain_layer = hparams[HP_RETRAIN]
        dropout = hparams[HP_DROPOUT]
        if dropout == 0.0:
            dropout = None
        if retrain_layer == -1:
            retrain_layer = None
        type = hparams[HP_TYPE]

        for i, size in enumerate(dense_size):
            dense_size[i] = int(size)
        self.train.batch_size = hparams.get(HP_BATCH_SIZE, 32)
        self.validate.batch_size = hparams.get(HP_BATCH_SIZE, 32)
        self.train.loaded_epochs = 0
        self.validate.loaded_epochs = 0
        self.build_model(
            dense_sizes=dense_size, retrain_from=retrain_layer, dropout=dropout,
        )

        opt = None
        learning_rate = hparams[HP_LEARNING_RATE]
        epsilon = hparams[HP_EPSILON]

        if hparams[HP_OPTIMIZER] == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, epsilon=epsilon)
        self.model.compile(
            optimizer=opt, loss=self.loss(), metrics=["accuracy"],
        )
        history = self.model.fit(
            self.train, epochs=epochs, shuffle=False, validation_data=self.validate
        )

        # _, accuracy = self.model.evaluate(self.validate)
        self.train.stop_load()
        # self.validate.stop_load()
        return history

    def test_hparams(self):
        self.datasets.train.set_samples(cap_at="wallaby", label_cap=1000)
        self.datasets.validation.set_samples(cap_at="wallaby", label_cap=200)
        epochs = 6
        type = 12
        batch_size = 32
        self.train = DataGenerator(
            self.datasets.train,
            self.datasets.train.labels,
            len(self.datasets.train.labels),
            batch_size=batch_size,
            lstm=self.params.lstm,
            buffer_size=self.params.buffer_size,
            use_thermal=self.params.use_thermal,
            use_filtered=self.params.use_filtered,
            model_preprocess=self.preprocess_fn,
            load_threads=self.params.train_load_threads,
            use_movement=self.params.use_movement,
            cap_at="wallaby",
            randomize_epoch=False,
            shuffle=True,
            keep_epoch=True,
            type=type,
            square_width=self.params.square_width,
        )
        self.validate = DataGenerator(
            self.datasets.validation,
            self.datasets.train.labels,
            len(self.datasets.train.labels),
            batch_size=batch_size,
            buffer_size=self.params.buffer_size,
            lstm=self.params.lstm,
            use_thermal=self.params.use_thermal,
            use_filtered=self.params.use_filtered,
            model_preprocess=self.preprocess_fn,
            epochs=epochs,
            use_movement=self.params.use_movement,
            cap_at="wallaby",
            randomize_epoch=False,
            shuffle=True,
            keep_epoch=True,
            type=type,
            square_width=self.params.square_width,
        )

        dir = self.log_dir + "/hparam_tuning"
        with tf.summary.create_file_writer(dir).as_default():
            hp.hparams_config(
                hparams=[
                    HP_BATCH_SIZE,
                    HP_DENSE_SIZES,
                    HP_LEARNING_RATE,
                    HP_OPTIMIZER,
                    HP_EPSILON,
                    HP_TYPE,
                    HP_RETRAIN,
                    HP_DROPOUT,
                ],
                metrics=[
                    hp.Metric(METRIC_ACCURACY, display_name="Accuracy"),
                    hp.Metric(METRIC_LOSS, display_name="Loss"),
                ],
            )
        session_num = 0
        hparams = {}
        for batch_size in HP_BATCH_SIZE.domain.values:
            for dense_size in HP_DENSE_SIZES.domain.values:
                for retrain in HP_RETRAIN.domain.values:

                    for learning_rate in HP_LEARNING_RATE.domain.values:
                        for type in HP_TYPE.domain.values:
                            for optimizer in HP_OPTIMIZER.domain.values:
                                for epsilon in HP_EPSILON.domain.values:
                                    for dropout in HP_DROPOUT.domain.values:
                                        hparams = {
                                            HP_DENSE_SIZES: dense_size,
                                            HP_BATCH_SIZE: batch_size,
                                            HP_LEARNING_RATE: learning_rate,
                                            HP_OPTIMIZER: optimizer,
                                            HP_EPSILON: epsilon,
                                            HP_TYPE: type,
                                            HP_RETRAIN: retrain,
                                            HP_DROPOUT: dropout,
                                        }
                                        run_name = "run-%d" % session_num
                                        print("--- Starting trial: %s" % run_name)
                                        print({h.name: hparams[h] for h in hparams})
                                        self.run(dir + "/" + run_name, hparams)
                                        session_num += 1

    def run(self, log_dir, hparams):

        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            history = self.train_test_model(hparams, log_dir)
            val_accuracy = history.history["val_accuracy"]
            val_loss = history.history["val_loss"]

            for step, accuracy in enumerate(val_accuracy):
                loss = val_loss[step]
                tf.summary.scalar(METRIC_ACCURACY, accuracy, step=step)
                tf.summary.scalar(METRIC_LOSS, loss, step=step)

    @property
    def hyperparams_string(self):
        """ Returns list of hyperparameters as a string. """
        print(self.params)
        return "\n".join(
            ["{}={}".format(param, value) for param, value in self.params.items()]
        )

    def add_lstm(self, cnn):
        input_layer = tf.keras.Input(
            shape=(None, self.params.frame_size, self.params.frame_size, 3)
        )
        encoded_frames = tf.keras.layers.TimeDistributed(cnn)(input_layer)
        lstm_outputs = tf.keras.layers.LSTM(
            self.params["lstm_units"],
            dropout=self.params["keep_prob"],
            return_state=False,
        )(encoded_frames)

        hidden_layer = tf.keras.layers.Dense(1024, activation="relu")(lstm_outputs)
        hidden_layer = tf.keras.layers.Dense(512, activation="relu")(hidden_layer)

        preds = tf.keras.layers.Dense(
            len(self.labels), activation="softmax", name="pred"
        )(hidden_layer)
        model = tf.keras.models.Model(input_layer, preds)
        return model

    def classify_track(self, track_id, data, keep_all=True, regions=None):
        track_prediction = TrackPrediction(track_id, 0, keep_all)
        if self.params.lstm:
            prediction = self.classify_frame(data)
            track_prediction.classified_frame(0, prediction, None)
        elif self.params.use_movement:
            predictions = self.classify_frames(data, regions=regions)
            for i, prediction in enumerate(predictions):
                track_prediction.classified_frame(i, prediction, None)

        else:
            skip = 9
            for i, frame in enumerate(data):
                if i % skip == 0:
                    continue
                prediction = self.classify_frame(frame)
                track_prediction.classified_frame(i, prediction, None)

        return track_prediction

    def confusion(self, dataset, filename="confusion.png"):
        dataset.set_read_only(True)
        dataset.use_segments = self.params.use_segments
        test = DataGenerator(
            dataset,
            self.labels,
            len(self.labels),
            batch_size=self.params.batch_size,
            lstm=self.params.lstm,
            use_thermal=self.params.use_thermal,
            use_filtered=self.params.use_filtered,
            use_movement=self.params.use_movement,
            shuffle=True,
            model_preprocess=self.preprocess_fn,
            epochs=1,
            load_threads=self.params.train_load_threads,
            keep_epoch=True,
            type=self.type,
            cap_samples=True,
            cap_at="bird",
            square_width=self.params.square_width,
        )
        test_pred_raw = self.model.predict(test)
        test.stop_load()
        test_pred = np.argmax(test_pred_raw, axis=1)

        batch_y = test.get_epoch_predictions(0)
        y = []
        for batch in batch_y:
            y.extend(np.argmax(batch, axis=1))

        # test.epoch_data = None
        cm = confusion_matrix(y, test_pred)

        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=test.labels)
        plt.savefig(filename, format="png")

    def evaluate(self, dataset):
        dataset.set_read_only(True)
        dataset.use_segments = self.params.use_segments
        test = DataGenerator(
            dataset,
            self.labels,
            len(self.labels),
            batch_size=self.params.batch_size,
            lstm=self.params.lstm,
            use_thermal=self.params.use_thermal,
            use_filtered=self.params.use_filtered,
            use_movement=self.params.use_movement,
            shuffle=True,
            model_preprocess=self.preprocess_fn,
            epochs=1,
            load_threads=self.params.train_load_threads,
            cap_samples=True,
            cap_at="bird",
            type=self.type,
            square_width=self.params.square_width,
        )
        test_accuracy = self.model.evaluate(test)
        test.stop_load()

        logging.info("Test accuracy is %s", test_accuracy)


# from tensorflow examples
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


def log_confusion_matrix(epoch, logs, model, validate, writer):
    # Use the model to predict the values from the validation dataset.
    print("doing confusing", epoch)
    batch_y = validate.get_epoch_predictions(epoch)
    validate.use_previous_epoch = epoch
    # Calculate the confusion matrix.
    if validate.keep_epoch:
        # x = np.array(x)
        y = []
        for batch in batch_y:
            y.extend(np.argmax(batch, axis=1))
    else:
        y = batch_y

    print("predicting")
    test_pred_raw = model.predict(validate)
    print("predicted")
    validate.epoch_data[epoch] = []

    # reset validation generator will be 1 epoch ahead
    validate.use_previous_epoch = None
    validate.cur_epoch -= 1
    del validate.epoch_data[-1]
    test_pred = np.argmax(test_pred_raw, axis=1)

    cm = confusion_matrix(y, test_pred)

    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=validate.labels)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with writer.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)
    print("done confusion")


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
