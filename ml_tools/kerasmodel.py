import psutil
import joblib
import itertools
import io
import time
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import os
import numpy as np
import gc
import time
import matplotlib.pyplot as plt
import json
import logging

from sklearn.metrics import confusion_matrix
import cv2
from ml_tools import tools
from ml_tools.datagenerator import DataGenerator
from ml_tools.preprocess import preprocess_movement, preprocess_frame, preprocess_ir
from ml_tools.interpreter import Interpreter
from classify.trackprediction import TrackPrediction
from ml_tools.hyperparams import HyperParams
from ml_tools.thermaldataset import get_resampled as get_thermal_dataset
from ml_tools.thermaldataset import get_distribution
from ml_tools.irdataset import get_resampled as get_ir_dataset


class KerasModel(Interpreter):
    """Defines a deep learning model"""

    VERSION = 1

    def __init__(self, train_config=None, labels=None):
        self.model = None
        self.datasets = None
        # dictionary containing current hyper parameters
        self.params = HyperParams()
        self.type = None

        if train_config:
            self.log_base = os.path.join(train_config.train_dir, "logs")
            self.log_dir = self.log_base
            self.checkpoint_folder = os.path.join(train_config.train_dir, "checkpoints")
            self.params.update(train_config.hyper_params)
            self.type = train_config.type
        self.labels = labels
        self.preprocess_fn = None
        self.validate = None
        self.train = None
        self.test = None
        self.mapped_labels = None
        self.label_probabilities = None
        self.class_weights = None

    def load_training_meta(self, base_dir):
        file = f"{base_dir}/training-meta.json"
        logging.info("loading meta %s", file)
        with open(file, "r") as f:
            meta = json.load(f)
        self.labels = meta.get("labels", [])
        self.type = meta.get("type", "thermal")

    def shape(self):
        if self.model is None:
            return None
        return self.model.get_config()["layers"][0]["config"]["batch_input_shape"]

    def get_base_model(self, input_shape, weights="imagenet"):
        pretrained_model = self.params.model_name
        if pretrained_model == "resnet":

            return (
                tf.keras.applications.ResNet50(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif pretrained_model == "resnetv2":
            return (
                tf.keras.applications.ResNet50V2(
                    weights=weights, include_top=False, input_shape=input_shape
                ),
                tf.keras.applications.resnet_v2.preprocess_input,
            )
        elif pretrained_model == "resnet152":
            return (
                tf.keras.applications.ResNet152(
                    weights=weights, include_top=False, input_shape=input_shape
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif pretrained_model == "vgg16":
            return (
                tf.keras.applications.VGG16(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.vgg16.preprocess_input,
            )
        elif pretrained_model == "vgg19":
            return (
                tf.keras.applications.VGG19(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.vgg19.preprocess_input,
            )
        elif pretrained_model == "mobilenet":
            return (
                tf.keras.applications.MobileNetV2(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.mobilenet_v2.preprocess_input,
            )
        elif pretrained_model == "densenet121":
            return (
                tf.keras.applications.DenseNet121(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.densenet.preprocess_input,
            )
        elif pretrained_model == "inceptionresnetv2":
            return (
                tf.keras.applications.InceptionResNetV2(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.inception_resnet_v2.preprocess_input,
            )
        elif pretrained_model == "inceptionv3":
            return (
                tf.keras.applications.InceptionV3(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.inception_v3.preprocess_input,
            )
        elif pretrained_model == "efficientnetb5":
            return (
                tf.keras.applications.EfficientNetB5(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                None,
            )
        elif pretrained_model == "efficientnetb0":
            return (
                tf.keras.applications.EfficientNetB0(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                None,
            )
        elif pretrained_model == "efficientnetb1":
            return (
                tf.keras.applications.EfficientNetB1(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                None,
            )
        raise Exception("Could not find model" + pretrained_model)

    def get_preprocess_fn(self):
        pretrained_model = self.params.model_name
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
        logging.warn(
            "pretrained model %s has no preprocessing function", pretrained_model
        )
        return None

    def build_model(self, dense_sizes=None, retrain_from=None, dropout=None):

        # width = self.params.frame_size
        width = self.params.output_dim[0]
        inputs = tf.keras.Input(shape=(width, width, 3), name="input")
        weights = None if self.params.base_training else "imagenet"
        base_model, preprocess = self.get_base_model((width, width, 3), weights=weights)
        self.preprocess_fn = preprocess
        x = base_model(inputs, training=self.params.base_training)

        if self.params.lstm:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            for i in dense_sizes:
                x = tf.keras.layers.Dense(i, activation="relu")(x)
            # gp not sure how many should be pre lstm, and how many post
            cnn = tf.keras.models.Model(inputs, outputs=x)

            self.model = self.add_lstm(cnn)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            if self.params.mvm:
                mvm_inputs = Input((25, 9))
                inputs = [inputs, mvm_inputs]

                mvm_features = Flatten()(mvm_inputs)
                x = Concatenate()([x, mvm_features])
                x = tf.keras.layers.Dense(2048, activation="relu")(x)
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
        self.model.compile(
            optimizer=optimizer(self.params),
            loss=loss(self.params),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
            ],
        )

    def load_weights(self, model_path, meta=True, training=False):
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
        if not training:
            self.model.trainable = False
        # self.model.summary()
        # self.model.load_weights(dir + "/variables/variables")

    def load_model(self, model_path, training=False, weights=None):
        logging.info("Loading %s with weight %s", model_path, weights)
        dir = os.path.dirname(model_path)
        self.model = tf.keras.models.load_model(dir)
        self.model.trainable = training
        self.load_meta(dir)
        if weights is not None:
            self.model.load_weights(weights).expect_partial()
        logging.info("Loaded weight %s", weights)

    def load_meta(self, dir):
        meta = json.load(open(os.path.join(dir, "metadata.txt"), "r"))
        self.params = HyperParams()
        self.params.update(meta["hyperparams"])
        self.labels = meta["labels"]
        self.mapped_labels = meta.get("mapped_labels")
        self.label_probabilities = meta.get("label_probabilities")
        self.preprocess_fn = self.get_preprocess_fn()
        self.type = meta.get("type", "thermal")

        logging.debug(
            "using types r %s g %s b %s type %s",
            self.params.red_type,
            self.params.green_type,
            self.params.blue_type,
            self.type,
        )

    def save(self, run_name=None, history=None, test_results=None):
        # create a save point
        if run_name is None:
            run_name = self.params.model_name
        self.model.save(os.path.join(self.checkpoint_folder, run_name))
        self.save_metadata(run_name, history, test_results)

    def save_metadata(self, run_name=None, history=None, test_results=None):
        #  save metadata
        if run_name is None:
            run_name = self.params.model_name
        model_stats = {}
        model_stats["name"] = self.params.model_name
        model_stats["labels"] = self.labels
        model_stats["hyperparams"] = self.params
        model_stats["training_date"] = str(time.time())
        model_stats["version"] = self.VERSION
        model_stats["mapped_labels"] = self.mapped_labels
        model_stats["label_probabilities"] = self.label_probabilities
        model_stats["type"] = self.type
        if self.class_weights is not None:
            model_stats["class_weights"] = self.class_weights

        if history:
            json_history = {}
            for key, item in history.items():
                if isinstance(item, list) and isinstance(item[0], np.floating):
                    json_history[key] = [float(i) for i in item]
                else:
                    json_history[key] = item
            model_stats["history"] = json_history
        if test_results:
            model_stats["test_loss"] = test_results[0]
            model_stats["test_acc"] = test_results[1]
        run_dir = os.path.join(self.checkpoint_folder, run_name)
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        json.dump(
            model_stats,
            open(
                os.path.join(run_dir, "metadata.txt"),
                "w",
            ),
            indent=4,
        )
        best_loss = os.path.join(run_dir, "val_loss")
        if os.path.exists(best_loss):
            json.dump(
                model_stats,
                open(
                    os.path.join(best_loss, "metadata.txt"),
                    "w",
                ),
                indent=4,
            )
        best_acc = os.path.join(run_dir, "val_acc")
        if os.path.exists(best_acc):
            json.dump(
                model_stats,
                open(
                    os.path.join(best_acc, "metadata.txt"),
                    "w",
                ),
                indent=4,
            )

    def close(self):
        # if self.test:
        #     self.test.stop_load()
        # if self.validate:
        #     self.validate.stop_load()
        # if self.train:
        #     self.train.stop_load()

        self.validate = None
        self.test = None
        self.train = None
        self.model = None
        tf.keras.backend.clear_session()
        gc.collect()
        del self.model
        del self.train
        del self.validate
        del self.test
        gc.collect()

    def get_dataset(
        self,
        pattern,
        augment=False,
        reshuffle=True,
        deterministic=False,
        resample=True,
        weights=None,
        stop_on_empty_dataset=True,
    ):
        logging.info("Getting dataset %s", self.type)
        if self.type == "thermal":
            return get_thermal_dataset(
                pattern,
                self.params.batch_size,
                self.params.output_dim[:2],
                self.labels,
                augment=augment,
                reshuffle=reshuffle,
                deterministic=deterministic,
                resample=resample,
                weights=weights,
                stop_on_empty_dataset=stop_on_empty_dataset,
            )
        return get_ir_dataset(
            pattern,
            self.params.batch_size,
            self.params.output_dim[:2],
            len(self.labels),
            augment=augment,
            reshuffle=reshuffle,
            deterministic=deterministic,
            resample=resample,
        )

    def train_model_tfrecords(
        self, epochs, run_name, base_dir, weights=None, rebalance=False, resample=False
    ):
        logging.info(
            "%s Training model for %s epochs with weights %s", run_name, epochs, weights
        )

        os.makedirs(self.log_base, exist_ok=True)
        self.log_dir = os.path.join(self.log_base, run_name)
        os.makedirs(self.log_base, exist_ok=True)

        if not self.model:
            self.build_model(
                dense_sizes=self.params.dense_sizes,
                retrain_from=self.params.retrain_layer,
                dropout=self.params.dropout,
            )
        self.model.summary()
        if weights is not None:
            self.model.load_weights(weights)
        base_dir = os.path.join(base_dir, "training-data")
        train_files = base_dir + "/train"
        validate_files = base_dir + "/validation"
        self.train = self.get_dataset(
            train_files, augment=True, resample=resample, stop_on_empty_dataset=False
        )
        self.validate = self.get_dataset(
            validate_files,
            augment=False,
            resample=resample,
            stop_on_empty_dataset=False,
        )
        distribution = get_distribution(self.train)
        if rebalance:
            self.class_weights = {}
            total = np.sum(distribution)
            multiplier = total / len(self.labels)
            for index, label in enumerate(self.labels):
                if distribution[index] == 0:
                    self.class_weights[index] = 0
                elif label == "bird":
                    self.class_weights[index] = (
                        1 / distribution[index] * (total / len(self.labels)) * 1.2
                    )
                elif label == "wallaby":
                    # wallabies not so important better to predict birds
                    self.class_weights[index] = (
                        1 / distribution[index] * (total / len(self.labels)) * 0.8
                    )
                else:
                    self.class_weights[index] = 1 / distribution[index] * multiplier
        logging.info(
            "%s Dist is %s giving weights %s",
            self.labels,
            distribution,
            self.class_weights,
        )

        self.save_metadata(run_name)

        checkpoints = self.checkpoints(run_name)

        history = self.model.fit(
            self.train,
            validation_data=self.validate,
            epochs=epochs,
            shuffle=False,
            class_weight=self.class_weights,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    self.log_dir, write_graph=True, write_images=True
                ),
                *checkpoints,
            ],  # log metricslast_stats
        )
        history = history.history
        test_accuracy = None
        test_files = base_dir + "/test"
        if len(test_files) > 0:
            self.test = self.get_dataset(
                test_files, augment=False, reshuffle=False, resample=False
            )
            if self.test:
                test_accuracy = self.model.evaluate(self.validate)

        self.save(run_name, history=history, test_results=test_accuracy)

    def train_model(self, epochs, run_name, weights=None):
        logging.info(
            "%s Training model for %s epochs with weights %s", run_name, epochs, weights
        )

        os.makedirs(self.log_base, exist_ok=True)
        self.log_dir = os.path.join(self.log_base, run_name)
        os.makedirs(self.log_base, exist_ok=True)
        if not self.model:
            self.build_model(
                dense_sizes=self.params.dense_sizes,
                retrain_from=self.params.retrain_layer,
                dropout=self.params.dropout,
            )
        self.model.summary()
        self.save()
        if weights is not None:
            self.model.load_weights(weights)
        self.train = DataGenerator(
            self.train_dataset,
            self.labels,
            self.params.output_dim,
            augment=True,
            cap_at="bird",
            epochs=epochs,
            model_preprocess=self.preprocess_fn,
            preload=True,
            **self.params,
        )
        self.validate = DataGenerator(
            self.validation_dataset,
            self.labels,
            self.params.output_dim,
            cap_at="bird",
            model_preprocess=self.preprocess_fn,
            epochs=epochs,
            preload=True,
            **self.params,
        )

        self.save_metadata(run_name)

        weight_for_0 = 1
        weight_for_1 = 1 / 4
        class_weight = {}
        for i, label in enumerate(self.labels):
            if label == "bird":
                class_weight[i] = 1.6
            elif label == "wallaby":
                # wallabies not so important better to predict birds
                class_weight[i] = 0.6
            else:
                class_weight[i] = 1
        logging.info("training with class wieghts %s", class_weight)
        # give a bit of time for preloader to cache data
        checkpoints = self.checkpoints(run_name)
        history = self.model.fit(
            self.train,
            validation_data=self.validate,
            epochs=epochs,
            shuffle=False,
            class_weight=class_weight,
            callbacks=[
                ClearMemory(),
                *checkpoints,
            ],  # log metricslast_stats
        )
        history = history.history
        self.train.stop_load()
        self.validate.stop_load()
        test_accuracy = None
        if self.test_dataset and self.test_dataset.has_data():
            self.test = DataGenerator(
                self.test_dataset,
                self.train_dataset.labels,
                self.params.output_dim,
                model_preprocess=self.preprocess_fn,
                epochs=1,
                cap_at="bird",
                preload=True,
                **self.params,
            )
            logging.info("Evaluating test %s", len(self.test))
            test_accuracy = self.model.evaluate(self.test)
            logging.info("Test accuracy is %s", test_accuracy)
            self.test.stop_load()
        self.save(run_name, history=history, test_results=test_accuracy)

    def checkpoints(self, run_name):
        val_loss = os.path.join(self.checkpoint_folder, run_name, "val_loss")

        checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(
            val_loss,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
        )
        val_acc = os.path.join(self.checkpoint_folder, run_name, "val_acc")

        checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
            val_acc,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )

        val_precision = os.path.join(self.checkpoint_folder, run_name, "val_recall")

        checkpoint_recall = tf.keras.callbacks.ModelCheckpoint(
            val_precision,
            monitor="val_recall",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )
        earlyStopping = tf.keras.callbacks.EarlyStopping(
            patience=22, monitor="val_accuracy"
        )
        # havent found much use in this just takes training time
        # file_writer_cm = tf.summary.create_file_writer(
        #     self.log_base + "/{}/cm".format(run_name)
        # )
        # cm_callback = keras.callbacks.LambdaCallback(
        #     on_epoch_end=lambda epoch, logs: log_confusion_matrix(
        #         epoch, logs, self.model, self.test, file_writer_cm
        #     )
        # )
        #         "lr_callback": {
        #   "monitor": "val_categorical_accuracy",
        #   "mode": "max",
        #   "factor": 0.65,
        #   "patience": 15,
        #   "min_lr": 0.00002,
        #   "verbose": 1
        # },
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", verbose=1
        )
        return [earlyStopping, checkpoint_acc, checkpoint_loss, reduce_lr_callback]

    def regroup(self, shuffle=True):
        # can use this to put animals into groups i.e. wallaby vs not
        if not self.mapped_labels:
            logging.warn("Cant regroup without specifying mapped_labels")
            return
        for dataset in self.datasets.values():
            dataset.regroup(self.mapped_labels, shuffle=shuffle)
            dataset.labels.sort()
        self.labels = self.train_dataset.labels.copy()

    def import_dataset(self, base_dir, ignore_labels=None, lbl_p=None):
        """
        Import dataset.
        :param dataset_filename: path and filename of the dataset
        :param ignore_labels: (optional) these labels will be removed from the dataset.
        :param lbl_p: (optional) probably for each label
        :return:
        """
        self.label_probabilities = lbl_p
        datasets = ["train", "validation", "test"]
        self.datasets = {}
        for i, name in enumerate(datasets):
            self.datasets[name] = joblib.load(
                open(f"{os.path.join(base_dir, name)}.dat", "rb")
            )

        for dataset in self.datasets.values():
            dataset.labels.sort()
            dataset.set_read_only(True)
            dataset.lbl_p = lbl_p
            dataset.use_segments = self.params.use_segments

            if ignore_labels:
                for label in ignore_labels:
                    dataset.remove_label(label)
        self.labels = self.train_dataset.labels

        if self.mapped_labels:
            self.regroup()

    @property
    def test_dataset(self):
        return self.datasets["test"]

    @property
    def validation_dataset(self):
        return self.datasets["validation"]

    @property
    def train_dataset(self):
        return self.datasets["train"]

    @property
    def hyperparams_string(self):
        """Returns list of hyperparameters as a string."""
        return "\n".join(
            ["{}={}".format(param, value) for param, value in self.params.items()]
        )

    def add_lstm(self, cnn):
        input_layer = tf.keras.Input(shape=(None, *self.params.output_dim))
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

    def classify_thermal_track(self, clip, track, keep_all=True, segment_frames=None):
        track_data = {}
        thermal_median = np.empty(len(track.bounds_history), dtype=np.uint16)
        for i, region in enumerate(track.bounds_history):
            frame = clip.frame_buffer.get_frame(region.frame_number)
            if frame is None:
                logging.error(
                    "Clasifying clip %s track %s can't get frame %s",
                    clip.get_id(),
                    track.get_id(),
                    region.frame_number,
                )
                raise Exception(
                    "Clasifying clip {} track {} can't get frame {}".format(
                        clip.get_id(), track.get_id(), region.frame_number
                    )
                )
            cropped_frame = frame.crop_by_region(region)
            thermal_median[i] = np.median(frame.thermal)
            cropped_frame.resize_with_aspect(
                (self.params.frame_size, self.params.frame_size),
                clip.crop_rectangle,
                True,
            )
            track_data[frame.frame_number] = cropped_frame

        segments = track.get_segments(
            clip.ffc_frames,
            thermal_median,
            self.params.square_width ** 2,
            repeats=4,
            segment_frames=segment_frames,
        )
        return self.classify_track_data(
            track.get_id(),
            track_data,
            segments,
        )

    def classify_track(self, clip, track, keep_all=True, segment_frames=None):
        logging.debug("Classifying track %s", self.type)
        if self.type == "IR":
            return self.classify_ir(clip, track)
        return self.classify_thermal_track(clip, track, keep_all, segment_frames)

    def classify_ir(self, clip, track, keep_all=True, segment_frames=None):
        data = []
        crop = True
        thermal_median = np.empty(len(track.bounds_history), dtype=np.uint16)
        for i, region in enumerate(track.bounds_history):
            frame = clip.frame_buffer.get_frame(region.frame_number)
            if frame is None:
                logging.error(
                    "Clasifying clip %s track %s can't get frame %s",
                    clip.get_id(),
                    track.get_id(),
                    region.frame_number,
                )
                raise Exception(
                    "Clasifying clip {} track {} can't get frame {}".format(
                        clip.get_id(), track.get_id(), region.frame_number
                    )
                )

            preprocessed = preprocess_ir(
                frame.copy(),
                (
                    self.params.frame_size,
                    self.params.frame_size,
                ),
                crop,
                region,
                self.preprocess_fn,
                save_info=f"{region.frame_number} - {region}",
            )

            data.append(preprocessed)
        data = np.float32(data)
        output = self.model.predict(data)
        track_prediction = TrackPrediction(track.get_id(), self.labels)

        track_prediction.classified_clip(output, output, None)
        track_prediction.normalize_score()
        return track_prediction

    def classify_track_data(
        self,
        track_id,
        data,
        segments,
    ):
        track_prediction = TrackPrediction(track_id, self.labels)
        start = time.time()
        predictions = []
        smoothed_predictions = []
        for segment in segments:
            segment_frames = []
            median = np.zeros((len(segment.frame_indices)))
            for frame_i in segment.frame_indices:
                f = data[frame_i]
                segment_frames.append(f.copy())
            frames = preprocess_movement(
                segment_frames,
                self.params.square_width,
                self.params.frame_size,
                self.params.red_type,
                self.params.green_type,
                self.params.blue_type,
                self.preprocess_fn,
                reference_level=segment.frame_temp_median,
                keep_edge=self.params.keep_edge,
            )
            if frames is None:
                logging.warn("No frames to predict on")
                continue
            output = self.model.predict(frames[np.newaxis, :])

            track_prediction.classified_frames(
                segment.frame_indices, output[0], max(1, segment.mass)
            )
        track_prediction.classify_time = time.time() - start
        track_prediction.normalize_score()
        return track_prediction

    def predict(self, frame):
        return self.model.predict(frame[np.newaxis, :])[0]

    def classify_frame(self, frame, thermal_median, preprocess=True):
        if preprocess:
            frame = preprocess_frame(
                frame,
                False,
                thermal_median,
                0,
                self.params.output_dim,
                preprocess_fn=self.preprocess_fn,
            )
            if frame is None:
                return None
        output = self.model.predict(frame[np.newaxis, :])
        return output[0]

    def confusion_tfrecords(self, dataset, filename):
        true_categories = tf.concat([y for x, y in dataset], axis=0)
        true_categories = np.int64(tf.argmax(true_categories, axis=1))
        y_pred = self.model.predict(dataset)

        predicted_categories = np.int64(tf.argmax(y_pred, axis=1))

        cm = confusion_matrix(
            true_categories, predicted_categories, labels=np.arange(len(self.labels))
        )
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=self.labels)
        plt.savefig(filename, format="png")

    # Obselete
    def confusion(self, dataset, filename="confusion.png"):
        dataset.set_read_only(True)
        dataset.use_segments = self.params.use_segments
        test = DataGenerator(
            dataset,
            self.labels,
            self.params.output_dim,
            batch_size=self.params.batch_size,
            channel=self.params.channel,
            use_movement=self.params.use_movement,
            shuffle=True,
            model_preprocess=self.preprocess_fn,
            epochs=1,
            keep_epoch=True,
            cap_samples=True,
            cap_at="bird",
            square_width=self.params.square_width,
            type=self.params.type,
            segment_type=self.params.segment_type,
            keep_edge=self.params.keep_edge,
            preload=True,
        )
        test_pred_raw = self.model.predict(test)
        test.stop_load()
        test_pred = np.argmax(test_pred_raw, axis=1)

        batch_y = test.get_epoch_labels(0)
        for i in range(len(batch_y)):
            mapped_label = dataset.mapped_label(batch_y[i])
            batch_y[i] = self.labels.index(mapped_label)
        batch_y = np.int32(batch_y)
        self.f1(batch_y, test_pred_raw)
        # test.epoch_data = None
        cm = confusion_matrix(batch_y, test_pred, labels=np.arange(len(self.labels)))
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=self.labels)
        plt.savefig(filename, format="png")

    # Obsolete
    def f1(self, batch_y, pred_raw):
        import tensorflow_addons as tfa

        one_hot_y = tf.keras.utils.to_categorical(batch_y, num_classes=len(self.labels))
        metric = tfa.metrics.F1Score(num_classes=len(self.labels))
        metric.update_state(one_hot_y, pred_raw)
        result = metric.result().numpy()
        logging.info("F1 score")
        by_label = {}
        for i, label in enumerate(self.labels):
            by_label[label] = round(100 * result[i])
        sorted = self.labels.copy()
        sorted.sort()
        for label in sorted:
            logging.info("%s = %s", label, by_label[label])

    def evaluate(self, dataset):
        test_accuracy = self.model.evaluate(dataset)
        logging.info("Test accuracy is %s", test_accuracy)

    # needs to be updated to work with tfrecord datagen
    def track_accuracy(self, dataset, confusion="confusion.png"):
        dataset.set_read_only(True)
        dataset.use_segments = self.params.use_segments
        predictions = []
        actual = []
        raw_predictions = []
        total = 0
        correct = 0
        samples_by_label = {}
        incorrect_labels = {}
        for sample in dataset.segments:
            label_samples = samples_by_label.setdefault(sample.label, {})
            if sample.track_id in label_samples:
                label_samples[sample.track_id].append(sample)
            else:
                label_samples[sample.track_id] = [sample]
        bird_tracks = len(label_samples.get("bird", []))

        for label in dataset.label_mapping.keys():
            incorrect = {}
            incorrect_labels[label] = incorrect
            track_samples = samples_by_label.get(label)
            if not track_samples:
                logging.warn("No samples for %s", label)
                continue
            track_samples = track_samples.values()
            if label == "insect" or label == "false-positive":
                track_samples = np.random.choice(
                    list(track_samples),
                    min(len(track_samples), bird_tracks),
                    replace=False,
                )
            logging.info("taking %s tracks for %s", len(track_samples), label)
            mapped_label = dataset.mapped_label(label)
            for track_segments in track_samples:
                segment_db = dataset.numpy_data.load_segments(track_segments)
                frame_db = {}
                for frames in segment_db.values():
                    for f in frames:
                        frame_db[f.frame_number] = f
                track_prediction = self.classify_track_data(
                    track_segments[0].track_id,
                    frame_db,
                    segments=track_segments,
                )

                total += 1
                if track_prediction is None or len(track_prediction.predictions) == 0:
                    logging.warn("No predictions for %s", track_segments[0].track_id)
                    continue
                avg = np.mean(track_prediction.predictions, axis=0)
                actual.append(self.labels.index(mapped_label))
                predictions.append(track_prediction.best_label_index)

                raw_predictions.append(avg)
                if actual[-1] == predictions[-1]:
                    correct += 1
                else:

                    if track_prediction.predicted_tag() in incorrect:
                        incorrect[track_prediction.predicted_tag()].append(
                            track_segments[0].unique_track_id
                        )
                    else:
                        incorrect[track_prediction.predicted_tag()] = [
                            track_segments[0].unique_track_id
                        ]

                if total % 50 == 0:
                    logging.info("Processed %s", total)
        for label, incorrect in incorrect_labels.items():
            logging.info("Incorrect ************ %s", label)
            logging.info(incorrect.get("false-positive"))
        logging.info("Predicted correctly %s", round(100 * correct / total))
        self.f1(actual, raw_predictions)

        if confusion is not None:
            cm = confusion_matrix(
                actual, predictions, labels=np.arange(len(self.labels))
            )
            figure = plot_confusion_matrix(cm, class_names=self.labels)
            plt.savefig(confusion, format="png")


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
    cm = np.nan_to_num(cm)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def log_confusion_matrix(epoch, logs, model, dataset, writer):
    # Use the model to predict the values from the validation dataset.

    dataset.reload_samples()
    test_pred_raw = model.predict(dataset)
    dataset.cur_epoch -= 1
    batch_y = dataset.get_epoch_labels(epoch=dataset.cur_epoch)

    y = []
    for batch in batch_y:
        y.extend(np.argmax(batch, axis=1))

    # reset validation generator will be 1 epoch ahead
    test_pred = np.argmax(test_pred_raw, axis=1)

    cm = confusion_matrix(y, test_pred, labels=np.arange(len(dataset.labels)))

    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=dataset.labels)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with writer.as_default():
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


def loss(params):
    softmax = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=params.label_smoothing,
    )
    return softmax


def optimizer(params):
    if params.learning_rate_decay != 1.0:
        learning_rate = tf.compat.v1.train.exponential_decay(
            self.params.learning_rate,
            self.global_step,
            1000,
            self.params["learning_rate_decay"],
            staircase=True,
        )
        tf.compat.v1.summary.scalar("params/learning_rate", learning_rate)
    else:
        learning_rate = params.learning_rate  # setup optimizer
    if learning_rate:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return optimizer


def validate_model(model_file):
    path, ext = os.path.splitext(model_file)
    if not os.path.exists(model_file):
        return False
    return True


# HYPER PARAM TRAINING OF A MODEL
#
HP_DENSE_SIZES = hp.HParam("dense_sizes", hp.Discrete([""]))
HP_TYPE = hp.HParam("type", hp.Discrete([1]))

HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([32]))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam"]))
HP_LEARNING_RATE = hp.HParam("learning_rate", hp.Discrete([0.01]))
HP_EPSILON = hp.HParam("epislon", hp.Discrete([1e-7]))  # 1.0 and 0.1 for inception
HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.0]))
HP_RETRAIN = hp.HParam("retrain_layer", hp.Discrete([-1]))
HP_SEGMENT_TYPE = hp.HParam("segment_type", hp.Discrete([0, 1, 2, 3, 4, 5, 6]))

METRIC_ACCURACY = "accuracy"
METRIC_LOSS = "loss"


# GRID SEARCH
def train_test_model(model, hparams, params, log_dir, writer, epochs=15):
    # if not self.model:
    learning_rate = hparams[HP_SEGMENT_TYPE]
    # note gp this means we need to keep tracks in dataset file in build step
    keras_model.train_dataset.recalculate_segments(segment_type=segment_type)
    keras_model.validation_dataset.recalculate_segments(segment_type=segment_type)
    keras_model.test_dataset.recalculate_segments(segment_type=segment_type)
    labels = keras_model.train_dataset.labels
    train = DataGenerator(
        keras_model.train_dataset,
        labels,
        params.output_dim,
        batch_size=batch_size,
        model_preprocess=preprocess_fn,
        epochs=epochs,
        shuffle=True,
        cap_at="bird",
        type=type,
        **params,
    )
    validate = DataGenerator(
        keras_model.validation_dataset,
        labels,
        params.output_dim,
        batch_size=batch_size,
        model_preprocess=preprocess_fn,
        epochs=epochs,
        shuffle=True,
        cap_at="bird",
        type=type,
        **params,
    )
    test = DataGenerator(
        keras_model.test_dataset,
        labels,
        params.output_dim,
        batch_size=batch_size,
        model_preprocess=preprocess_fn,
        epochs=1,
        shuffle=True,
        cap_at="bird",
        type=type,
        **params,
    )
    opt = None
    learning_rate = hparams[HP_LEARNING_RATE]
    epsilon = hparams[HP_EPSILON]

    if hparams[HP_OPTIMIZER] == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, epsilon=epsilon)
    model.compile(
        optimizer=opt, loss=loss(params), metrics=["accuracy"], run_eagerly=True
    )
    cm_callback = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: log_confusion_matrix(
            epoch, logs, model, test, writer
        )
    )
    history = model.fit(
        train,
        epochs=epochs,
        shuffle=False,
        validation_data=validate,
        callbacks=[cm_callback],
        verbose=2,
    )
    train.stop_load()
    validate.stop_load()
    test.stop_load()
    validate = None
    test = None
    train = None
    tf.keras.backend.clear_session()
    gc.collect()
    del model
    del train
    del validate
    del test
    gc.collect()
    return history


def test_hparams(keras_model, log_dir):

    epochs = 15
    batch_size = 32

    dir = log_dir + "/hparam_tuning"
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
                HP_SEGMENT_TYPE,
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
            for retrain_layer in HP_RETRAIN.domain.values:
                for learning_rate in HP_LEARNING_RATE.domain.values:
                    for type in HP_TYPE.domain.values:
                        for optimizer in HP_OPTIMIZER.domain.values:
                            for epsilon in HP_EPSILON.domain.values:
                                for dropout in HP_DROPOUT.domain.values:
                                    for segment_type in HP_SEGMENT_TYPE.domain.values:
                                        hparams = {
                                            HP_DENSE_SIZES: dense_size,
                                            HP_BATCH_SIZE: batch_size,
                                            HP_LEARNING_RATE: learning_rate,
                                            HP_OPTIMIZER: optimizer,
                                            HP_EPSILON: epsilon,
                                            HP_TYPE: type,
                                            HP_RETRAIN: retrain_layer,
                                            HP_DROPOUT: dropout,
                                            HP_SEGMENT_TYPE: segment_type,
                                        }

                                        dense_layers = []
                                        if dense_size != "":
                                            for i, size in enumerate(dense_size):
                                                dense_layers[i] = int(size)
                                        keras_model.build_model(
                                            dense_sizes=dense_layers,
                                            retrain_from=None
                                            if retrain_layer == -1
                                            else retrain_layer,
                                            dropout=None if dropout == 0.0 else dropout,
                                        )

                                        run_name = "run-%d" % session_num
                                        print("--- Starting trial: %s" % run_name)
                                        print({h.name: hparams[h] for h in hparams})
                                        self.run(
                                            keras_model,
                                            dir + "/" + run_name,
                                            hparams,
                                            epochs,
                                        )
                                        session_num += 1


def run(keras_model, log_dir, hparams, params, epochs):
    with tf.summary.create_file_writer(log_dir).as_default() as w:
        hp.hparams(hparams)  # record the values used in this trial
        history = train_test_model(
            keras_model, hparams, params, log_dir, w, epochs=epochs
        )
        val_accuracy = history.history["val_accuracy"]
        val_loss = history.history["val_loss"]
        # log_confusion_matrix(epochs, None, self.model, self.validate, None)

        for step, accuracy in enumerate(val_accuracy):
            loss = val_loss[step]
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=step)
            tf.summary.scalar(METRIC_LOSS, loss, step=step)


from tensorflow.keras.callbacks import Callback


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("epoch edned", epoch)
        gc.collect()
        tf.keras.backend.clear_session()
