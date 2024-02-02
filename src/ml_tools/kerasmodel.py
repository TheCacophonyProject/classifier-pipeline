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
from pathlib import Path

from sklearn.metrics import confusion_matrix
import cv2
from ml_tools import tools
from ml_tools.datasetstructures import SegmentType

from ml_tools.preprocess import (
    preprocess_movement,
    preprocess_frame,
)
from ml_tools.interpreter import Interpreter
from classify.trackprediction import TrackPrediction
from ml_tools.hyperparams import HyperParams
from ml_tools import thermaldataset
from ml_tools.resnet.wr_resnet import WRResNet

from ml_tools import irdataset
from ml_tools.tfdataset import get_weighting, get_distribution, get_dataset as get_tf
import tensorflow_decision_forests as tfdf
from ml_tools import forestmodel
from ml_tools.preprocess import FrameTypes

classify_i = 0


class KerasModel(Interpreter):
    """Defines a deep learning model"""

    VERSION = 1
    TYPE = "Keras"

    def __init__(self, train_config=None, labels=None, data_dir=None):
        self.model = None
        self.datasets = None
        self.remapped = None
        # dictionary containing current hyper parameters
        self.params = HyperParams()
        self.data_type = None
        self.data_dir = data_dir
        if train_config:
            self.log_base = train_config.train_dir / "logs"
            self.log_dir = self.log_base
            self.checkpoint_folder = train_config.train_dir / "checkpoints"
            self.params.update(train_config.hyper_params)
            self.data_type = train_config.type
        self.labels = labels
        self.preprocess_fn = None
        self.validate = None
        self.train = None
        self.test = None
        self.mapped_labels = None
        self.label_probabilities = None
        self.class_weights = None
        self.ds_by_label = True
        self.excluded_labels = []
        self.remapped_labels = []
        self.orig_labels = None

    def load_training_meta(self, base_dir):
        file = f"{base_dir}/training-meta.json"
        logging.info("loading meta %s", file)
        with open(file, "r") as f:
            meta = json.load(f)
        self.labels = meta.get("labels", [])
        self.data_type = meta.get("type", "thermal")
        self.dataset_counts = meta.get("counts")
        self.ds_by_label = meta.get("by_label", True)
        self.excluded_labels = meta.get("excluded_labels")
        self.remapped_labels = meta.get("remapped_labels")

    def shape(self):
        if self.model is None:
            return None
        inputs = self.model.inputs
        shape = []
        for input in inputs:
            shape.append(tuple(input.shape.as_list()))
        return len(shape), shape

    def get_base_model(self, input, weights="imagenet"):
        pretrained_model = self.params.model_name
        if pretrained_model == "resnet":
            return (
                tf.keras.applications.ResNet50(
                    weights=weights,
                    include_top=False,
                    input_tensor=input,
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif pretrained_model == "resnetv2":
            return (
                tf.keras.applications.ResNet50V2(
                    weights=weights, include_top=False, input_tensor=input
                ),
                tf.keras.applications.resnet_v2.preprocess_input,
            )
        elif pretrained_model == "resnet152":
            return (
                tf.keras.applications.ResNet152(
                    weights=weights, include_top=False, input_tensor=input
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif pretrained_model == "vgg16":
            return (
                tf.keras.applications.VGG16(
                    weights=weights,
                    include_top=False,
                    input_tensor=input,
                ),
                tf.keras.applications.vgg16.preprocess_input,
            )
        elif pretrained_model == "vgg19":
            return (
                tf.keras.applications.VGG19(
                    weights=weights,
                    include_top=False,
                    input_tensor=input,
                ),
                tf.keras.applications.vgg19.preprocess_input,
            )
        elif pretrained_model == "mobilenet":
            return (
                tf.keras.applications.MobileNetV2(
                    weights=weights,
                    include_top=False,
                    input_tensor=input,
                ),
                tf.keras.applications.mobilenet_v2.preprocess_input,
            )
        elif pretrained_model == "densenet121":
            return (
                tf.keras.applications.DenseNet121(
                    weights=weights,
                    include_top=False,
                    input_tensor=input,
                ),
                tf.keras.applications.densenet.preprocess_input,
            )
        elif pretrained_model == "inceptionresnetv2":
            return (
                tf.keras.applications.InceptionResNetV2(
                    weights=weights,
                    include_top=False,
                    input_tensor=input,
                ),
                tf.keras.applications.inception_resnet_v2.preprocess_input,
            )
        elif pretrained_model == "inceptionv3":
            print("Input", input)
            return (
                tf.keras.applications.InceptionV3(
                    weights=weights,
                    include_top=False,
                    input_tensor=input,
                ),
                tf.keras.applications.inception_v3.preprocess_input,
            )
        elif pretrained_model == "efficientnetb5":
            return (
                tf.keras.applications.EfficientNetB5(
                    weights=weights,
                    include_top=False,
                    input_tensor=input,
                ),
                None,
            )
        elif pretrained_model == "efficientnetb0":
            return (
                tf.keras.applications.EfficientNetB0(
                    weights=weights,
                    include_top=False,
                    input_tensor=input,
                ),
                None,
            )
        elif pretrained_model == "efficientnetb1":
            return (
                tf.keras.applications.EfficientNetB1(
                    weights=weights,
                    include_top=False,
                    input_tensor=input,
                ),
                None,
            )
        elif pretrained_model == "nasnet":
            return (
                tf.keras.applications.nasnet.NASNetLarge(
                    weights=weights, include_top=False, input_tensor=input
                ),
                tf.keras.applications.nasnet.preprocess_input,
            )
        elif pretrained_model == "wr-resnet":
            return (
                WRResNet(
                    input, k=self.params.get("k", 4), depth=self.params.get("depth", 22)
                ),
                None,
            )
        raise Exception("Could not find model " + pretrained_model)

    def get_forest_model(self, run_name):
        train_files = self.data_dir / "train"
        train, remapped, _, _ = get_dataset(
            train_files,
            self.data_type,
            self.orig_labels,
            batch_size=self.params.batch_size,
            image_size=self.params.output_dim[:2],
            preprocess_fn=self.preprocess_fn,
            augment=False,
            resample=False,
            stop_on_empty_dataset=False,
            only_features=True,
            one_hot=False,
            excluded_labels=self.excluded_labels,
            remapped_labels=self.remapped_labels,
        )
        # have to run fit firest
        rf = tfdf.keras.RandomForestModel()
        rf.fit(train)
        rf.save(str(self.checkpoint_folder / run_name / "rf"))
        save_metadata(self)
        return rf

    def build_model(
        self, dense_sizes=None, retrain_from=None, dropout=None, run_name=None
    ):
        # width = self.params.frame_size
        width = self.params.output_dim[0]
        inputs = tf.keras.Input(
            shape=(width, width, len(self.params.channels)), name="input"
        )
        weights = None if self.params.base_training else "imagenet"
        base_model, preprocess = self.get_base_model(inputs, weights=weights)
        self.preprocess_fn = preprocess

        # inputs = base_model.input
        x = base_model.output
        # x = base_model(inputs, training=self.params.base_training)
        if self.params.get("model_merge"):
            logging.info(
                "Loading cnn rf model %s %s",
                self.params.get("model_cnn"),
                self.params.get("model_rf"),
            )
            cnn = tf.keras.models.load_model(self.params.get("model_cnn"))
            cnn.load_weights(
                Path(self.params.get("model_cnn")) / "val_acc"
            ).expect_partial()
            feature_input = tf.keras.Input(shape=(188), name="feature_input")
            model_rf = tf.keras.models.load_model(self.params.get("model_rf"))
            rf = model_rf(feature_input)
            inputs = [cnn.input, feature_input]
            cnn.summary()
            model_rf.summary()
            print("Outputs", cnn.outputs, rf)
            x = tf.keras.layers.Concatenate()([cnn.outputs[0], rf])
            activation = "softmax"
            if self.params.multi_label:
                activation = "sigmoid"
            logging.info("Using %s activation", activation)
            preds = tf.keras.layers.Dense(
                len(self.labels), activation=activation, name="merged-prediction"
            )(x)
            self.model = tf.keras.models.Model(inputs, outputs=preds)
        elif self.params.lstm:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            for i in dense_sizes:
                x = tf.keras.layers.Dense(i, activation="relu")(x)
            # gp not sure how many should be pre lstm, and how many post
            cnn = tf.keras.models.Model(inputs, outputs=x)

            self.model = self.add_lstm(cnn)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            if self.params.mvm:
                mvm_inputs = tf.keras.layers.Input((188))
                inputs = [inputs, mvm_inputs]
                # mvm_features = tf.keras.layers.Flatten()(mvm_inputs)
                #
                # if self.params["hq_mvm"]:
                # print("HQ")
                if self.params.mvm_forest:
                    rf = self.get_forest_model(run_name)

                    rf = rf(mvm_inputs)
                    x = tf.keras.layers.Concatenate()([x, rf])

                else:
                    mvm_features = tf.keras.layers.Dense(128, activation="relu")(
                        mvm_inputs
                    )
                    mvm_features = tf.keras.layers.Dense(128, activation="relu")(
                        mvm_features
                    )
                    mvm_features = tf.keras.layers.Dropout(0.1)(mvm_features)

                    # else:
                    #     mvm_features = tf.keras.layers.Dense(32, activation="relu")(
                    #         mvm_inputs
                    #     )
                    x = tf.keras.layers.Concatenate()([x, mvm_features])
                # x = tf.keras.layers.Dense(1028, activation="relu")(x)
            if dense_sizes is not None:
                for i in dense_sizes:
                    x = tf.keras.layers.Dense(i, activation="relu")(x)
            if dropout:
                x = tf.keras.layers.Dropout(dropout)(x)

            activation = "softmax"
            if self.params.multi_label:
                activation = "sigmoid"
            logging.info("Using %s activation", activation)
            preds = tf.keras.layers.Dense(
                len(self.labels), activation=activation, name="prediction"
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

        if self.params.multi_label:
            acc = tf.metrics.binary_accuracy
        else:
            acc = tf.metrics.categorical_accuracy

        self.model.compile(
            optimizer=optimizer(self.params),
            loss=loss(self.params),
            metrics=[
                acc,
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
            ],
        )

    def load_model(self, model_file, training=False, weights=None):
        model_file = Path(model_file)
        super().__init__(model_file)
        logging.info("Loading %s with model weights %s", model_file, weights)
        if model_file.suffix == ".pb":
            self.model = tf.keras.models.load_model(model_file.parent)
        else:
            self.model = tf.keras.models.load_model(model_file)

        self.model.trainable = training

        if weights is not None:
            self.model.load_weights(weights).expect_partial()
            logging.info("Loaded weight %s", weights)
        # print(self.model.summary())

    def save(self, run_name=None, history=None, test_results=None):
        # create a save point
        if run_name is None:
            run_name = self.params.model_name

        self.model.save(str(self.checkpoint_folder / run_name / f"{run_name}.keras"))
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
        model_stats["type"] = self.data_type
        model_stats["remapped_labels"] = self.remapped_labels
        model_stats["excluded_labels"] = self.excluded_labels
        if self.remapped is not None:
            model_stats["remapped"] = self.remapped
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
        run_dir = self.checkpoint_folder / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if not run_dir.exists:
            run_dir.mkdir()

        json.dump(
            model_stats,
            (run_dir / f"{run_name}.json").open("w"),
            indent=4,
            cls=MetaJSONEncoder,
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

    def train_model(
        self, epochs, run_name, weights=None, rebalance=False, resample=False
    ):
        logging.info(
            "%s Training model for %s epochs with weights %s", run_name, epochs, weights
        )
        self.excluded_labels, self.remapped_labels = get_excluded(
            self.data_type, self.params.multi_label
        )
        train_files = self.data_dir / "train"
        validate_files = self.data_dir / "validation"
        logging.info(
            "Excluding %s remapping %s", self.excluded_labels, self.remapped_labels
        )

        if self.params.multi_label:
            self.labels.append("land-bird")
        self.orig_labels = self.labels.copy()
        for l in self.excluded_labels:
            if l in self.labels:
                self.labels.remove(l)
        for l in self.remapped_labels.keys():
            if l in self.labels:
                self.labels.remove(l)
        self.log_dir = self.log_base / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if not self.model:
            self.build_model(
                dense_sizes=self.params.dense_sizes,
                retrain_from=self.params.retrain_layer,
                dropout=self.params.dropout,
                run_name=run_name,
            )
        self.model.summary()

        self.train, remapped, new_labels, epoch_size = get_dataset(
            train_files,
            self.data_type,
            self.orig_labels,
            batch_size=self.params.batch_size,
            image_size=self.params.output_dim[:2],
            preprocess_fn=self.preprocess_fn,
            resample=resample,
            stop_on_empty_dataset=False,
            include_features=self.params.mvm,
            augment=True,
            excluded_labels=self.excluded_labels,
            remapped_labels=self.remapped_labels,
            # dist=self.dataset_counts["train"],
            multi_label=self.params.multi_label,
            num_frames=self.params.square_width**2,
            channels=self.params.channels,
        )
        self.remapped = remapped
        self.validate, remapped, _, _ = get_dataset(
            validate_files,
            self.data_type,
            self.orig_labels,
            batch_size=self.params.batch_size,
            image_size=self.params.output_dim[:2],
            preprocess_fn=self.preprocess_fn,
            resample=resample,
            stop_on_empty_dataset=False,
            include_features=self.params.mvm,
            excluded_labels=self.excluded_labels,
            remapped_labels=self.remapped_labels,
            multi_label=self.params.multi_label,
            num_frames=self.params.square_width**2,
            channels=self.params.channels,
            # dist=self.dataset_counts["validation"],
        )

        if weights is not None:
            self.model.load_weights(weights)
        if rebalance:
            self.class_weights = get_weighting(self.train, self.labels)
            logging.info(
                "Training on %s  with class weights %s",
                self.labels,
                self.class_weights,
            )

        self.save_metadata(run_name)
        self.save(run_name)
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
        test_files = self.data_dir / "test"

        if len(list(test_files.glob("*.tfrecord"))) > 0:
            self.test, _, _, _ = get_dataset(
                test_files,
                self.data_type,
                self.orig_labels,
                batch_size=self.params.batch_size,
                image_size=self.params.output_dim[:2],
                preprocess_fn=self.preprocess_fn,
                stop_on_empty_dataset=False,
                include_features=self.params.mvm,
                reshuffle=False,
                resample=False,
                excluded_labels=self.excluded_labels,
                remapped_labels=self.remapped_labels,
                multi_label=self.params.multi_label,
                num_frames=self.params.square_width**2,
                channels=self.params.channels,
            )
            if self.test:
                test_accuracy = self.model.evaluate(self.test)

        self.save(run_name, history=history, test_results=test_accuracy)

    def checkpoints(self, run_name):
        checkpoint_file = self.checkpoint_folder / run_name / "cp.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file, save_weights_only=True, verbose=1
        )
        val_loss = self.checkpoint_folder / run_name / "val_loss"

        checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(
            val_loss,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
        )
        val_acc = self.checkpoint_folder / run_name / "val_acc"

        checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
            val_acc,
            monitor=(
                "val_binary_accuracy"
                if self.params.multi_label
                else "val_categorical_accuracy"
            ),
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )

        val_precision = self.checkpoint_folder / run_name / "val_recall"

        checkpoint_recall = tf.keras.callbacks.ModelCheckpoint(
            val_precision,
            monitor="val_recall",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )
        earlyStopping = tf.keras.callbacks.EarlyStopping(
            patience=22,
            monitor=(
                "val_binary_accuracy"
                if self.params.multi_label
                else "val_categorical_accuracy"
            ),
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
            monitor=(
                "val_binary_accuracy"
                if self.params.multi_label
                else "val_categorical_accuracy"
            ),
            verbose=1,
        )
        return [
            earlyStopping,
            checkpoint_acc,
            checkpoint_loss,
            reduce_lr_callback,
            cp_callback,
        ]

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

    def classify_ir(self, clip, track, segment_frames=None):
        data = []
        thermal_median = np.empty(len(track.bounds_history), dtype=np.uint16)
        frames_used = []
        for i, region in enumerate(track.bounds_history):
            if region.blank:
                continue
            if region.width == 0 or region.height == 0:
                logging.warn(
                    "No width or height for frame %s regoin %s",
                    region.frame_number,
                    region,
                )
                continue
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
            logging.debug(
                "classifying ir with preprocess %s size %s crop? %s f shape %s region %s",
                self.preprocess_fn.__module__,
                self.params.frame_size,
                crop,
                frame.thermal.shape,
                region,
            )
            preprocessed = preprocess_ir(
                frame,
                (
                    self.params.frame_size,
                    self.params.frame_size,
                ),
                region,
                self.preprocess_fn,
                save_info=f"{region.frame_number} - {region}",
            )
            logging.debug(
                "preprocessed is %s max %s min %s",
                preprocessed.shape,
                np.amax(preprocessed),
                np.amin(preprocessed),
            )
            frames_used.append(region.frame_number)
            data.append(preprocessed)
        if len(data) == 0:
            return None
        data = np.float32(data)
        track_prediction = TrackPrediction(track.get_id(), self.labels)

        output = self.model.predict(data)
        track_prediction.classified_clip(output, output, np.array(frames_used))
        track_prediction.normalize_score()
        return track_prediction

    def predict(self, frames):
        return self.model.predict(frames)

    def confusion_tracks(self, dataset, filename, threshold=0.8):
        logging.info("Calculating confusion with threshold %s", threshold)
        true_categories = []
        track_ids = []
        avg_mass = []
        for x, y in dataset:
            true_categories.extend(y[0].numpy())
            # dataset_y[0]
            track_ids.extend(y[1].numpy())
            avg_mass.extend(y[2].numpy())
        if len(true_categories) > 1:
            if self.params.multi_label:
                # multi = []
                # for y in true_categories:
                # multi.append(tf.where(y).numpy().ravel())
                # print(y, tf.where(y))
                # true_categories = np.int64(true_categories)
                pass
            else:
                true_categories = np.int64(tf.argmax(true_categories, axis=1))
        y_pred = self.model.predict(dataset)
        pred_per_track = {}

        # if self.params.multi_label:
        self.labels.append("nothing")
        # predicted_categori/es = []
        # for p in y_pred:
        # predicted_categories.append(tf.where(p >= 0.8).numpy().ravel())
        # predicted_categories = np.int64(predicted_categories)

        flat_y = []
        for y, track_id, mass, p in zip(true_categories, track_ids, avg_mass, y_pred):
            if self.params.multi_label:
                y_max = np.argmax(y)
            else:
                y_max = y
            track_pred = pred_per_track.setdefault(
                track_id, (y_max, TrackPrediction(track_id, self.labels))
            )
            track_pred[1].classified_frame(None, p, mass)

        results = [
            ("No-Smoothing", []),
            ("Old-Smoothing", []),
            ("New-Smoothing", []),
        ]
        for y, pred in pred_per_track.values():
            pred.normalize_score()
            no_smoothing = np.mean(pred.predictions, axis=0)
            masses = np.array(pred.masses)[:, None]
            old_smoothing = pred.class_best_score
            new_smooth = pred.predictions * masses
            new_smooth = np.sum(new_smooth, axis=0)
            new_smooth /= np.sum(masses)
            # logging.info(
            #     "Smoothing %s with masses %s", np.round(100 * pred.predictions), masses
            # )
            # logging.info(
            #     "N smooth %s old %s new %s",
            #     np.round(100 * no_smoothing),
            #     np.round(100 * old_smoothing),
            #     np.round(100 * new_smooth),
            # )
            for i, pred_type in enumerate([no_smoothing, old_smoothing, new_smooth]):
                best_pred = np.argmax(pred_type)
                confidence = pred_type[best_pred]
                if confidence < threshold:
                    best_pred = len(self.labels) - 1  # Nothing
                results[i][1].append(best_pred)
            flat_y.append(y)

        true_categories = np.int64(flat_y)
        # else:
        #     predicted_categories = np.int64(tf.argmax(y_pred, axis=1))
        for result in results:
            cm = confusion_matrix(
                true_categories, np.int64(result[1]), labels=np.arange(len(self.labels))
            )
            # Log the confusion matrix as an image summary.
            figure = plot_confusion_matrix(cm, class_names=self.labels)
            plt.savefig(f"{result[0]}-{filename}", format="png")

    def confusion_tfrecords(self, dataset, filename):
        true_categories = tf.concat([y for x, y in dataset], axis=0)
        if len(true_categories) > 1:
            if self.params.multi_label:
                # multi = []
                # for y in true_categories:
                # multi.append(tf.where(y).numpy().ravel())
                # print(y, tf.where(y))
                # true_categories = np.int64(true_categories)
                pass
            else:
                true_categories = np.int64(tf.argmax(true_categories, axis=1))
        y_pred = self.model.predict(dataset)
        if self.params.multi_label:
            self.labels.append("nothing")
            # predicted_categori/es = []
            # for p in y_pred:
            # predicted_categories.append(tf.where(p >= 0.8).numpy().ravel())
            # predicted_categories = np.int64(predicted_categories)

            flat_p = []
            flat_y = []
            for y, p in zip(true_categories, y_pred):
                index = 0
                for y_l, p_l in zip(y, p):
                    predicted = p_l >= 0.8
                    if y_l == 0 and predicted:
                        flat_y.append(len(self.labels) - 1)
                        flat_p.append(index)
                    elif y_l == 1 and predicted:
                        flat_y.append(index)
                        flat_p.append(index)
                    elif y_l == 1 and not predicted:
                        flat_y.append(index)
                        flat_p.append(len(self.labels) - 1)
                    # elif y_l == 0 and not predicted:
                    # all good
                    # continue
                    index += 1
            true_categories = np.int64(flat_p)
            predicted_categories = np.int64(flat_y)
        else:
            predicted_categories = np.int64(tf.argmax(y_pred, axis=1))

        cm = confusion_matrix(
            true_categories, predicted_categories, labels=np.arange(len(self.labels))
        )
        np.save(str(Path(filename).with_suffix(".npy")), cm)
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

    #
    # # needs to be updated to work with tfrecord datagen
    # def track_accuracy(self, dataset, confusion="confusion.png"):
    #     dataset.set_read_only(True)
    #     dataset.use_segments = self.params.use_segments
    #     predictions = []
    #     actual = []
    #     raw_predictions = []
    #     total = 0
    #     correct = 0
    #     samples_by_label = {}
    #     incorrect_labels = {}
    #     for sample in dataset.segments:
    #         label_samples = samples_by_label.setdefault(sample.label, {})
    #         if sample.track_id in label_samples:
    #             label_samples[sample.track_id].append(sample)
    #         else:
    #             label_samples[sample.track_id] = [sample]
    #     bird_tracks = len(label_samples.get("bird", []))
    #
    #     for label in dataset.label_mapping.keys():
    #         incorrect = {}
    #         incorrect_labels[label] = incorrect
    #         track_samples = samples_by_label.get(label)
    #         if not track_samples:
    #             logging.warn("No samples for %s", label)
    #             continue
    #         track_samples = track_samples.values()
    #         if label == "insect" or label == "false-positive":
    #             track_samples = np.random.choice(
    #                 list(track_samples),
    #                 min(len(track_samples), bird_tracks),
    #                 replace=False,
    #             )
    #         logging.info("taking %s tracks for %s", len(track_samples), label)
    #         mapped_label = dataset.mapped_label(label)
    #         for track_segments in track_samples:
    #             segment_db = dataset.numpy_data.load_segments(track_segments)
    #             frame_db = {}
    #             for frames in segment_db.values():
    #                 for f in frames:
    #                     frame_db[f.frame_number] = f
    #             track_prediction = self.classify_track_data(
    #                 track_segments[0].track_id,
    #                 frame_db,
    #                 segments=track_segments,
    #             )
    #
    #             total += 1
    #             if track_prediction is None or len(track_prediction.predictions) == 0:
    #                 logging.warn("No predictions for %s", track_segments[0].track_id)
    #                 continue
    #             avg = np.mean(track_prediction.predictions, axis=0)
    #             actual.append(self.labels.index(mapped_label))
    #             predictions.append(track_prediction.best_label_index)
    #
    #             raw_predictions.append(avg)
    #             if actual[-1] == predictions[-1]:
    #                 correct += 1
    #             else:
    #                 if track_prediction.predicted_tag() in incorrect:
    #                     incorrect[track_prediction.predicted_tag()].append(
    #                         track_segments[0].unique_track_id
    #                     )
    #                 else:
    #                     incorrect[track_prediction.predicted_tag()] = [
    #                         track_segments[0].unique_track_id
    #                     ]
    #
    #             if total % 50 == 0:
    #                 logging.info("Processed %s", total)
    #     for label, incorrect in incorrect_labels.items():
    #         logging.info("Incorrect ************ %s", label)
    #         logging.info(incorrect.get("false-positive"))
    #     logging.info("Predicted correctly %s", round(100 * correct / total))
    #     self.f1(actual, raw_predictions)
    #
    #     if confusion is not None:
    #         cm = confusion_matrix(
    #             actual, predictions, labels=np.arange(len(self.labels))
    #         )
    #         figure = plot_confusion_matrix(cm, class_names=self.labels)
    #         plt.savefig(confusion, format="png")


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

    # Use white text if squares are dark; otherwise black.
    counts = cm.copy()
    threshold = counts.max() / 2.0

    print("Threshold is", threshold, " for ", cm.max())
    # Normalize the confusion matrix.

    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm = np.nan_to_num(cm)
    cm = np.uint8(np.round(cm * 100))

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if counts[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def log_confusion_matrix(epoch, logs, model, dataset, writer):
    # Use the model to predict the values from the validation dataset.

    true_categories = tf.concat([y for x, y in dataset], axis=0)
    true_categories = np.int64(tf.argmax(true_categories, axis=1))
    y_pred = model.model.predict(dataset)

    predicted_categories = np.int64(tf.argmax(y_pred, axis=1))

    cm = confusion_matrix(
        true_categories, predicted_categories, labels=np.arange(len(model.labels))
    )
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=model.labels)
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
    if params.multi_label:
        return tf.keras.losses.BinaryCrossentropy(
            label_smoothing=params.label_smoothing,
        )
    return tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=params.label_smoothing,
    )


def optimizer(params):
    if params.learning_rate_decay is not None:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            params.learning_rate,
            decay_steps=100000,
            decay_rate=params.learning_rate_decay,
            staircase=True,
        )
    else:
        learning_rate = params.learning_rate  # setup optimizer
    if learning_rate:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return optimizer


def validate_model(model_file):
    return Path(model_file).exists()


# HYPER PARAM TRAINING OF A MODEL
#
HP_DENSE_SIZES = hp.HParam("dense_sizes", hp.Discrete([""]))
HP_MVM = hp.HParam("mvm", hp.Discrete([3.0]))

HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([64]))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam"]))
HP_LEARNING_RATE = hp.HParam("learning_rate", hp.Discrete([0.001]))
HP_EPSILON = hp.HParam("epislon", hp.Discrete([1e-7]))  # 1.0 and 0.1 for inception
HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.0]))
HP_RETRAIN = hp.HParam("retrain_layer", hp.Discrete([-1]))
HP_LEARNING_RATE_DECAY = hp.HParam("learning_rate_decay", hp.Discrete([1.0]))

METRIC_ACCURACY = "accuracy"
METRIC_LOSS = "loss"


# GRID SEARCH
def train_test_model(model, hparams, log_dir, writer, epochs):
    # if not self.model:
    train_files = model.data_dir + "/train"
    validate_files = model.data_dir + "/validation"
    test_files = model.data_dir + "/test"
    mvm = hparams[HP_MVM]
    if mvm >= 1.0:
        mvm = True
    else:
        mvm = False
    train, remapped = get_dataset(
        train_files,
        model.type,
        model.labels,
        batch_size=model.params.batch_size,
        image_size=model.params.output_dim[:2],
        preprocess_fn=model.preprocess_fn,
        include_features=mvm,
        augment=True,
        scale_epoch=4,
    )
    validate, remapped = get_dataset(
        validate_files,
        model.type,
        model.labels,
        batch_size=model.params.batch_size,
        image_size=model.params.output_dim[:2],
        preprocess_fn=model.preprocess_fn,
        include_features=mvm,
        augment=False,
        scale_epoch=4,
    )

    test, _ = get_dataset(
        test_files,
        model.type,
        model.labels,
        batch_size=model.params.batch_size,
        image_size=model.params.output_dim[:2],
        preprocess_fn=model.preprocess_fn,
        include_features=mvm,
        deterministic=True,
        augment=False,
    )

    labels = model.labels

    opt = None
    learning_rate = hparams[HP_LEARNING_RATE]
    epsilon = hparams[HP_EPSILON]

    if hparams[HP_OPTIMIZER] == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, epsilon=epsilon)
    model.model.compile(
        optimizer=opt, loss=loss(model.params), metrics=["accuracy"], run_eagerly=True
    )
    cm_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: log_confusion_matrix(
            epoch, logs, model, test, writer
        )
    )
    history = model.model.fit(
        train,
        validation_data=validate,
        epochs=epochs,
        shuffle=False,
        # callbacks=[cm_callback],
        verbose=2,
    )
    tf.keras.backend.clear_session()
    gc.collect()
    del model
    del train
    del validate
    gc.collect()
    return history


def grid_search(keras_model, epochs=1):
    dir_name = keras_model.log_dir + "/hparam_tuning"
    with tf.summary.create_file_writer(dir_name).as_default():
        hp.hparams_config(
            hparams=[
                HP_MVM,
                HP_BATCH_SIZE,
                HP_DENSE_SIZES,
                HP_LEARNING_RATE,
                HP_OPTIMIZER,
                HP_EPSILON,
                HP_RETRAIN,
                HP_DROPOUT,
                HP_LEARNING_RATE_DECAY,
            ],
            metrics=[
                hp.Metric(METRIC_ACCURACY, display_name="Accuracy"),
                hp.Metric(METRIC_LOSS, display_name="Loss"),
            ],
        )
    session_num = 0
    hparams = {}
    for mvm in HP_MVM.domain.values:
        for batch_size in HP_BATCH_SIZE.domain.values:
            for dense_size in HP_DENSE_SIZES.domain.values:
                for retrain_layer in HP_RETRAIN.domain.values:
                    for learning_rate in HP_LEARNING_RATE.domain.values:
                        for optimizer in HP_OPTIMIZER.domain.values:
                            for epsilon in HP_EPSILON.domain.values:
                                for dropout in HP_DROPOUT.domain.values:
                                    for (
                                        learning_rate_decay
                                    ) in HP_LEARNING_RATE_DECAY.domain.values:
                                        hparams = {
                                            HP_MVM: mvm,
                                            HP_DENSE_SIZES: dense_size,
                                            HP_BATCH_SIZE: batch_size,
                                            HP_LEARNING_RATE: learning_rate,
                                            HP_OPTIMIZER: optimizer,
                                            HP_EPSILON: epsilon,
                                            HP_RETRAIN: retrain_layer,
                                            HP_DROPOUT: dropout,
                                            HP_LEARNING_RATE_DECAY: learning_rate_decay,
                                        }

                                        dense_layers = []
                                        if dense_size != "":
                                            for i, size in enumerate(dense_size):
                                                dense_layers[i] = int(size)

                                        # for some reason cant have None values in hyper params array
                                        if learning_rate_decay == 1.0:
                                            learning_rate_decay = None
                                        keras_model.params["learning_rate_decay"] = (
                                            learning_rate_decay
                                        )
                                        if mvm >= 1.0:
                                            keras_model.params["mvm"] = True
                                            keras_model.params["hq_mvm"] = mvm > 1
                                            keras_model.params["forest"] = mvm > 2

                                        else:
                                            keras_model.params["mvm"] = False

                                        keras_model.build_model(
                                            dense_sizes=dense_layers,
                                            retrain_from=(
                                                None
                                                if retrain_layer == -1
                                                else retrain_layer
                                            ),
                                            dropout=None if dropout == 0.0 else dropout,
                                        )
                                        keras_model.model.summary()

                                        run_name = "run-%d" % session_num
                                        print("--- Starting trial: %s" % run_name)
                                        print({h.name: hparams[h] for h in hparams})
                                        run(
                                            keras_model,
                                            dir_name + "/" + run_name,
                                            hparams,
                                            epochs,
                                        )
                                        session_num += 1


def run(keras_model, log_dir, hparams, epochs):
    with tf.summary.create_file_writer(log_dir).as_default() as w:
        hp.hparams(hparams)  # record the values used in this trial
        history = train_test_model(keras_model, hparams, log_dir, w, epochs=epochs)
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


def get_excluded(type, multi_label=False):
    if type == "thermal":
        return thermaldataset.get_excluded(), thermaldataset.get_remapped(multi_label)
    else:
        return irdataset.get_excluded(), irdataset.get_remapped()


def get_dataset(
    pattern,
    type,
    labels,
    **args,
):
    if type == "thermal":
        return get_tf(thermaldataset.load_dataset, pattern, labels, **args)
    else:
        return get_tf(irdataset.load_dataset, pattern, labels, **args)


class MetaJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SegmentType) or isinstance(obj, FrameTypes):
            return obj.name
        return json.JSONEncoder.default(self, obj)
