import itertools
import io
import time
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


import numpy as np
import gc
import time
import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path

from sklearn.metrics import confusion_matrix
from ml_tools.datasetstructures import SegmentType

from ml_tools.interpreter import Interpreter
from classify.trackprediction import TrackPrediction
from ml_tools.hyperparams import HyperParams
from ml_tools import thermaldataset
from ml_tools.resnet.wr_resnet import WRResNet

from ml_tools import irdataset
from ml_tools.tfdataset import get_weighting, get_dataset as get_tf, apply_label_mapping
from ml_tools.preprocess import FrameTypes
from ml_tools.thermalwriter import MeanData

classify_i = 0


class KerasModel(Interpreter):
    """Defines a deep learning model"""

    VERSION = 1
    TYPE = "Keras"

    def __init__(
        self, train_config=None, labels=None, data_dir=None, run_over_network=False
    ):
        self.model = None
        self.datasets = None
        self.remapped = None
        self.epochs = None
        self.run_over_network = run_over_network
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
        self.excluded_labels = None
        self.remapped_labels = None
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
        self.params.set_use_segments(
            meta.get("config", {}).get("build", {}).get("use_segments", True)
        )
        pads = meta.get("background_average")
        if pads is None:
            self.pads = MeanData()
        else:
            self.pads = MeanData(
                thermal=pads["thermal"],
                filtered=pads["filtered"],
                thermal_norm=pads["thermal_norm"],
                frames_used=1,
            )
            self.pads = self.pads * 255
        logging.info("Pads are %s", self.pads)

    def shape(self):
        if self.model is None:
            return None
        inputs = self.model.inputs
        shape = []
        for input in inputs:
            in_shape = input.shape
            if isinstance(in_shape, tuple):
                shape.append(in_shape)
            else:
                shape.append(tuple(in_shape.as_list()))
        if len(shape) == 1:
            return len(shape), shape[0]
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
        elif pretrained_model == "efficientnetv2b3":
            return (
                tf.keras.applications.EfficientNetV2B3(
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
        import tensorflow_decision_forests as tfdf

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

    def build_model_lstm(self):
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow import keras

        # 1. Inputs
        tile_size = 32
        num_tiles = 25
        mask_input = layers.Input(shape=(num_tiles,), name="input_mask")

        input_image = layers.Input(
            shape=(num_tiles, tile_size, tile_size, 3), name="input_image"
        )
        # Shape: (Batch, 25, 1) -> Using your relative time / missing flag array
        inputs = {"input_image": input_image, "input_mask": mask_input}
        expanded_mask = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(
            mask_input
        )  # (None, 25, 1)

        # 3. Apply the native Keras Masking Layer to the timeline data
        # This flags the -1.0 values and propagates a hidden boolean mask to the LSTM
        masked_timeline = layers.Masking(mask_value=-1.0, name="sequence_masking")(
            expanded_mask
        )

        time_features = layers.Dense(64, activation="swish", name="time_projection")(
            masked_timeline
        )
        time_features = layers.Dense(128, activation="swish", name="time_expansion")(
            time_features
        )

        # 4. Lightweight Backbone Configuration
        # We include pooling to squeeze the 32x32 spatial dimensions down into a flat vector
        base_cnn = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(tile_size, tile_size, 3),
        )
        base_cnn.trainable = True

        x = layers.TimeDistributed(
            layers.BatchNormalization(axis=-1), name="channel_standardizer"
        )(input_image)
        # 5. Extract features from each frame independently using TimeDistributed
        # Output shape transforms from (None, 25, 32, 32, 3) to (None, 25, 1280)
        visual_embeddings = layers.TimeDistributed(
            base_cnn, name="cnn_feature_extractor"
        )(x)

        # 6. Merge the visual features and the masked timestamps
        # Output shape: (None, 25, 1281)
        combined_sequence = layers.Concatenate(axis=-1, name="merge_features")(
            [visual_embeddings, time_features]
        )

        # 1. Grab the boolean mask already generated by your Masking layer and squeeze it
        # (None, 25, 1) -> (None, 25)
        explicit_mask = layers.Lambda(
            lambda x: x._keras_mask, output_shape=(25,), name="clean_mask"
        )(masked_timeline)

        # 7. Recurrent Sequence Processing
        # The LSTM automatically reads the mask and skips computing the padded slots!
        # lstm_out = layers.LSTM(128, return_sequences=False, name="temporal_lstm")(combined_sequence)
        gru_out = layers.Bidirectional(
            layers.GRU(128, return_sequences=False, dropout=0.2), name="sequence_logic"
        )(combined_sequence, mask=explicit_mask)
        x = layers.BatchNormalization(name="batch_norm")(gru_out)

        # 8. Classification Head
        x = layers.Dropout(0.5)(x)

        # Dense reasoning block right before the final bottleneck
        x = layers.Dense(
            128,
            activation="swish",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="dense_header",
        )(x)
        x = layers.Dropout(0.4, name="head_dropout")(x)

        output = tf.keras.layers.Dense(
            len(self.labels),
            activation=None,
            name="prediction",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
        model = models.Model(inputs=inputs, outputs=output)
        return model

    def build_model(
        self,
        dense_sizes=None,
        retrain_from=None,
        dropout=None,
        run_name=None,
        single_input=True,
    ):
        RNN_MODEL = False
        if RNN_MODEL:
            # this isn't performing well and is slow to train
            # the dataset also needs to be adjusted to handle this
            return self.build_model_lstm()
        from tensorflow.keras import layers
        from tensorflow import keras

        # width = self.params.frame_size
        width = self.params.output_dim[0]
        input_image = tf.keras.Input(
            shape=(width, width, len(self.params.channels)), name="input_image"
        )
        weights = "imagenet"
        base_model, preprocess = self.get_base_model(input_image, weights=weights)
        self.preprocess_fn = preprocess
        # inputs = base_model.input

        # Step A: Standardise channel means (92.96, 47.33, 30.62) & variances automatically
        x = layers.BatchNormalization(axis=-1, name="channel_standardizer")(input_image)

        # 2. Trainable 1x1 conv with 3 filters to re-weight and re-bias the RGB channels
        # This lets the network automatically discover the optimal math to align your normalisations
        x = tf.keras.layers.Conv2D(3, (1, 1), activation=None, name="channel_aligner")(
            x
        )

        x = base_model(x)
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
            input_image = [cnn.input, feature_input]
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
            self.model = tf.keras.models.Model(input_image, outputs=preds)
        elif self.params.lstm:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            for i in dense_sizes:
                x = tf.keras.layers.Dense(i, activation="relu")(x)
            # gp not sure how many should be pre lstm, and how many post
            cnn = tf.keras.models.Model(input_image, outputs=x)

            self.model = self.add_lstm(cnn)
        else:
            # Multi input adding information about the frame number used
            if not single_input:
                # --- Input 2: The Timeline Mask Layer (5x5x1) ---
                mask_input = layers.Input(shape=(5, 5, 1), name="input_mask")
                input_image = {"input_image": input_image, "input_mask": mask_input}

                # Generate temporal feature maps matching the spatial dimensions
                # AI was insistent on this being the way to go until i questioned it and then it said Dense was obviously better
                # t = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(mask_input)
                # t = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(t)

                # input_mask = Input(shape=(5, 5, 1), name='input_mask')

                # # 1. Shift the mask natively
                # shifted_mask = mask_input + 1.0

                # # 2. Use Keras operations instead of tf.nn / tf.math
                # relu_mask = keras.ops.relu(shifted_mask)
                # binary_presence_gate = keras.ops.ceil(relu_mask)

                # Step 1: Project the single timestamp into a 64-dimensional time embedding vector per cell
                # A Dense layer applied to a 3D tensor operates independently on every single (5,5) cell!
                time_embedding = layers.Dense(
                    64, activation="relu", name="time_feature_projection"
                )(
                    mask_input
                )  # Shape: (None, 5, 5, 64)
                time_embedding = layers.Dense(
                    128, activation="relu", name="time_feature_expansion"
                )(
                    time_embedding
                )  # Shape: (None, 5, 5, 128)

                # --- Feature Fusion ---
                # Concatenate visual maps (5x5x1536) and time maps (5x5x128) along the channels
                image_features = x
                # maybe add
                image_features = tf.keras.layers.SpatialDropout2D(0.1)(image_features)

                # sounds good in practice but actually gives worse results

                # # 1. Shift the mask natively
                # shifted_mask = mask_input + 1.0

                # # 2. Use Keras operations instead of tf.nn / tf.math
                # relu_mask = keras.ops.relu(shifted_mask)
                # binary_presence_gate = keras.ops.ceil(relu_mask)

                # image_features = image_features * binary_presence_gate

                combined = layers.Concatenate(name="input_concat")(
                    [image_features, time_embedding]
                )  # Shape: (None, 5, 5, 1664)

                # 1. Compress channel depth from 1664 to 256 using 1x1 convolution
                x = layers.Conv2D(256, (1, 1), activation="swish", padding="same")(
                    combined
                )  # ~426K params

                # Mix the combined space-time features together
                x = layers.Conv2D(256, (3, 3), activation="swish", padding="same")(x)
            # else:
            # input_image = {"input_image": input_image}

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            if self.params.mvm:
                mvm_inputs = tf.keras.layers.Input((188))
                input_image = [input_image, mvm_inputs]
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
                    x = tf.keras.layers.Dense(i, activation="swish")(x)
            if dropout:
                logging.info("Using dropout of %s", dropout)
                x = tf.keras.layers.Dropout(dropout)(x)

            activation = "softmax"
            if self.params.multi_label:
                activation = "sigmoid"
                # will need to add this in after training
                activation = None
            logging.info("Using %s activation", activation)
            preds = tf.keras.layers.Dense(
                len(self.labels), activation=activation, name="prediction"
            )(x)
            self.model = tf.keras.models.Model(input_image, outputs=preds)
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
        return self.model

    def adjust_final_layer(self):
        # Adjust final layer to a new set of labels, by removing it and re adding
        # new_model = tf.keras.models.Sequential(self.model.layers[:-3])
        self.model = tf.keras.Model(
            inputs=self.model.input, outputs=self.model.layers[-2].output
        )

        # model = tf.keras.Model(inputs=self.model.input, outputs=x)

        activation = "softmax"
        if self.params.multi_label:
            activation = "sigmoid"

        retrain_from = self.params.retrain_layer
        if retrain_from:
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    # apparently this shouldn't matter as we set base_training = False
                    layer.trainable = False
                    logging.info("dont train %s %s", i, layer.name)
                else:
                    layer.trainable = i >= retrain_from
        else:
            self.model.trainable = self.params.base_training

        # add final layer after as always want this trainable
        logging.info(
            "Adding new final layer with %s activation and %s labels ",
            activation,
            len(self.labels),
        )
        preds = tf.keras.layers.Dense(
            len(self.labels), activation=activation, name="prediction"
        )(self.model.output)

        self.model = tf.keras.models.Model(self.model.inputs, outputs=preds)
        self.model.summary()

    def init_model(self, model_file, weights=None, load_model=True):
        super().__init__(model_file, self.run_over_network)
        self.weights = weights
        if self.run_over_network:
            return
        if load_model:
            self.load_model()

    def load_model(self):
        if self.run_over_network:
            return
        logging.info(
            "Loading %s with model weights %s without compiling",
            self.model_file,
            self.weights,
        )
        if self.model_file.suffix == ".pb":
            self.model = tf.keras.models.load_model(
                self.model_file.parent, compile=False
            )
        else:
            self.model = tf.keras.models.load_model(self.model_file, compile=False)

        self.model.trainable = False

        if self.weights is not None:
            self.model.load_weights(self.weights)
            logging.info("Loaded weight %s", self.weights)

    def save(
        self,
        run_name=None,
        history=None,
        test_results=None,
        rebalance=False,
        fine_tune=None,
        single_input=False,
    ):
        # create a save point
        if run_name is None:
            run_name = self.params.model_name

        run_dir = self.checkpoint_folder / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        self.model.save(str(self.checkpoint_folder / run_name / f"{run_name}.keras"))
        self.save_metadata(
            run_name, history, test_results, rebalance, fine_tune, single_input
        )

    def save_metadata(
        self,
        run_name=None,
        history=None,
        test_results=None,
        rebalance=False,
        fine_tune=None,
        single_input=False,
    ):
        #  save metadata
        if run_name is None:
            run_name = self.params.model_name
        model_stats = {}
        model_stats["name"] = self.params.model_name
        model_stats["labels"] = self.labels
        model_stats["single_input"] = single_input
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
        if fine_tune is not None:
            model_stats["fine_tune"] = str(fine_tune)
        if rebalance:
            model_stats["rebalance"] = rebalance
        model_stats["pads"] = self.pads.to_dict()
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

    def init_train(self, epochs):
        self.epochs = epochs
        if self.params.excluded_labels is not None:
            self.excluded_labels = self.params.excluded_labels
        else:
            self.excluded_labels, self.remapped_labels = get_excluded(
                self.data_type, self.params.multi_label
            )

        if self.params.remapped_labels is not None:
            self.remapped_labels = self.params.remapped_labels
        else:
            self.remapped_labels, self.remapped_labels = get_excluded(
                self.data_type, self.params.multi_label
            )
        acceptable_types = get_acceptable_labels(self.data_type, self.remapped_labels)
        if acceptable_types is not None:
            for lbl in self.labels:
                if lbl not in acceptable_types and lbl not in self.excluded_labels:
                    logging.info(
                        "Adding %s to excluded list as it is not in our acceptable label list",
                        lbl,
                    )
                    self.excluded_labels.append(lbl)

        logging.info(
            "Excluding %s remapping %s accepted labels %s",
            self.excluded_labels,
            self.remapped_labels,
            acceptable_types,
        )
        if self.params.multi_label:
            if "weka" not in self.labels:
                self.labels.append("weka")
            if "chicken" not in self.labels:
                self.labels.append("chicken")
        self.labels.sort()
        self.orig_labels = self.labels.copy()
        self.preprocess_fn = self.get_preprocess_fn()

        self.labels, tf_mappings = apply_label_mapping(
            self.labels, self.excluded_labels, self.remapped_labels
        )
        logging.info(
            "Applied label remapping from %s have model labels of %s",
            self.orig_labels,
            self.labels,
        )
        self.remapped = {}
        for k, v in tf_mappings.items():
            self.remapped[self.orig_labels[k]] = (
                self.labels[v] if v != -1 else "Nothing"
            )
            logging.info(
                "Original %s is mapped to %s",
                self.orig_labels[k],
                "Nothing" if v == -1 else self.labels[v],
            )
        logging.info("Remapped is %s", self.remapped)
        return tf_mappings

    def train_model(
        self,
        epochs,
        run_name,
        weights=None,
        rebalance=False,
        resample=False,
        fine_tune=None,
        warm_down=False,
        single_input=True,
    ):
        logging.info(
            "%s Training model for %s epochs with weights %s with single input as: %s",
            run_name,
            epochs,
            weights,
            single_input,
        )
        tf_mappings = self.init_train(epochs)

        self.log_dir = self.log_base / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if fine_tune is not None:
            self.init_model(fine_tune, weights=weights)
            # dont know if this is needed
            self.model.trainable = self.params.base_training

            # for multi input model this needs to be adjusted
            # self.adjust_final_layer()
            if rebalance:
                logging.info(
                    "Fine tuning on a balanced dataset, setting all layers before the concatenate to not be trainable"
                )
                found_concat = False
                for layer in self.model.layers:

                    if isinstance(layer, tf.keras.layers.Concatenate):
                        found_concat = True
                        layer.trainable = False
                        continue

                    # Everything up to and including concat stays False; everything after becomes True
                    layer.trainable = found_concat
                    if layer.trainable and isinstance(layer, tf.keras.layers.Dropout):
                        layer.rate = 0.5  # Update the rate directly
                        logging.info(
                            f"Successfully updated {layer.name} rate to {layer.rate}"
                        )
            self.model.summary()
        else:

            self.model = self.build_model(
                dense_sizes=self.params.dense_sizes,
                retrain_from=self.params.retrain_layer,
                dropout=self.params.dropout,
                run_name=run_name,
                single_input=single_input,
            )

            if weights is not None:
                self.model.load_weights(weights)

        self.model.summary()

        train_files = self.data_dir / "train"
        validate_files = self.data_dir / "validation"
        augment = fine_tune is None
        self.train, epoch_size = get_dataset(
            train_files,
            self.data_type,
            self.labels,
            batch_size=self.params.batch_size,
            image_size=self.params.output_dim[:2],
            preprocess_fn=self.preprocess_fn,
            resample=resample,
            stop_on_empty_dataset=False,
            include_features=self.params.mvm,
            augment=augment,
            excluded_labels=self.excluded_labels,
            remapped_labels=self.remapped_labels,
            # dist=self.dataset_counts["train"],
            multi_label=self.params.multi_label,
            num_frames=self.params.square_width**2,
            channels=self.params.channels,
            pads=self.pads,
            tf_mappings=tf_mappings,
            downsize_fp=True,
            rebalance=rebalance,
            single_input=single_input,
        )

        steps = epoch_size // self.params.batch_size

        # self.remapped = remapped
        self.validate, _ = get_dataset(
            validate_files,
            self.data_type,
            self.labels,
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
            pads=self.pads,
            tf_mappings=tf_mappings,
            single_input=single_input,
        )
        logging.info(
            "Training on %s  with class weights %s",
            self.labels,
            self.class_weights,
        )

        self.save(run_name, fine_tune=fine_tune, rebalance=rebalance)

        checkpoints = self.checkpoints(run_name, warmup_epochs=2, fine_tuning=warm_down)
        if warm_down:
            optimizer_fn = tf.keras.optimizers.Adam(
                learning_rate=self.params.fine_tune_learning_rate
            )
            logging.info(
                "Warming down with adam and augment %s and learning rate %s",
                augment,
                self.params.fine_tune_learning_rate,
            )
        else:
            if fine_tune is None:
                warmup_callback = StepWarmupCallback(
                    target_lr=self.params.learning_rate,
                    warmup_epochs=2,
                    steps_per_epoch=steps,
                )
                checkpoints.append(warmup_callback)

            optimizer_fn = optimizer(
                self.params, steps, self.epochs, fine_tune=fine_tune is not None
            )
        self.model.compile(
            optimizer=optimizer_fn,
            loss=loss(self.params),
            metrics={"prediction": metrics(self.params.multi_label)},
        )

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
            ],
        )
        history = history.history
        test_accuracy = None
        test_files = self.data_dir / "test"

        if len(list(test_files.glob("*.tfrecord"))) > 0:
            self.test, _ = get_dataset(
                test_files,
                self.data_type,
                self.labels,
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
                pads=self.pads,
                tf_mappings=tf_mappings,
                single_input=single_input,
            )
            if self.test:
                test_accuracy = self.model.evaluate(self.test)

        self.save(
            run_name,
            history=history,
            test_results=test_accuracy,
            rebalance=rebalance,
            fine_tune=fine_tune,
            single_input=single_input,
        )

    def warm_down(self, run_name, weights, tf_mappings, epochs=5, single_input=False):
        logging.info(
            "Warming down for 5 epochs with weights %s without augmentation", weights
        )

        self.model.load_weights(weights)
        log_dir = self.log_base / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        train_files = self.data_dir / "train"
        # reload train dataset with augment false
        self.train, epoch_size = get_dataset(
            train_files,
            self.data_type,
            self.labels,
            batch_size=self.params.batch_size,
            image_size=self.params.output_dim[:2],
            preprocess_fn=self.preprocess_fn,
            stop_on_empty_dataset=False,
            include_features=self.params.mvm,
            augment=False,
            excluded_labels=self.excluded_labels,
            remapped_labels=self.remapped_labels,
            # dist=self.dataset_counts["train"],
            multi_label=self.params.multi_label,
            num_frames=self.params.square_width**2,
            channels=self.params.channels,
            pads=self.pads,
            downsize_fp=True,
            tf_mappings=tf_mappings,
            single_input=single_input,
        )

        self.save_metadata(run_name)
        self.save(run_name)
        checkpoints = self.checkpoints(run_name, True)
        logging.info(
            "Fine tuning with adam and a learning rate of %s",
            self.params.fine_tune_learning_rate,
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.params.fine_tune_learning_rate
            ),
            loss=loss(self.params),
            metrics={"prediction": metrics(self.params.multi_label)},
        )

        history = self.model.fit(
            self.train,
            validation_data=self.validate,
            epochs=epochs,
            shuffle=False,
            callbacks=[
                *checkpoints,
            ],
        )
        history = history.history

        if self.test:
            test_accuracy = self.model.evaluate(self.test)

        self.save(run_name, history=history, test_results=test_accuracy)

    def checkpoints(
        self, run_name, fine_tuning=False, stop_on=("val_loss", "min"), warmup_epochs=2
    ):
        checkpoint_file = self.checkpoint_folder / run_name / "cp.weights.h5"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file, save_weights_only=True, verbose=1
        )
        val_f1 = self.checkpoint_folder / run_name / "val_macro_f1.weights.h5"

        f1_loss = tf.keras.callbacks.ModelCheckpoint(
            val_f1,
            monitor="val_macro_f1",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )

        val_loss = self.checkpoint_folder / run_name / "val_loss.weights.h5"

        checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(
            val_loss,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
        )
        val_acc = self.checkpoint_folder / run_name / "val_acc.weights.h5"

        checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
            val_acc,
            monitor=(
                "val_acc_thresh"
                if self.params.multi_label
                else "val_categorical_accuracy"
            ),
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )

        val_precision = self.checkpoint_folder / run_name / "val_recall.weights.h5"

        checkpoint_recall = tf.keras.callbacks.ModelCheckpoint(
            val_precision,
            monitor="val_recall",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )
        checkpoints = [f1_loss, checkpoint_acc, checkpoint_loss, cp_callback]
        if not fine_tuning:
            earlyStopping = tf.keras.callbacks.EarlyStopping(
                patience=11,
                monitor=stop_on[0],
                # monitor=(
                #     "val_binary_accuracy"
                #     if self.params.multi_label
                #     else "val_categorical_accuracy"
                # ),
                mode=stop_on[1],
                restore_best_weights=True,
            )
            checkpoints.append(earlyStopping)

            reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=stop_on[0],
                verbose=1,
                mode=stop_on[1],
                factor=0.5,
                patience=8,
                min_delta=0.0001,
                cooldown=max(warmup_epochs, 3),
                min_lr=0.000001,  # Safety floor: Never drops lower than 10% of standard fine-tuning speed
            )
            checkpoints.append(reduce_lr_callback)
        return checkpoints

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
        # reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor=(
        #         "val_binary_accuracy"
        #         if self.params.multi_label
        #         else "val_categorical_accuracy"
        #     ),
        #     mode="max",
        #     verbose=1,
        # )

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
        track_prediction.classified_track(output, np.array(frames_used))
        track_prediction.normalize_score()
        return track_prediction

    def predict(self, frames):
        if self.run_over_network:
            return self.predict_over_network(frames)
        return self.model.predict(frames)

    def confusion_tracks(
        self, dataset, filename, threshold=0.8, thresholds_per_label=None
    ):
        logging.info(
            "Calculating confusion with threshold %s saving to %s", threshold, filename
        )
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
        # predicted_categori/es = []
        # for p in y_pred:
        # predicted_categories.append(tf.where(p >= 0.8).numpy().ravel())
        # predicted_categories = np.int64(predicted_categories)

        for y, track_id, mass, p in zip(true_categories, track_ids, avg_mass, y_pred):
            # if self.params.multi_label:
            #     y_max = np.argmax(y)
            # else:
            #     y_max = y
            track_pred = pred_per_track.setdefault(
                track_id, (np.nonzero(y)[0], TrackPrediction(track_id, self.labels))
            )
            track_pred[1].classified_frame(None, p, mass)
        flat_y = []
        results = []
        confidences = []
        raw_class_confidences = []
        labels = self.labels.copy()
        labels.append("Nothing")
        for y_true, pred in pred_per_track.values():
            pred.normalize_score()
            preds = np.array([p.prediction for p in pred.predictions])
            # if we do multi label we may of multiple y_true and preds
            # otherwise this will calculate the same as before
            no_smoothing = np.mean(preds, axis=0)
            preds = np.where(no_smoothing >= 0.5)[0]
            if len(preds) == 0:
                preds = [np.argmax(no_smoothing)]
            if len(y_true) > 1:
                ll = []
                for y in y_true:
                    ll.append(labels[y])
                logging.info("Have multiple labels %s", ll)
            for y in y_true:
                if y in preds:
                    idx = y
                    results.append(y)
                    confidences.append(no_smoothing[y])
                    raw_class_confidences.append(no_smoothing)
                    flat_y.append(y)
                    if len(y_true) > 1:
                        logging.info(
                            "Pred %s for %s confs %s",
                            labels[idx],
                            labels[y],
                            np.round(100 * no_smoothing),
                        )

                else:
                    for idx in preds:
                        results.append(idx)
                        confidences.append(no_smoothing[idx])
                        flat_y.append(y)
                        raw_class_confidences.append(no_smoothing)
                        if len(y_true) > 1:
                            logging.info(
                                "Wrong Pred %s for %s confs %s",
                                labels[idx],
                                labels[y],
                                np.round(100 * no_smoothing),
                            )

            assert len(results) == len(flat_y)
        true_categories = np.int64(flat_y)
        # else:
        #     predicted_categories = np.int64(tf.argmax(y_pred, axis=1))

        results = np.int64(results)
        confidences = np.array(confidences)

        # raw_preds_i = np.uint8(raw_preds_i)
        raw_class_confidences = np.array(raw_class_confidences)
        npy_file = filename.parent / f"{filename.stem}-raw.npy"
        logging.info("Saving %s", npy_file)
        with npy_file.open("wb") as f:
            np.save(f, true_categories)
            np.save(f, results)
            np.save(f, raw_class_confidences)

        if thresholds_per_label is not None:
            thresholds_per_label = np.array(thresholds_per_label)
            thresholds_per_label[thresholds_per_label < 0.5] = 0.5

            preds = results.copy()
            for i, lbl_thresh in enumerate(thresholds_per_label):
                pred_mask = preds == i
                # set these to None
                conf_mask = confidences < lbl_thresh
                preds[pred_mask & conf_mask] = len(labels) - 1
            cm = confusion_matrix(true_categories, preds, labels=np.arange(len(labels)))
            # Log the confusion matrix as an image summary.
            figure = plot_confusion_matrix(cm, class_names=labels)
            fscore_file = filename.parent / f"{filename.stem}-fscore"
            plt.savefig(fscore_file.with_suffix(".png"), format="png")
            np.save(fscore_file.with_suffix(".npy"), cm)

        preds = results.copy()

        # set these to None
        preds[confidences < threshold] = len(labels) - 1
        cm = confusion_matrix(true_categories, preds, labels=np.arange(len(labels)))
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=labels)
        out_file = filename.parent / f"{filename.stem}-{round(100*threshold)}%"
        plt.savefig(out_file.with_suffix(".png"), format="png")
        np.save(out_file.with_suffix(".npy"), cm)

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
            self.labels.append("Nothing")
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
def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    plt.clf()
    figure = plt.figure(figsize=(16, 16))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    ylabels = []
    for i, label in enumerate(class_names):
        ylabel = f"{label} ({np.sum(cm[i])})"
        ylabels.append(ylabel)
    plt.yticks(tick_marks, ylabels)

    # Use white text if squares are dark; otherwise black.
    counts = cm.copy()
    threshold = counts.max() / 2.0

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
        # return tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
        return tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            label_smoothing=params.label_smoothing,
        )
    return tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=params.label_smoothing,
    )


def optimizer(params, steps_per_epoch, epochs, fine_tune=False):
    if fine_tune:
        logging.info(
            "Using fine tune cosine optimizer with warm of 2 epochs and final rate %s",
            params.fine_tune_learning_rate,
        )
        # 3 epochs
        warmup_steps = steps_per_epoch * 2
        # 2. Configure the built-in schedule
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,  # Step 0 start rate
            decay_steps=epochs * int(steps_per_epoch),  # Point where decay finishes
            warmup_target=params.fine_tune_learning_rate,  # Peak fine-tuning learning rate
            warmup_steps=warmup_steps,  # Steps to transition from initial to target
        )
    else:

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     params.learning_rate,
        #     decay_steps=int(steps_per_epoch* epochs),
        #     decay_rate=params.learning_rate_decay,
        # )
        # using ReduceLROnPlateau instead
        lr_schedule = 0.0
        # using warmup to set lr
        # params.learning_rate

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

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


def get_acceptable_labels(type, remapped_labels):
    if type == "thermal":
        return thermaldataset.get_acceptable_labels(remapped_labels)
    return irdataset.get_acceptable_labels(remapped_labels)


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


# Registered at module level (rather than nested in metrics()) so Keras can
# resolve them by name when loading a saved model.
@tf.keras.utils.register_keras_serializable(package="Custom")
class LogitPrecision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.sigmoid(y_pred), sample_weight)


@tf.keras.utils.register_keras_serializable(package="Custom")
class LogitRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.sigmoid(y_pred), sample_weight)


@tf.keras.utils.register_keras_serializable(package="Custom")
class LogitMacroF1(tf.keras.metrics.F1Score):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.sigmoid(y_pred), sample_weight)


def metrics(multi_label=True, from_logits=True):
    # 1. Base Accuracy Definition
    if multi_label:
        # If inputs are logits, the threshold is 0.0 (positive vs negative numbers)
        # If inputs are probabilities, the threshold is 0.5
        thresh = 0.0 if from_logits else 0.5
        acc = tf.keras.metrics.BinaryAccuracy(threshold=thresh, name="acc_thresh")
    else:
        acc = tf.metrics.categorical_accuracy

    # 2. Build the output list
    metrics_list = [acc]

    # 3. Handle AUC (It has native logit support)
    metrics_list.append(
        tf.keras.metrics.AUC(
            multi_label=multi_label, from_logits=from_logits, name="auc"
        )
    )

    # 4. Handle Precision, Recall, and F1Score
    if from_logits:
        # Use the module-level Logit* wrappers (apply sigmoid before calculation)
        # so they remain resolvable when the model is saved/loaded.
        metrics_list.extend(
            [
                LogitRecall(name="recall"),
                LogitPrecision(name="precision"),
                LogitMacroF1(average="macro", name="macro_f1"),
            ]
        )
    else:
        # Fallback to your original code if from_logits=False
        metrics_list.extend(
            [
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.F1Score(average="macro", name="macro_f1"),
            ]
        )

    return metrics_list


# def metrics(multi_label):
#     if multi_label:
#         acc = tf.metrics.binary_accuracy
#     else:
#         acc = tf.metrics.categorical_accuracy

#     return [
#                     acc,
#                     tf.keras.metrics.AUC(multi_label= multi_label),
#                     tf.keras.metrics.Recall(),
#                     tf.keras.metrics.Precision(),
#                     tf.keras.metrics.F1Score(average="macro", name="macro_f1")
#                 ]


from tensorflow.keras.callbacks import Callback


class StepWarmupCallback(Callback):
    def __init__(self, target_lr, warmup_epochs, steps_per_epoch):
        super(StepWarmupCallback, self).__init__()
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
        # Calculate the total linear steps for the warmup phase
        self.total_warmup_steps = warmup_epochs * steps_per_epoch
        self.global_step = 0

    def on_train_batch_begin(self, batch, logs=None):
        # Only run adjustments during the warmup phase
        if self.global_step < self.total_warmup_steps:
            # Linear scaling formula2
            lr = (self.global_step / self.total_warmup_steps) * self.target_lr

            # Dynamically update the backend float value (safe for ReduceLROnPlateau)
            self.model.optimizer.learning_rate = lr
        self.global_step += 1

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            current_epoch_start_lr = (
                self.global_step / self.total_warmup_steps
            ) * self.target_lr
            msg = f"[Warmup Phase] Epoch {epoch + 1}/{self.warmup_epochs}: Starting learning rate set to {current_epoch_start_lr:.4e}"

            # 1. Use PRINT with a newline to break past the Keras progress bar cleanly
            print(f"\n🔥 {msg}")

            # 2. Use LOGGING to write a clean, timestamped record into your log file backup
            logging.info(msg)

        self.global_step += 1
