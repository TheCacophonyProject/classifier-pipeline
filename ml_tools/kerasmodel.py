import math
import os
import json
import logging
import tensorflow as tf
import numpy as np
from track.track import TrackChannels
from classify.trackprediction import TrackPrediction
from ml_tools.preprocess import (
    FrameTypes,
    preprocess_movement,
    preprocess_frame,
)


class KerasModel:
    """Defines a deep learning model using the tensorflow v2 keras framework"""

    def __init__(self, train_config=None):
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
        if train_config:
            self.params.update(train_config.hyper_params)
        self.preprocess_fn = None
        self.labels = None
        self.frame_size = None
        self.pretrained_model = None
        self.model = None
        self.use_movement = False
        self.square_width = 1

    def get_base_model(self, input_shape):
        if self.pretrained_model == "resnet":
            return (
                tf.keras.applications.ResNet50(
                    weights="imagenet",
                    include_top=False,
                    input_shape=input_shape,
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
                    weights="imagenet",
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.vgg16.preprocess_input,
            )
        elif self.pretrained_model == "vgg19":
            return (
                tf.keras.applications.VGG19(
                    weights="imagenet",
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.vgg19.preprocess_input,
            )
        elif self.pretrained_model == "mobilenet":
            return (
                tf.keras.applications.MobileNetV2(
                    weights="imagenet",
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.mobilenet_v2.preprocess_input,
            )
        elif self.pretrained_model == "densenet121":
            return (
                tf.keras.applications.DenseNet121(
                    weights="imagenet",
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.densenet.preprocess_input,
            )
        elif self.pretrained_model == "inceptionresnetv2":
            return (
                tf.keras.applications.InceptionResNetV2(
                    weights="imagenet",
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.inception_resnet_v2.preprocess_input,
            )
        elif self.pretrained_model == "inceptionv3":
            return (
                tf.keras.applications.InceptionV3(
                    weights="imagenet",
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.inception_v3.preprocess_input,
            )

        raise "Could not find model" + self.pretrained_model

    def build_model(self, dense_sizes=[1024, 512]):
        input_shape = (
            self.frame_size * self.square_width,
            self.frame_size * self.square_width,
            3,
        )
        inputs = tf.keras.Input(shape=input_shape)
        base_model, preprocess = self.get_base_model(input_shape)
        self.preprocess_fn = preprocess

        base_model.trainable = False
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        for i in dense_sizes:
            x = tf.keras.layers.Dense(i, activation="relu")(x)
        preds = tf.keras.layers.Dense(len(self.labels), activation="softmax")(x)
        self.model = tf.keras.models.Model(inputs, outputs=preds)
        self.model.compile(
            optimizer=self.optimizer(),
            loss=self.loss(),
            metrics=["accuracy"],
        )

    def loss(self):
        softmax = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=self.params.get("label_smoothing"),
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

    def load_weights(self, file):
        dir = os.path.dirname(file)
        weights_path = dir + "/variables/variables"

        self.load_meta(dir)

        if not self.model:
            self.build_model(self.dense_sizes)
        self.model.load_weights(weights_path)

    def load_model(self, model_path, training=False):
        logging.info("Loading %s", model_path)
        dir = os.path.dirname(model_path)
        self.model = tf.keras.models.load_model(dir)
        self.load_meta(dir)
        if not training:
            self.model.trainable = False
        self.model.summary()

    def load_meta(self, dir):
        meta = json.load(open(os.path.join(dir, "metadata.txt"), "r"))
        self.params = meta["hyperparams"]
        self.labels = meta["labels"]
        self.pretrained_model = self.params.get("model", "resnetv2")
        self.preprocess_fn = self.get_preprocess_fn()
        self.frame_size = meta.get("frame_size", 48)
        self.square_width = meta.get("square_width", 1)
        self.use_movement = self.params.get("use_movement", False)
        self.green_type = self.params.get("green_type", FrameTypes.filtered_square)
        self.blue_type = self.params.get("blue_type", FrameTypes.overlay)
        if self.params.get("use_thermal", False):
            self.red_type = FrameTypes.thermal_square
        else:
            self.red_type = FrameTypes.filtered_square
        self.keep_aspect = self.params.get("keep_aspect", False)
        self.dense_sizes = self.params.get("dense_sizes", [1024, 512])

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
        elif self.pretrained_model == "inceptionv3":
            return tf.keras.applications.inception_v3.preprocess_input
        logging.warn(
            "pretrained model %s has no preprocessing function", self.pretrained_model
        )
        return None

    def classify_frame(self, frame, preprocess=True):
        if preprocess:
            frame = preprocess_frame(
                frame,
                (self.frame_size, self.frame_size, 3),
                self.params.get("use_thermal", True),
                augment=False,
                preprocess_fn=self.preprocess_fn,
            )
            if frame is None:
                return None
        output = self.model.predict(frame[np.newaxis, :])
        return output[0]

    def classify_track(
        self,
        clip,
        track,
        keep_all=True,
    ):
        try:
            fp_index = self.labels.index("false-positive")
        except ValueError:
            fp_index = None
        track_prediction = TrackPrediction(
            track.get_id(), track.start_frame, self.labels, keep_all=keep_all
        )

        if self.use_movement:
            data = []
            thermal_median = []
            for region in track.bounds_history:
                frame = clip.frame_buffer.get_frame(region.frame_number)
                frame = frame.crop_by_region(region)
                thermal_median.append(np.median(frame.thermal))
                data.append(frame)
            predictions = self.classify_using_movement(
                data, thermal_median, regions=track.bounds_history
            )
            for i, prediction in enumerate(predictions):
                track_prediction.classified_frame(i, prediction, None)
        else:
            for i, region in enumerate(track.bounds_history):
                frame = clip.frame_buffer.get_frame(region.frame_number)
                frame = frame.crop_by_region(region)
                prediction = self.classify_frame(frame)
                if prediction is None:
                    continue
                mass = region.mass
                # we use the square-root here as the mass is in units squared.
                # this effectively means we are giving weight based on the diameter
                # of the object rather than the mass.
                mass_weight = np.clip(mass / 20, 0.02, 1.0) ** 0.5

                # cropped frames don't do so well so restrict their score
                cropped_weight = 0.7 if region.was_cropped else 1.0
                track_prediction.classified_frame(
                    i, prediction, mass_weight * cropped_weight
                )
        return track_prediction

    def classify_cropped_data(
        self,
        track_id,
        start_frame,
        data,
        thermal_median,
        regions,
        keep_all=True,
        overlay=None,
    ):

        try:
            fp_index = self.labels.index("false-positive")
        except ValueError:
            fp_index = None
        track_prediction = TrackPrediction(
            track_id, start_frame, self.labels, keep_all=keep_all
        )

        if self.use_movement:
            predictions = self.classify_using_movement(
                data, thermal_median, regions=regions, overlay=overlay
            )
            for i, prediction in enumerate(predictions):
                track_prediction.classified_frame(i, prediction, None)
        else:
            for i, frame in enumerate(data):
                region = regions[i]
                prediction = self.classify_frame(frame)
                mass = region.mass
                # we use the square-root here as the mass is in units squared.
                # this effectively means we are giving weight based on the diameter
                # of the object rather than the mass.
                mass_weight = np.clip(mass / 20, 0.02, 1.0) ** 0.5

                # cropped frames don't do so well so restrict their score
                cropped_weight = 0.7 if region.was_cropped else 1.0
                track_prediction.classified_frame(
                    i, prediction, mass_weight * cropped_weight
                )
        return track_prediction

    def classify_using_movement(self, data, thermal_median, regions, overlay=None):
        """
        take any square_width, by square_width amount of frames and sort by
        time use as the r channel, g and b channel are the overall movment of
        the track
        """

        predictions = []

        frames_per_classify = self.square_width ** 2
        num_frames = len(data)

        # note we can use more classifications but since we are using all track
        # bounding regions with each classify for the over all movement, it
        # doesn't change the result much
        # take frames_per_classify random frames, sort by time then use this to classify

        num_classifies = math.ceil(float(num_frames) / frames_per_classify)

        # since we classify a random segment each time, take a few permutations
        combinations = max(1, frames_per_classify // 9)
        for _ in range(combinations):
            frame_sample = np.arange(num_frames)
            np.random.shuffle(frame_sample)
            for i in range(num_classifies):
                seg_frames = frame_sample[:frames_per_classify]
                segment = []
                medians = []
                # update remaining
                frame_sample = frame_sample[frames_per_classify:]
                seg_frames.sort()
                for frame_i in seg_frames:
                    f = data[frame_i]
                    segment.append(f.copy())
                    medians.append(thermal_median[i])
                frames = preprocess_movement(
                    data,
                    segment,
                    self.square_width,
                    self.frame_size,
                    regions,
                    self.preprocess_fn,
                    reference_level=medians
                    if self.params.get("subtract_median", True)
                    else None,
                    red_type=self.red_type,
                    green_type=self.green_type,
                    blue_type=self.blue_type,
                    keep_aspect=self.params.get("keep_aspect", False),
                    overlay=overlay,
                )
                if frames is None:
                    continue
                output = self.model.predict(frames[np.newaxis, :])
                predictions.append(output[0])
        return predictions


def is_keras_model(model_file):
    path, ext = os.path.splitext(model_file)
    if ext == ".pb":
        return True
    return False


def validate_model(model_file):
    path, ext = os.path.splitext(model_file)
    if ext == ".pb":
        weights_path = os.path.dirname(model_file) + "/variables/variables.index"
        if not os.path.exists(os.path.join(weights_path)):
            logging.error("No weights found named '{}'.".format(weights_path))
            return False
    elif not os.path.exists(model_file + ".meta"):
        logging.error("No model found named '{}'.".format(model_file + ".meta"))
        return False
    return True
