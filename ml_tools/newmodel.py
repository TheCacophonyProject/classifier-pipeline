import os
import json
import logging
import tensorflow.keras as keras
import tensorflow as tf
import cv2
import numpy as np
from ml_tools.dataset import TrackChannels


class NewModel:
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
        self.params.update(train_config.hyper_params)
        self.preprocess_fn = None
        self.labels = None
        self.frame_size = None
        self.pretrained_model = None

    def load_model(self, dir):
        self.model = tf.keras.models.load_model(dir)
        self.load_meta(dir)

    def load_meta(self, dir):
        meta = json.load(open(os.path.join(dir, "metadata.txt"), "r"))
        self.params = meta["hyperparams"]
        self.labels = meta["labels"]
        self.pretrained_model = self.params.get("model", "resnetv2")
        self.preprocess_fn = self.get_preprocess_fn()
        self.frame_size = self.params.get("frame_size", 48)

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

    def classify_frame(self, frame, preprocess=True):
        if preprocess:
            frame = preprocess_frame(
                frame,
                (self.frame_size, self.frame_size, 3),
                self.params.get("use_thermal", True),
                augment=False,
                preprocess_fn=self.preprocess_fn,
            )
        output = self.model.predict(frame[np.newaxis, :])
        return output[0]


def reisze_cv(image, dim, interpolation=cv2.INTER_LINEAR, extra_h=0, extra_v=0):
    return cv2.resize(
        image, dsize=(dim[0] + extra_h, dim[1] + extra_v), interpolation=interpolation,
    )


def preprocess_frame(
    data, output_dim, use_thermal=True, augment=False, preprocess_fn=None
):
    if use_thermal:
        channel = TrackChannels.thermal
    else:
        channel = TrackChannels.filtered
    data = data[channel]

    # normalizes data, constrast stretch good or bad?
    percent = 0
    max = int(np.percentile(data, 100 - percent))
    min = int(np.percentile(data, percent))
    if max == min:
        return None

    data -= min
    data = data / (max - min)
    np.clip(data, a_min=0, a_max=None, out=data)

    data = data[np.newaxis, :]
    data = np.transpose(data, (1, 2, 0))
    data = np.repeat(data, output_dim[2], axis=2)
    data = reisze_cv(data, output_dim)

    # pre proce expects values in range 0-255
    if preprocess_fn:
        data = data * 255
        data = preprocess_fn(data)
    return data
