import argparse
import logging
import json
import os
import numpy as np
import pickle
import sys
from pathlib import Path
import joblib

from ml_tools.interpreter import Interpreter
from classify.trackprediction import TrackPrediction


class ForestModel(Interpreter):
    TYPE = "RandomForest"

    def __init__(self, model_file):
        super().__init__(model_file)
        self.model = joblib.load(model_file)
        self.buffer_length = 5

    def classify_track(self, clip, track, segment_frames=None):
        track_prediction = TrackPrediction(track.get_id(), self.labels)

        x = process_track(clip, track)
        if x is None:
            logging.warning("Random forest could not classify track")
            return None
        x = x[np.newaxis, :]
        predictions = self.model.predict_proba(x)
        track_prediction.classified_clip(predictions, predictions * 100, [-1])
        return track_prediction

    def shape(self):
        return (1, 52)


def process_track(
    clip,
    track,
    buf_len=5,
):
    frame_features = []
    avg_features = None
    std_features = None
    maximum_features = None
    f_count = 0
    prev_count = 0
    background = clip.background
    # return None
    if len(track) <= buf_len:
        return None
    for i, region in enumerate(track.bounds_history):

        if region.blank or region.width == 0 or region.height == 0:
            prev_count = 0
            continue

        frame = clip.frame_buffer.get_frame(region.frame_number)
        frame.float_arrays()
        t_median = np.median(frame.thermal)
        cropped_frame = frame.crop_by_region(region)
        thermal = cropped_frame.thermal.copy()
        f_count += 1
        thermal = thermal + np.median(background) - t_median

        sub_back = region.subimage(background).copy()
        filtered = thermal - sub_back
        feature = FrameFeatures(region)

        feature.calculate(thermal, sub_back)
        count_back = min(buf_len, prev_count)
        for i in range(count_back):
            prev = frame_features[-i - 1]
            vel = feature.cent - prev.cent
            feature.speed[i] = np.sqrt(np.sum(vel * vel))
            feature.rel_speed[i] = feature.speed[i] / feature.sqrt_area
            feature.rel_speed_x[i] = np.abs(vel[0]) / feature.sqrt_area
            feature.rel_speed_y[i] = np.abs(vel[1]) / feature.sqrt_area

        # if count_back >= 5:
        # 1 / 0
        frame_features.append(feature)
        features = feature.features()
        prev_count += 1
        if maximum_features is None:
            maximum_features = features
            avg_features = features
            std_features = features * features
        else:
            maximum_features = np.maximum(features, maximum_features)
            # Aggregate
            avg_features += features
            std_features += features * features

    # Compute statistics for all tracks that have the min required duration
    valid_counter = 0
    N = len(track) - np.array(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5]
    )  # Normalise each measure by however many samples went into it
    avg_features /= N
    std_features = np.sqrt(std_features / N - avg_features**2)
    X = np.hstack(
        (avg_features, std_features, maximum_features, np.array([len(track)]))
    )
    return X


class FrameFeatures:
    def __init__(self, region, buff_len=5):
        # self.thermal = thermal
        self.region = region
        self.cent = None
        self.extent = None
        self.thera = None
        self.rel_speed = np.zeros(buff_len)
        self.rel_speed_x = np.zeros(buff_len)
        self.rel_speed_y = np.zeros(buff_len)
        self.speed = np.zeros(buff_len)

    def calculate(self, thermal, sub_back):
        self.thermal_max = np.amax(thermal)
        self.thermal_std = np.std(thermal)
        filtered = thermal - sub_back
        filtered = np.abs(filtered)
        f_max = filtered.max()

        if f_max > 0.0:
            filtered /= f_max

        # Calculate weighted centroid and second moments etc
        cent, extent, theta = intensity_weighted_moments(filtered, self.region)

        self.cent = cent
        self.extent = extent
        self.theta = theta
        # Instantaneous shape features
        area = np.pi * extent[0] * extent[1]
        self.sqrt_area = np.sqrt(area)
        self.elongation = extent[0] / extent[1]
        self.std_back = np.std(sub_back) + 1.0e-9

        # Instantaneous intensity features
        self.peak_snr = (self.thermal_max - np.mean(sub_back)) / self.std_back
        self.mean_snr = self.thermal_std / self.std_back
        self.fill_factor = np.sum(filtered) / area

    def features(self):
        return np.array(
            [
                self.sqrt_area,
                self.elongation,
                self.peak_snr,
                self.mean_snr,
                self.fill_factor,
                self.speed[0],
                self.rel_speed[0],
                self.rel_speed_x[0],
                self.rel_speed_y[0],
                self.speed[2],
                self.rel_speed[2],
                self.rel_speed_x[2],
                self.rel_speed_y[2],
                self.speed[4],
                self.rel_speed[4],
                self.rel_speed_x[4],
                self.rel_speed_y[4],
            ]
        )


# Find centre of mass and size/orientation of the hot spot
def intensity_weighted_moments(sub, region=None):
    tot = np.sum(sub)
    # print(tot, "using", region)
    if tot <= 0.0:
        # Zero image - replace with ones so calculations can continue
        sub = np.ones(sub.shape)
        tot = sub.size

    # Calculate weighted centroid
    Y, X = np.mgrid[0 : sub.shape[0], 0 : sub.shape[1]]
    cx = np.sum(sub * X) / tot
    cy = np.sum(sub * Y) / tot
    X = X - cx
    Y = Y - cy
    cent = np.array([region.x + cx, region.y + cy])

    # Second moments matrix
    mxx = np.sum(X * X * sub) / tot
    mxy = np.sum(X * Y * sub) / tot
    myy = np.sum(Y * Y * sub) / tot
    M = np.array([[mxx, mxy], [mxy, myy]])

    # Extent and angle
    w, v = np.linalg.eigh(M)
    w = np.abs(w)
    if w[0] < w[1]:
        w = w[::-1]
        v = v[:, ::-1]
    extent = (
        np.sqrt(w) + 0.5
    )  # Add half a pixel so that a single bright pixel has non-zero extent
    theta = np.arctan2(v[1, 0], v[0, 0])

    return cent, extent, theta
