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
import cv2


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


def forest_features(db, clip_id, track_id, buf_len=5):
    frame_features = []
    avg_features = None
    std_features = None
    maximum_features = None
    minimum_features = None

    f_count = 0
    prev_count = 0
    # return None
    track_frames = db.get_track(clip_id, track_id, original=False)
    background = db.get_clip_background(clip_id)
    clip_meta = db.get_clip_meta(clip_id)
    frame_temp_median = clip_meta["frame_temp_median"]
    if len(track_frames) <= buf_len:
        return None

    for i, frame in enumerate(track_frames):
        region = frame.region
        if region.blank or region.width == 0 or region.height == 0:
            prev_count = 0
            continue
        frame.float_arrays()
        feature = FrameFeatures(region)

        sub_back = region.subimage(background).copy()
        feature.histogram(sub_back, frame.thermal)
        t_median = frame_temp_median[frame.frame_number]
        cropped_frame = frame
        thermal = cropped_frame.thermal.copy()
        f_count += 1
        thermal = thermal + np.median(background) - t_median
        filtered = thermal - sub_back

        feature.calculate(thermal, sub_back)
        count_back = min(buf_len, prev_count)
        for i in range(count_back):
            prev = frame_features[-i - 1]
            vel = feature.cent - prev.cent
            feature.speed[i] = np.sqrt(np.sum(vel * vel))
            feature.rel_speed[i] = feature.speed[i] / feature.sqrt_area
            feature.rel_speed_x[i] = np.abs(vel[0]) / feature.sqrt_area
            feature.rel_speed_y[i] = np.abs(vel[1]) / feature.sqrt_area
            feature.speed_x[i] = np.abs(vel[0])
            feature.speed_y[i] = np.abs(vel[1])
        # if count_back >= 5:
        # 1 / 0
        frame_features.append(feature)
        features = feature.features()
        prev_count += 1
        if maximum_features is None:
            maximum_features = features
            minimum_features = features

            avg_features = features
            std_features = features * features
        else:
            maximum_features = np.maximum(features, maximum_features)
            for i, (new, min_f) in enumerate(zip(features, minimum_features)):
                if min_f == 0:
                    minimum_features[i] = new
                elif new != 0 and new < min_f:
                    minimum_features[i] = new
            # Aggregate
            avg_features += features
            std_features += features * features

    # Compute statistics for all tracks that have the min required duration
    valid_counter = 0
    N = f_count - np.array(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            3,
            3,
            3,
            3,
            5,
            5,
            5,
            5,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )  # Normalise each measure by however many samples went into it
    avg_features /= N
    std_features = np.sqrt(std_features / N - avg_features**2)
    diff_features = maximum_features - minimum_features

    X = np.hstack(
        (
            avg_features,
            std_features,
            maximum_features,
            minimum_features,
            diff_features,
            np.array([len(track_frames)]),
        )
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
        self.speed_x = np.zeros(buff_len)
        self.speed_y = np.zeros(buff_len)
        self.speed = np.zeros(buff_len)
        self.histogram_diff = 0

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
        non_zero = np.array([s for s in self.speed if s > 0])
        max_speed = 0
        min_speed = 0
        avg_speed = 0
        if len(non_zero) > 0:
            max_speed = np.amax(non_zero)
            min_speed = np.amin(non_zero)
            avg_speed = np.mean(non_zero)

        non_zero = np.array([s for s in self.speed_x if s > 0])
        max_speed_x = 0
        min_speed_x = 0
        avg_speed_x = 0
        if len(non_zero) > 0:
            max_speed_x = np.amax(non_zero)
            min_speed_x = np.amin(non_zero)
            avg_speed_x = np.mean(non_zero)

        non_zero = np.array([s for s in self.speed_y if s > 0])
        max_speed_y = 0
        min_speed_y = 0
        avg_speed_y = 0
        if len(non_zero) > 0:
            max_speed_y = np.amax(non_zero)
            min_speed_y = np.amin(non_zero)
            avg_speed_y = np.mean(non_zero)

        non_zero = np.array([s for s in self.rel_speed if s > 0])
        max_rel_speed = 0
        min_rel_speed = 0
        avg_rel_speed = 0
        if len(non_zero) > 0:
            max_rel_speed = np.amax(non_zero)
            min_rel_speed = np.amin(non_zero)
            avg_rel_speed = np.mean(non_zero)

        non_zero = np.array([s for s in self.rel_speed_x if s > 0])
        max_rel_speed_x = 0
        min_rel_speed_x = 0
        avg_rel_speed_x = 0
        if len(non_zero) > 0:
            max_rel_speed_x = np.amax(non_zero)
            min_rel_speed_x = np.amin(non_zero)
            avg_rel_speed_x = np.mean(non_zero)

        non_zero = np.array([s for s in self.rel_speed_y if s > 0])
        max_rel_speed_y = 0
        min_rel_speed_y = 0
        avg_rel_speed_y = 0
        if len(non_zero) > 0:
            max_rel_speed_y = np.amax(non_zero)
            min_rel_speed_y = np.amin(non_zero)
            avg_rel_speed_y = np.mean(non_zero)

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
                max_speed,
                min_speed,
                avg_speed,
                max_speed_x,
                min_speed_x,
                avg_speed_x,
                max_speed_y,
                min_speed_y,
                avg_speed_y,
                max_rel_speed,
                min_rel_speed,
                avg_rel_speed,
                max_rel_speed_x,
                min_rel_speed_x,
                avg_rel_speed_x,
                max_rel_speed_y,
                min_rel_speed_y,
                avg_rel_speed_y,
                self.histogram_diff,
            ]
        )

    def histogram(self, sub_back, crop_t):
        max_v = np.amax(sub_back)
        min_v = np.amin(sub_back)
        sub_back = (np.float32(sub_back) - min_v) / (max_v - min_v)
        max_v = np.amax(crop_t)
        min_v = np.amin(crop_t)
        crop_t = (np.float32(crop_t) - min_v) / (max_v - min_v)

        sub_back *= 255
        crop_t *= 255

        # sub_back = np.uint8(sub_back)
        # crop_t = np.uint8(crop_t)
        sub_back = sub_back[..., np.newaxis]
        crop_t = crop_t[..., np.newaxis]
        h_bins = 50
        histSize = [h_bins]
        channels = [0]

        hist_base = cv2.calcHist(
            [sub_back],
            channels,
            None,
            histSize,
            [0, 255],
            accumulate=False,
        )
        cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        hist_track = cv2.calcHist(
            [crop_t],
            channels,
            None,
            histSize,
            [0, 255],
            accumulate=False,
        )
        # print(hist_track)
        cv2.normalize(
            hist_track,
            hist_track,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
        )
        self.histogram_diff = cv2.compareHist(hist_base, hist_track, 0)


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
