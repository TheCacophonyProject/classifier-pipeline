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

FEAT_LABELS = [
    "sqrt_area",
    "elongation",
    "peak_snr",
    "mean_snr",
    "fill_factor",
    "move_1",
    "rel_move_1",
    "rel_x_move_1",
    "rel_y_move_1",
    "move_3",
    "rel_move_3",
    "rel_x_move_3",
    "rel_y_move_3",
    "move_5",
    "rel_move_5",
    "rel_x_move_5",
    "rel_y_move_5",
    "max_speed",
    "min_speed",
    "avg_speed",
    "max_speed_x",
    "min_speed_x",
    "avg_speed_x",
    "max_speed_y",
    "min_speed_y",
    "avg_speed_y",
    "max_rel_speed",
    "min_rel_speed",
    "avg_rel_speed",
    "max_rel_speed_x",
    "min_rel_speed_x",
    "avg_rel_speed_x",
    "max_rel_speed_y",
    "min_rel_speed_y",
    "avg_rel_speed_y",
    "hist_diff",
]
# EXTRA_FEATURES = [
#     "speed_distance_ratio",
#     "speed_ratio",
#     "burst_min",
#     "burst_max",
#     "birst_mean",
#     "burst_chance",
#     "burst_per_frame",
#     "total frames",
# ]

EXTRA = ["avg", "std", "max", "min", "diff"]

ALL_FEATURES = []
for extra_lbl in EXTRA:
    for f in FEAT_LABELS:
        ALL_FEATURES.append(f"{extra_lbl}-{f}")
# ALL_FEATURES.extend(EXTRA_FEATURES)

important_features = [
    "std-fill_factor",
    "max-peak_snr",
    "std-move_1",
    "max-fill_factor",
    "std-hist_diff",
    "diff-hist_diff",
    "max-hist_diff",
    "min-hist_diff",
    "diff-fill_factor",
    "max-sqrt_area",
    "std-mean_snr",
    "max-min_rel_speed",
    "min-fill_factor",
    "std-rel_move_1",
    "diff-rel_x_move_1",
    "diff-move_1",
    "std-sqrt_area",
    "avg-move_3",
    "diff-elongation",
    "diff-move_5",
    "std-min_speed_x",
    "max-max_speed_x",
    "avg-max_speed_y",
    "max-elongation",
    "diff-move_3",
    "max-rel_x_move_3",
]


def feature_mask(features_used):
    feature_indexes = []
    for f in features_used:
        feature_indexes.append(ALL_FEATURES.index(f))
    feature_indexes = np.array(feature_indexes)
    return feature_indexes


class ForestModel(Interpreter):
    TYPE = "RandomForest"

    def __init__(self, model_file):
        super().__init__(model_file)
        self.model = joblib.load(model_file)
        self.buffer_length = 5
        self.features_used = self.params.get("features_used")

    def classify_track(self, clip, track, segment_frames=None):
        track_prediction = TrackPrediction(track.get_id(), self.labels)

        x = process_track(clip, track)
        if self.features_used is not None:
            f_mask = feature_mask(self.features_used)
            x = np.take(x, f_mask)
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
    background = clip.background
    frames = []
    frame_temp_median = {}
    for r in track.bounds_history:
        frame = clip.frame_buffer.get_frame(r.frame_number)
        frames.append(frame)
        frame_temp_median[r.frame_number] = np.median(frame.thermal)
    return forest_features(
        frames, background, frame_temp_median, track.bounds_history, cropped=False
    )
    # # return None
    # if len(track) <= buf_len:
    #     return None
    # for i, region in enumerate(track.bounds_history):
    #
    #     if region.blank or region.width == 0 or region.height == 0:
    #         prev_count = 0
    #         continue
    #
    #     frame = clip.frame_buffer.get_frame(region.frame_number)
    #     frame.float_arrays()
    #     t_median = np.median(frame.thermal)
    #     cropped_frame = frame.crop_by_region(region)
    #     thermal = cropped_frame.thermal.copy()
    #     f_count += 1
    #     thermal = thermal + np.median(background) - t_median
    #
    #     sub_back = region.subimage(background).copy()
    #     filtered = thermal - sub_back
    #     feature = FrameFeatures(region)
    #
    #     feature.calculate(thermal, sub_back)
    #     count_back = min(buf_len, prev_count)
    #     for i in range(count_back):
    #         prev = frame_features[-i - 1]
    #         vel = feature.cent - prev.cent
    #         feature.speed[i] = np.sqrt(np.sum(vel * vel))
    #         feature.rel_speed[i] = feature.speed[i] / feature.sqrt_area
    #         feature.rel_speed_x[i] = np.abs(vel[0]) / feature.sqrt_area
    #         feature.rel_speed_y[i] = np.abs(vel[1]) / feature.sqrt_area
    #
    #     # if count_back >= 5:
    #     # 1 / 0
    #     frame_features.append(feature)
    #     features = feature.features()
    #     prev_count += 1
    #     if maximum_features is None:
    #         maximum_features = features
    #         avg_features = features
    #         std_features = features * features
    #     else:
    #         maximum_features = np.maximum(features, maximum_features)
    #         # Aggregate
    #         avg_features += features
    #         std_features += features * features
    #
    # # Compute statistics for all tracks that have the min required duration
    # valid_counter = 0
    # N = len(track) - np.array(
    #     [0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5]
    # )  # Normalise each measure by however many samples went into it
    # avg_features /= N
    # std_features = np.sqrt(std_features / N - avg_features**2)
    # X = np.hstack(
    #     (avg_features, std_features, maximum_features, np.array([len(track)]))
    # )
    # return X


def forest_features(
    track_frames, background, frame_temp_median, regions, buf_len=5, cropped=True
):
    frame_features = []
    avg_features = None
    std_features = None
    maximum_features = None
    minimum_features = None

    f_count = 0
    prev_count = 0
    if len(track_frames) <= buf_len:
        return None

    for i, frame in enumerate(track_frames):
        region = regions[i]
        if region.blank or region.width == 0 or region.height == 0:
            prev_count = 0
            continue
        frame.float_arrays()
        feature = FrameFeatures(region)

        sub_back = region.subimage(background).copy()
        feature.calc_histogram(sub_back, frame.thermal)
        t_median = frame_temp_median[frame.frame_number]
        if cropped:
            cropped_frame = frame
        else:
            cropped_frame = frame.crop_by_region(region)
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

    def calc_histogram(self, sub_back, crop_t):
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


# normalization values for features probably not needed
std_v = [
    0.15178847,
    0.027188646,
    0.8006742,
    0.116820745,
    0.018843347,
    0.11099239,
    0.015115197,
    0.012320447,
    0.00834021,
    0.329471,
    0.04178793,
    0.0347461,
    0.022275506,
    0.5834877,
    0.07105486,
    0.060715888,
    0.036653668,
    0.39131087,
    0.09954733,
    0.25194815,
    0.3400745,
    0.07829856,
    0.21469903,
    0.19698487,
    0.05322211,
    0.12876655,
    0.0511503,
    0.013950486,
    0.033352852,
    0.044274736,
    0.011132633,
    0.02844392,
    0.02625876,
    0.007514648,
    0.017128784,
    0.111962445,
    5.1221814,
    0.3431858,
    17.55858,
    3.5914004,
    0.30473834,
    1.0399487,
    0.17141788,
    0.14244178,
    0.10674831,
    2.5097816,
    0.38684186,
    0.33887756,
    0.21738514,
    3.76918,
    0.5712618,
    0.51205426,
    0.30125138,
    3.7100444,
    0.9942686,
    2.3026998,
    3.3430483,
    0.8423652,
    2.0555134,
    1.9229096,
    0.57602894,
    1.1992636,
    0.5754329,
    0.16434622,
    0.35606363,
    0.5118486,
    0.13673806,
    0.3135709,
    0.3143866,
    0.09890821,
    0.19578817,
    0.21177165,
    7.927947,
    0.8920534,
    27.46837,
    5.548869,
    0.39970443,
    5.966005,
    1.1091628,
    0.939299,
    0.77037466,
    10.794262,
    1.8495663,
    1.6004384,
    1.251242,
    14.1369505,
    2.474899,
    2.1609187,
    1.5738066,
    14.170539,
    5.854833,
    9.569272,
    12.7385235,
    4.984896,
    8.47711,
    8.890397,
    4.285393,
    6.2739615,
    2.4881508,
    1.0854256,
    1.6666657,
    2.1699314,
    0.91762626,
    1.4487588,
    1.5917629,
    0.7473355,
    1.1073718,
    0.31953955,
    0.15178847,
    0.027188646,
    0.8006742,
    0.116820745,
    0.018843347,
    0.11099239,
    0.015115197,
    0.012320447,
    0.00834021,
    0.329471,
    0.04178793,
    0.0347461,
    0.022275506,
    0.5834877,
    0.07105486,
    0.060715888,
    0.036653668,
    0.39131087,
    0.09954733,
    0.25194815,
    0.3400745,
    0.07829856,
    0.21469903,
    0.19698487,
    0.05322211,
    0.12876655,
    0.0511503,
    0.013950486,
    0.033352852,
    0.044274736,
    0.011132633,
    0.02844392,
    0.02625876,
    0.007514648,
    0.017128784,
    0.111962445,
    7.89409,
    0.89023757,
    27.364487,
    5.5273733,
    0.4001557,
    5.9426584,
    1.1057465,
    0.9364583,
    0.76861244,
    10.711998,
    1.837606,
    1.5904051,
    1.2455189,
    13.997581,
    2.4562283,
    2.1443408,
    1.5653683,
    14.053731,
    5.8327866,
    9.494816,
    12.636312,
    4.9669333,
    8.413056,
    8.8396435,
    4.2758307,
    6.241441,
    2.471447,
    1.0821238,
    1.6556802,
    2.155183,
    0.9149555,
    1.4393982,
    1.5839542,
    0.74572027,
    1.1021901,
    0.25196075,
    781.5659,
]
mean_v = [
    0.0775348,
    0.0184866,
    0.17836875,
    0.043902412,
    0.012626628,
    0.021495642,
    0.0035725704,
    0.002592158,
    0.0017336052,
    0.05838492,
    0.00982762,
    0.007427965,
    0.00459327,
    0.09296366,
    0.015483195,
    0.012066118,
    0.0069819894,
    0.079731025,
    0.019891461,
    0.05146406,
    0.06302364,
    0.0138297,
    0.039454263,
    0.038030125,
    0.008967926,
    0.024306273,
    0.013587593,
    0.0033395255,
    0.0087557,
    0.010673989,
    0.0023517516,
    0.0066972063,
    0.0064035966,
    0.0014733237,
    0.0040744897,
    -0.10393414,
    7.435397,
    1.449636,
    21.730503,
    4.788892,
    1.0398704,
    1.019794,
    0.16331418,
    0.12718,
    0.09059019,
    2.3248448,
    0.3675765,
    0.29620314,
    0.18705869,
    3.3680613,
    0.5306535,
    0.43374568,
    0.25865704,
    3.4652042,
    0.96486276,
    2.1687937,
    2.8522239,
    0.74997395,
    1.761103,
    1.7225814,
    0.5047326,
    1.0735674,
    0.5475687,
    0.15530284,
    0.34295163,
    0.4471794,
    0.12006074,
    0.27630955,
    0.27425075,
    0.08154664,
    0.17121173,
    0.24049263,
    10.316648,
    2.1681795,
    36.265934,
    7.6294994,
    1.4859247,
    6.4596114,
    1.0668923,
    0.8410836,
    0.6629097,
    11.425764,
    1.8498229,
    1.522181,
    1.0764292,
    14.704571,
    2.382863,
    1.995958,
    1.3144724,
    14.834788,
    6.2911367,
    10.257662,
    12.623511,
    4.9804153,
    8.574546,
    8.250467,
    3.9312124,
    5.886599,
    2.4140992,
    1.0354444,
    1.6724486,
    2.0209582,
    0.8128887,
    1.3785493,
    1.3409489,
    0.632448,
    0.95295936,
    0.45117363,
    0.0775348,
    0.0184866,
    0.17836875,
    0.043902412,
    0.012626628,
    0.021495642,
    0.0035725704,
    0.002592158,
    0.0017336052,
    0.05838492,
    0.00982762,
    0.007427965,
    0.00459327,
    0.09296366,
    0.015483195,
    0.012066118,
    0.0069819894,
    0.079731025,
    0.019891461,
    0.05146406,
    0.06302364,
    0.0138297,
    0.039454263,
    0.038030125,
    0.008967926,
    0.024306273,
    0.013587593,
    0.0033395255,
    0.0087557,
    0.010673989,
    0.0023517516,
    0.0066972063,
    0.0064035966,
    0.0014733237,
    0.0040744897,
    -0.10393414,
    10.239126,
    2.1496806,
    36.087593,
    7.5856633,
    1.4732772,
    6.4381003,
    1.0633264,
    0.83849317,
    0.661174,
    11.367273,
    1.8399847,
    1.5147517,
    1.0718305,
    14.611557,
    2.3673713,
    1.9838716,
    1.3074843,
    14.754991,
    6.2712064,
    10.206222,
    12.560379,
    4.966598,
    8.535155,
    8.212366,
    3.9222562,
    5.8623166,
    2.4005082,
    1.0321054,
    1.6637,
    2.0102568,
    0.8105333,
    1.3718503,
    1.3345402,
    0.63097554,
    0.9488823,
    0.5551107,
    544.0967,
]
max_v = [
    5.266641,
    0.7295258,
    55.874355,
    12.407583,
    0.42574856,
    4.6970706,
    0.5913621,
    0.4891105,
    0.5105152,
    18.541414,
    1.8234379,
    1.3877827,
    1.3213683,
    32.52449,
    3.1560757,
    2.902113,
    2.2729917,
    17.635914,
    3.7969172,
    11.300453,
    15.681583,
    3.3487377,
    9.637456,
    11.559932,
    3.2512615,
    8.063315,
    1.7205237,
    0.5322259,
    1.1013454,
    1.53471,
    0.43452173,
    0.98158187,
    0.8223772,
    0.45946372,
    0.56265736,
    0.1218277,
    59.146835,
    6.826627,
    363.75354,
    42.90071,
    2.642128,
    15.768243,
    3.0670645,
    2.103203,
    2.4106874,
    45.111282,
    5.565148,
    4.6710477,
    5.129014,
    66.91348,
    8.009847,
    7.891436,
    5.5637903,
    54.692432,
    15.177426,
    37.247868,
    49.73333,
    13.531644,
    33.229866,
    29.094265,
    9.657692,
    18.89423,
    7.739102,
    2.4851654,
    4.6395707,
    6.7167616,
    2.0008981,
    4.0531044,
    7.1430173,
    1.902089,
    4.2391486,
    0.98569405,
    81.47396,
    9.658119,
    447.86182,
    93.05687,
    3.0322921,
    86.93836,
    14.231551,
    14.154847,
    11.425444,
    149.80586,
    24.475191,
    23.96301,
    18.465813,
    158.59717,
    25.904163,
    24.993992,
    23.303602,
    158.59717,
    86.93836,
    136.87172,
    142.558,
    80.54633,
    125.66128,
    94.24729,
    55.38941,
    70.02093,
    25.904163,
    14.231551,
    21.588293,
    24.993992,
    14.154847,
    21.145641,
    23.303602,
    11.224309,
    16.532688,
    1.0,
    5.266641,
    0.7295258,
    55.874355,
    12.407583,
    0.42574856,
    4.6970706,
    0.5913621,
    0.4891105,
    0.5105152,
    18.541414,
    1.8234379,
    1.3877827,
    1.3213683,
    32.52449,
    3.1560757,
    2.902113,
    2.2729917,
    17.635914,
    3.7969172,
    11.300453,
    15.681583,
    3.3487377,
    9.637456,
    11.559932,
    3.2512615,
    8.063315,
    1.7205237,
    0.5322259,
    1.1013454,
    1.53471,
    0.43452173,
    0.98158187,
    0.8223772,
    0.45946372,
    0.56265736,
    0.1218277,
    80.22052,
    9.651446,
    409.3049,
    80.64929,
    3.0133426,
    86.93618,
    14.130492,
    14.057236,
    11.425389,
    149.80127,
    24.474245,
    23.962605,
    17.860376,
    158.58932,
    25.218443,
    24.821898,
    22.51365,
    158.5896,
    86.93673,
    136.86754,
    142.5539,
    80.54497,
    125.65789,
    94.24673,
    55.38919,
    70.020584,
    25.22434,
    14.1322775,
    21.587366,
    24.827751,
    14.058092,
    21.14517,
    22.575365,
    11.224263,
    16.016043,
    1.2910514,
    5355.0,
]
min_v = [
    0.0024567149,
    0.00041773738,
    -98.94636,
    0.00018393743,
    0.00024931348,
    2.000577e-06,
    3.753841e-07,
    3.321249e-08,
    3.016718e-08,
    8.151944e-06,
    4.532685e-07,
    2.0593902e-08,
    8.996015e-08,
    0.0,
    0.0,
    0.0,
    0.0,
    6.9304488e-06,
    1.1758193e-06,
    6.9304488e-06,
    2.6487057e-06,
    3.985186e-09,
    2.6487057e-06,
    6.3630896e-06,
    1.402394e-07,
    4.3063815e-06,
    1.3096371e-06,
    3.076307e-07,
    7.7549294e-07,
    7.4281337e-07,
    1.0732978e-09,
    6.6022767e-07,
    5.447441e-07,
    3.00843e-08,
    2.5661197e-07,
    -0.4988542,
    2.7620313,
    1.0562624,
    1.0029786,
    0.24650556,
    0.23686479,
    0.049913026,
    0.00514635,
    0.004097767,
    0.0014722049,
    0.09457448,
    0.0094766645,
    0.0073745004,
    0.0026115745,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1413091,
    0.044131838,
    0.08782365,
    0.07706492,
    0.027799856,
    0.049964942,
    0.036259208,
    0.011387103,
    0.022686789,
    0.014638636,
    0.004568144,
    0.009061815,
    0.011063126,
    0.0030667211,
    0.006853062,
    0.0037210577,
    0.0011650543,
    0.0023222072,
    0.006621363,
    3.1884089,
    1.1166701,
    1.1562647,
    0.38648725,
    0.34201077,
    0.18428358,
    0.018387238,
    0.015434656,
    0.0065906085,
    0.21205981,
    0.023061559,
    0.01946192,
    0.009188475,
    0.0,
    0.0,
    0.0,
    0.0,
    0.25144592,
    0.16254942,
    0.17913234,
    0.20892958,
    0.11469433,
    0.17540546,
    0.10607244,
    0.05943781,
    0.0854247,
    0.028246945,
    0.01621867,
    0.019429315,
    0.025273267,
    0.015434656,
    0.018803911,
    0.010267803,
    0.006257438,
    0.008269104,
    -0.37620667,
    0.0024567149,
    0.00041773738,
    -98.94636,
    0.00018393743,
    0.00024931348,
    2.000577e-06,
    3.753841e-07,
    3.321249e-08,
    3.016718e-08,
    8.151944e-06,
    4.532685e-07,
    2.0593902e-08,
    8.996015e-08,
    0.0,
    0.0,
    0.0,
    0.0,
    6.9304488e-06,
    1.1758193e-06,
    6.9304488e-06,
    2.6487057e-06,
    3.985186e-09,
    2.6487057e-06,
    6.3630896e-06,
    1.402394e-07,
    4.3063815e-06,
    1.3096371e-06,
    3.076307e-07,
    7.7549294e-07,
    7.4281337e-07,
    1.0732978e-09,
    6.6022767e-07,
    5.447441e-07,
    3.00843e-08,
    2.5661197e-07,
    -0.4988542,
    2.9720984,
    1.0015243,
    1.1329304,
    0.3863033,
    0.33826277,
    0.184254,
    0.018384408,
    0.015368973,
    0.006587795,
    0.21168202,
    0.02302543,
    0.019347746,
    0.009186907,
    0.0,
    0.0,
    0.0,
    0.0,
    0.25080606,
    0.16252005,
    0.17895478,
    0.20215565,
    0.11449913,
    0.17524897,
    0.10597302,
    0.05943491,
    0.08538564,
    0.028185762,
    0.016215863,
    0.019412337,
    0.025034452,
    0.01540511,
    0.018684408,
    0.0102582965,
    0.006257161,
    0.00826537,
    0.033692285,
    9.0,
]
