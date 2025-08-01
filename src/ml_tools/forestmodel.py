import logging
import numpy as np
from pathlib import Path
import joblib

from ml_tools.interpreter import Interpreter
from classify.trackprediction import TrackPrediction
import cv2
import time

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


def normalize(features):
    from ml_tools.featurenorms import mean_v, std_v

    features -= mean_v
    features /= std_v
    return features


class ForestModel(Interpreter):
    TYPE = "RandomForest"

    def __init__(self, model_file, data_type=None):
        super().__init__(model_file)
        model_file = Path(model_file)
        self.model = joblib.load(model_file)
        self.buffer_length = self.params.get("buffer_length", 1)
        self.features_used = self.params.get("features_used")
        self.features = self.params.get("features")
        self.mgrid = np.mgrid[:120, :160]
        # sligtly faster to reuse this mgrd

    def classify_track(
        self, clip, track, last_x_frames=None, segment_frames=None, min_segments=None
    ):

        track_prediction = TrackPrediction(track.get_id(), self.labels)
        result = self.predict_track(
            clip, track, last_x_frames=last_x_frames, normalize=True
        )
        if result is None:
            return None
        frames, predictions, masses = result
        track_prediction.classified_clip(predictions, predictions, frames, masses)
        return track_prediction

    def shape(self):
        return 1, (1, len(self.features))

    def preprocess(self):
        return

    def predict(self, x):
        return self.model.predict_proba(x)

    def predict_track(self, clip, track, **args):
        predict_from_last = args.get(
            "predict_from_last",
        )
        max_frames = args.get(
            "max_frames",
        )
        scale = args.get("scale")
        result = process_track(
            clip,
            track,
            self.mgrid,
            predict_from_last=predict_from_last,
            max_frames=max_frames,
            scale=scale,
            normalize=args.get("normalize", True),
            buf_len=self.buffer_length,
            last_frame_predicted=args.get("last_frame_predicted"),
        )
        if result is None:
            return None
        x, frames, masses = result
        if self.features_used is not None and len(self.features_used) > 0:
            f_mask = feature_mask(self.features_used)
            x = np.take(x, f_mask)

        # x = x[np.newaxis, :]
        predictions = self.model.predict_proba(x)
        # print("predictions", predictions.shape)
        return frames, predictions, masses


def process_track(
    clip,
    track,
    mgrid,
    predict_from_last=None,
    max_frames=None,
    buf_len=5,
    scale=None,
    normalize=True,
    last_frame_predicted=None,
):
    background = clip.background
    all_frames = None
    frame_temp_median = {}
    available_frames = len(clip.frame_buffer)
    if predict_from_last is None:
        bounds = track.bounds_history
        if last_frame_predicted is not None:
            last_track_frame = bounds[-1].frame_number
            bounds = bounds[-(last_track_frame - last_frame_predicted) :]

        if len(bounds) == 0:
            return None
        first_frame = bounds[0].frame_number
        last_frame = bounds[-1].frame_number
    else:
        bounds = track.bounds_history[-min(available_frames, predict_from_last) :]
        if last_frame_predicted is not None:
            last_track_frame = bounds[-1].frame_number
            bounds = bounds[-(last_track_frame - last_frame_predicted) :]
        all_frames = clip.frame_buffer.get_last_x(len(bounds))
        if len(all_frames) == 0:
            return None

        # possibility of last bound in track not being the last frame
        first_frame = all_frames[0].frame_number
        last_frame = all_frames[-1].frame_number

    indices = [
        i
        for i, region in enumerate(bounds)
        if not region.blank
        and region.width > 0
        and region.height > 0
        and region.frame_number >= first_frame
        and region.frame_number <= last_frame
    ]
    if len(indices) == 0:
        logging.info("No valid regions to predict on  track %s", track)
        return None
    logging.debug(
        "taking %s from %s with scale %s", len(bounds), len(track.bounds_history), scale
    )
    frames = []
    if clip.crop_rectangle is None:
        logging.warning("Clip has no crop rectangle")

    # iterator =enumerate(range(len(bounds)))
    if max_frames is not None:
        if len(indices) > max_frames:
            indices = np.random.choice(indices, max_frames, replace=False)
            indices.sort()

    data_bounds = np.empty(len(indices), dtype="O")

    for i, frame_i in enumerate(indices):
        region = bounds[frame_i].copy()
        data_bounds[i] = region
        if clip.crop_rectangle is not None:
            region.crop(clip.crop_rectangle)

        if all_frames is None:
            frame = clip.get_frame(region.frame_number)
        else:
            frame_index = region.frame_number - last_frame - 1
            # logging.info("Frame index is %s frame_i is %s",frame_index, frame_i)
            frame = all_frames[frame_index]
        frames.append(frame)
        frame_temp_median[region.frame_number] = np.median(frame.thermal)
        assert frame.frame_number == region.frame_number
    if scale is not None and scale != 1:
        resize = 1 / scale
        background = clip.rescaled_background(
            (int(background.shape[1] * resize), int(background.shape[0] * resize))
        )
    else:
        background = clip.background
    start = time.time()
    x, frames_used, masses = forest_features(
        frames,
        background,
        frame_temp_median,
        data_bounds,
        mgrid,
        cropped=False,
        normalize=normalize,
        buf_len=buf_len,
    )
    # logging.info("Feature time was %s",time.time()-start)
    return x, frames_used, masses


def forest_features(
    track_frames,
    background,
    frame_temp_median,
    regions,
    mgrid,
    buf_len=1,
    cropped=True,
    normalize=True,
):
    frame_features = []
    avg_features = None
    maximum_features = None
    minimum_features = None
    all_features = []
    f_count = 0
    prev_count = 0
    frames_used = []
    masses = []
    back_med = np.median(background)
    if len(track_frames) < buf_len:
        return None, None, None
    for i, frame in enumerate(track_frames):
        region = regions[i]
        if region.blank or region.width <= 0 or region.height <= 0:
            prev_count = 0
            continue
        frames_used.append(region.frame_number)
        masses.append(region.mass)
        feature = FrameFeatures(region)
        sub_back = region.subimage(background).copy()
        t_median = frame_temp_median[frame.frame_number]
        if cropped:
            cropped_frame = frame
        else:
            cropped_frame = frame.crop_by_region(region)
        thermal = cropped_frame.thermal
        # feature.calc_histogram(sub_back, thermal, normalize=normalize)

        f_count += 1

        thermal = thermal + back_med - t_median

        feature.calculate(thermal, cropped_frame.filtered, sub_back, mgrid)
        if buf_len > 1:
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

            frame_features.append(feature)
        features = feature.features()
        all_features.append(features)
        prev_count += 1
        if buf_len > 1:
            if maximum_features is None:
                maximum_features = features.copy()
                minimum_features = features.copy()

                avg_features = features.copy()
            else:
                maximum_features = np.maximum(features, maximum_features)
                non_zero = features != 0
                current_zero = minimum_features == 0
                minimum_features[current_zero] = features[current_zero]
                minimum_features[non_zero] = np.minimum(
                    minimum_features[non_zero], features[non_zero]
                )
                # Aggregate
                avg_features += features
    if f_count < buf_len:
        logging.error("Count is less than buff len %s %s", f_count, buf_len)
        return None
    # Compute statistics for all tracks that have the min required duration
    if buf_len == 1:
        return np.array(all_features), frames_used, masses
    else:
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
        std_features = np.sqrt(np.sum((all_features - avg_features) ** 2, axis=0) / N)
        diff_features = maximum_features - minimum_features
        burst_features = calculate_burst_features(frame_features, avg_features[5])

        X = np.hstack(
            (
                avg_features,
                std_features,
                maximum_features,
                minimum_features,
                diff_features,
                burst_features,
                np.array([len(track_frames)]),
            )
        )

        return X, frames_used, masses


def calculate_burst_features(frames, mean_speed):
    #

    cut_off = max(2, (1 + mean_speed))
    speed_above = len([f for f in frames if f.speed[0] > cut_off])
    speed_below = len([f for f in frames if f.speed[0] <= cut_off])

    burst_frames = 0
    burst_ratio = []
    burst_history = []
    total_birst_frames = 0
    low_speed_distance = 0
    high_speed_distance = 0
    for i, frame in enumerate(frames):
        if frame.speed[0] < cut_off:
            low_speed_distance += frame.speed[0]
        else:
            high_speed_distance += frame.speed[0]
        if i > 0:
            prev = frames[i - 1]
            if prev.speed[0] > cut_off and frame.speed[0] > cut_off:
                burst_frames += 1
            else:
                if burst_frames > 0:
                    burst_start = i - burst_frames - 1
                    if len(burst_history) > 0:
                        # length of non burst frames is from previous burst end
                        prev = burst_history[-1]
                        burst_start -= prev[0] + prev[1]
                    burst_history.append((i - burst_frames - 1, burst_frames + 1))
                    burst_ratio.append(burst_start / (burst_frames + 1))
                    total_birst_frames += burst_frames + 1
                    burst_frames = 0
    burst_ratio = np.array(burst_ratio)
    if speed_above == 0:
        speed_ratio = 0
        speed_distance_ratio = 0
    else:
        speed_distance_ratio = low_speed_distance / high_speed_distance
        speed_ratio = speed_below / speed_above

    if len(burst_ratio) == 0:
        burst_min = 0
        burst_max = 0
        burst_mean = 0
    else:
        burst_min = np.amin(burst_ratio)
        burst_max = np.amax(burst_ratio)
        burst_mean = np.mean(burst_ratio)
    burst_chance = len(burst_ratio) / len(frames)
    burst_per_frame = total_birst_frames / len(frames)
    return np.array(
        [
            speed_distance_ratio,
            speed_ratio,
            burst_min,
            burst_max,
            burst_mean,
            burst_chance,
            burst_per_frame,
        ]
    )


class FrameFeatures:
    def __init__(self, region, buff_len=5):
        # self.thermal = thermal
        self.region = region
        self.cent = None
        self.extent = None
        self.thera = None

        self.sqrt_area = None
        self.std_back = None
        self.peak_snr = None
        self.mean_snr = None
        self.fill_factor = None
        self.histogram_diff = 0
        self.thermal_min = None

        self.thermal_max = None
        self.thermal_std = None
        self.filtered_max = None
        self.filtered_std = None
        self.filtered_min = None

        self.rel_speed = np.zeros(buff_len)
        self.rel_speed_x = np.zeros(buff_len)
        self.rel_speed_y = np.zeros(buff_len)
        self.speed_x = np.zeros(buff_len)
        self.speed_y = np.zeros(buff_len)
        self.speed = np.zeros(buff_len)
        self.histogram_diff = 0

    def calculate(self, thermal, filtered, sub_back, mgrid):
        self.thermal_min = np.min(thermal)
        self.thermal_max = np.amax(thermal)
        self.thermal_std = np.std(thermal)
        # filtered = thermal - sub_back
        filtered = np.abs(filtered)
        self.filtered_max = np.amax(filtered)
        self.filtered_min = np.amin(filtered)
        self.filtered_std = np.std(filtered)

        # Calculate weighted centroid and second moments etc
        cent, extent, theta = intensity_weighted_moments(filtered, mgrid, self.region)

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
                # self.histogram_diff,
                self.thermal_max,
                self.thermal_min,
                self.thermal_std,
                self.filtered_max,
                self.filtered_min,
                self.filtered_std,
            ]
        )
        # non_zero = np.array([s for s in self.speed if s > 0])
        # max_speed = 0
        # min_speed = 0
        # avg_speed = 0
        # if len(non_zero) > 0:
        #     max_speed = np.amax(non_zero)
        #     min_speed = np.amin(non_zero)
        #     avg_speed = np.mean(non_zero)

        # non_zero = np.array([s for s in self.speed_x if s > 0])
        # max_speed_x = 0
        # min_speed_x = 0
        # avg_speed_x = 0
        # if len(non_zero) > 0:
        #     max_speed_x = np.amax(non_zero)
        #     min_speed_x = np.amin(non_zero)
        #     avg_speed_x = np.mean(non_zero)

        # non_zero = np.array([s for s in self.speed_y if s > 0])
        # max_speed_y = 0
        # min_speed_y = 0
        # avg_speed_y = 0
        # if len(non_zero) > 0:
        #     max_speed_y = np.amax(non_zero)
        #     min_speed_y = np.amin(non_zero)
        #     avg_speed_y = np.mean(non_zero)

        # non_zero = np.array([s for s in self.rel_speed if s > 0])
        # max_rel_speed = 0
        # min_rel_speed = 0
        # avg_rel_speed = 0
        # if len(non_zero) > 0:
        #     max_rel_speed = np.amax(non_zero)
        #     min_rel_speed = np.amin(non_zero)
        #     avg_rel_speed = np.mean(non_zero)

        # non_zero = np.array([s for s in self.rel_speed_x if s > 0])
        # max_rel_speed_x = 0
        # min_rel_speed_x = 0
        # avg_rel_speed_x = 0
        # if len(non_zero) > 0:
        #     max_rel_speed_x = np.amax(non_zero)
        #     min_rel_speed_x = np.amin(non_zero)
        #     avg_rel_speed_x = np.mean(non_zero)

        # non_zero = np.array([s for s in self.rel_speed_y if s > 0])
        # max_rel_speed_y = 0
        # min_rel_speed_y = 0
        # avg_rel_speed_y = 0
        # if len(non_zero) > 0:
        #     max_rel_speed_y = np.amax(non_zero)
        #     min_rel_speed_y = np.amin(non_zero)
        #     avg_rel_speed_y = np.mean(non_zero)

        # return np.array(
        #     [
        #         self.sqrt_area,
        #         self.elongation,
        #         self.peak_snr,
        #         self.mean_snr,
        #         self.fill_factor,
        #         self.speed[0],
        #         self.rel_speed[0],
        #         self.rel_speed_x[0],
        #         self.rel_speed_y[0],
        #         self.speed[2],
        #         self.rel_speed[2],
        #         self.rel_speed_x[2],
        #         self.rel_speed_y[2],
        #         self.speed[4],
        #         self.rel_speed[4],
        #         self.rel_speed_x[4],
        #         self.rel_speed_y[4],
        #         max_speed,
        #         min_speed,
        #         avg_speed,
        #         max_speed_x,
        #         min_speed_x,
        #         avg_speed_x,
        #         max_speed_y,
        #         min_speed_y,
        #         avg_speed_y,
        #         max_rel_speed,
        #         min_rel_speed,
        #         avg_rel_speed,
        #         max_rel_speed_x,
        #         min_rel_speed_x,
        #         avg_rel_speed_x,
        #         max_rel_speed_y,
        #         min_rel_speed_y,
        #         avg_rel_speed_y,
        #         self.histogram_diff,
        #     ]
        # )

    def calc_histogram(self, sub_back, crop_t, normalize=False):
        if normalize:
            max_v = np.amax(sub_back)
            min_v = np.amin(sub_back)
            sub_back = (np.float32(sub_back) - min_v) / (max_v - min_v)
            max_v = np.amax(crop_t)
            min_v = np.amin(crop_t)
            crop_t = (np.float32(crop_t) - min_v) / (max_v - min_v)

            sub_back *= 255
            crop_t *= 255
        assert crop_t.shape == sub_back.shape
        # sub_back = np.uint8(sub_back)
        # crop_t = np.uint8(crop_t)
        sub_back = sub_back[..., np.newaxis]
        crop_t = crop_t[..., np.newaxis]
        h_bins = 60
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
        cv2.normalize(
            hist_track,
            hist_track,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
        )
        self.histogram_diff = cv2.compareHist(hist_base, hist_track, 0)


# Find centre of mass and size/orientation of the hot spot
def intensity_weighted_moments(sub, mgrid, region=None):
    tot = np.sum(sub)
    # print(tot, "using", region)
    if tot <= 0.0:
        # Zero image - replace with ones so calculations can continue
        sub = np.ones(sub.shape)
        tot = sub.size
        # surely all zeros in this case

    # Calculate weighted centroid
    Y = mgrid[0][: sub.shape[0], : sub.shape[1]]
    X = mgrid[1][: sub.shape[0], : sub.shape[1]]
    # Y, X = np.mgrid[0 : sub.shape[0], 0 : sub.shape[1]]

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
