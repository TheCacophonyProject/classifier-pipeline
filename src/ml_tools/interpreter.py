from abc import ABC, abstractmethod
import time
import json
import logging
import numpy as np
from ml_tools.hyperparams import HyperParams
from pathlib import Path


class Interpreter(ABC):
    def __init__(self, model_file, run_over_network=False):
        self.model_file = Path(model_file)

        self.load_json(model_file)
        self.run_over_network = run_over_network
        self.port = 8123
        self.id = None
        self._seed = None
        self.rng = np.random.default_rng(seed=self.seed)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.rng = np.random.default_rng(seed=value)

    def load_json(self, filename):
        """Loads model and parameters from file."""
        filename = Path(filename)
        filename = filename.with_suffix(".json")
        logging.info("Loading metadata from %s", filename)
        metadata = json.load(open(filename, "r"))
        self.version = metadata.get("version", None)
        self.labels = metadata["labels"]
        self.params = HyperParams()
        self.params["remapped_labels"] = metadata.get("remapped_labels")
        self.params["excluded_labels"] = metadata.get("excluded_labels")

        self.params.update(metadata.get("hyperparams", {}))
        self.data_type = metadata.get("type", "thermal")

        self.mapped_labels = metadata.get("mapped_labels")
        self.label_probabilities = metadata.get("label_probabilities")
        self.thresholds_per_label = metadata.get("thresholds")
        self.preprocess_fn = self.get_preprocess_fn()
        self.preprocess_v2 = metadata.get("v2_preprocess", False)
        self.multi_input = metadata.get("multi_input", False)
        self.scale_thresholds = metadata.get("scale_thresholds",False)
        self.enlarge = metadata.get("enlarge",True)
        from ml_tools.interpreter import get_mappings

        parent_mappings = {}
        mappings = get_mappings()
        for l in self.labels:
            path = mappings.get(l)
            if path is None:
                parent_mappings[l] = ("all", 0)
            else:
                parents = path.split(".")
                if len(parents) == 1:
                    parent_mappings[l] = l
                    continue
                # all.mammal.bird  makes mappings for parents up to alld
                depth = 0
                prev_parent = parents[0]

                for parent in parents[1:]:
                    # print("Depth ",depth,len(parents))
                    if depth == len(parents) - 2:
                        # print("Final depth ", parent,l)
                        # choose label as it can change
                        parent_mappings[l] = (prev_parent, depth)
                    else:
                        parent_mappings[parent] = (prev_parent, depth)
                    prev_parent = parent
                    depth += 1

        self.parent_mappings = parent_mappings
    def load_training_meta(self, base_dir):
        from ml_tools.thermalwriter import MeanData

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



    def confusion_tracks(
        self, dataset, filename, threshold=0.8, thresholds_per_label=None
    ):
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        import tensorflow as tf
        from classify.trackprediction import TrackPrediction
        logging.info(
            "Calculating confusion with threshold %s saving to %s", threshold, filename
        )
        true_categories = []
        track_ids = []
        for y in dataset.map(
            lambda _, y: y,
            num_parallel_calls=tf.data.AUTOTUNE,
        ):
            true_categories.extend(y[0].numpy())
            # dataset_y[0]
            track_ids.extend(y[1].numpy())
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
        if LiteInterpreter.TYPE == "TFLite":
            y_pred = []
            for x in dataset.map(
                        lambda x, _: x,
                        num_parallel_calls=tf.data.AUTOTUNE,
                    ):
                res = self.predict(x)
                y_pred.extend(res)
            y_pred  = np.array(y_pred)
        else:
            y_pred = self.model.predict(dataset.map(
                lambda x, _: x,
                num_parallel_calls=tf.data.AUTOTUNE,
            ))  
        pred_per_track = {}
        # if self.params.multi_label:
        # predicted_categori/es = []
        # for p in y_pred:
        # predicted_categories.append(tf.where(p >= 0.8).numpy().ravel())
        # predicted_categories = np.int64(predicted_categories)

        for y, track_id, p in zip(true_categories, track_ids, y_pred):
            # if self.params.multi_label:
            #     y_max = np.argmax(y)
            # else:
            #     y_max = y
            track_pred = pred_per_track.setdefault(
                track_id, (np.nonzero(y)[0], TrackPrediction(track_id, self.labels))
            )
            track_pred[1].classified_frame(None, p, 0)
        flat_y = []
        results = []
        confidences = []
        raw_class_confidences = []
        labels = self.labels.copy()
        labels.append("None")
        totals_row = np.zeros(len(labels))

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
            covered_preds = set()
            for y in y_true:
                totals_row[y] += 1
                if y in preds:
                    idx = y
                    covered_preds.add(idx)
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
                        covered_preds.add(idx)
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

            # predicted labels with no matching true label (false positives
            # that the loop above never touched, e.g. y_true=[0], preds=[0,3])
            nothing_idx = len(labels) - 1
            for idx in preds:
                if idx not in covered_preds:
                    results.append(idx)
                    confidences.append(no_smoothing[idx])
                    flat_y.append(nothing_idx)
                    raw_class_confidences.append(no_smoothing)
                    logging.info(
                        "Extra Pred %s with no true label confs %s",
                        labels[idx],
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
            np.save(f, len(pred_per_track))
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
            figure = plot_confusion_matrix(cm, class_names=labels,totals_row=totals_row)
            fscore_file = filename.parent / f"{filename.stem}-fscore"
            plt.savefig(fscore_file.with_suffix(".png"), format="png")
            np.savez(fscore_file.with_suffix(".npz"), cm = np.vstack((cm, totals_row)),labels = labels)

        preds = results.copy()

        # set these to None
        preds[confidences < threshold] = len(labels) - 1
        cm = confusion_matrix(true_categories, preds, labels=np.arange(len(labels)))

        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=labels,totals_row=totals_row)
        out_file = filename.parent / f"{filename.stem}-{round(100*threshold)}%"
        plt.savefig(out_file.with_suffix(".png"), format="png")
        np.savez(out_file.with_suffix(".npz"), cm = np.vstack((cm, totals_row)),labels = labels)

    @abstractmethod
    def shape(self):
        """Num Inputs, Prediction shape"""
        ...

    @abstractmethod
    def predict(self, frames):
        """predict"""
        ...

    def predict_over_network(self, data):
        import requests

        headers = {"content-type": "application/octet-stream"}
        response = requests.post(
            f"http://127.0.0.1:{self.port}/predict",
            data=data.tobytes(),
            headers=headers,
        )
        predictions = np.frombuffer(response.content, dtype=np.float32)
        predictions = predictions.reshape(len(data), -1)
        return predictions

    def get_preprocess_fn(self):
        model_name = self.params.model_name
        if model_name == "inceptionv3":
            # no need to use tf module, if train other model types may have to add
            #  preprocess definitions
            return inc3_preprocess
        elif model_name in ["wr-resnet", "efficientnetv2b3"]:
            return None
        else:
            import tensorflow as tf

            if model_name == "resnet":
                return tf.keras.applications.resnet.preprocess_input
            elif model_name == "nasnet":
                return tf.keras.applications.nasnet.preprocess_input
            elif model_name == "resnetv2":
                return tf.keras.applications.resnet_v2.preprocess_input

            elif model_name == "resnet152":
                return tf.keras.applications.resnet.preprocess_input

            elif model_name == "vgg16":
                return tf.keras.applications.vgg16.preprocess_input

            elif model_name == "vgg19":
                return tf.keras.applications.vgg19.preprocess_input

            elif model_name == "mobilenet":
                return tf.keras.applications.mobilenet_v2.preprocess_input

            elif model_name == "densenet121":
                return tf.keras.applications.densenet.preprocess_input

            elif model_name == "inceptionresnetv2":
                return tf.keras.applications.inception_resnet_v2.preprocess_input
        logging.warn("pretrained model %s has no preprocessing function", model_name)
        return None


    # use when predictin as tracks are being tracked i.e not finished yet
    def preprocess_track(self, clip, track, **args):
        samples = self.frames_for_prediction(clip, track, **args)
        min_frames_for_prediction =  args.get("min_frames_for_prediction",None)
        if min_frames_for_prediction is not None and len(samples)== 1 and len(samples[0].frame_indices) < min_frames_for_prediction:
            logging.info("Not enough frames for a prediction have %s required %s", len(samples[0].frame_indices), min_frames_for_prediction)

            return None
        
        frames, preprocessed, mass  =  self.preprocess(clip, track, samples, **args)
        if preprocessed is None or len(preprocessed) == 0:
            return None
        return frames, preprocessed, mass
    
    # use when predictin as tracks are being tracked i.e not finished yet
    def predict_recent_frames(self, clip, track, **args):
        preprocessed_result = self.preprocess_track(clip,track,**args)
        if preprocessed_result is None:
            return None
        frames, preprocessed, mass  = preprocessed_result
        try:
            prediction = self.predict(preprocessed)
        except:
            logging.error("Could not predict", exc_info=True)
            return None
        return prediction, frames, mass

    def preprocess(self, clip, track, samples, **args):
        frames_per_classify = args.get("frames_per_classify", 25)

        # this might be a little slower as it checks some massess etc
        # but keeps it the same for all ways of classifying
        if frames_per_classify > 1:
            if self.preprocess_v2:
                frames, preprocessed, masses = self.preprocess_segments_v2(
                    clip,
                    track,
                    samples,
                    predict_from_last=args.get(
                        "predict_from_last"
                    ),  # only used fo mvm model, needs to be changed to use samples
                )
            else:
                frames, preprocessed, masses = self.preprocess_segments(
                    clip,
                    track,
                    samples,
                    predict_from_last=args.get(
                        "predict_from_last"
                    ),  # only used fo mvm model, needs to be changed to use samples
                )
        else:
            frames, preprocessed, masses = self.preprocess_frames(
                clip,
                track,
            )
        return frames, preprocessed, masses

    def classify_track(self, clip, track, segment_frames=None, min_segments=None):
        start = time.time()
        prediction_frames, output, masses = self.predict_track(
            clip,
            track,
            segment_frames=segment_frames,
            frames_per_classify=self.params.square_width**2,
            min_segments=min_segments,
        )
        if output is None:
            logging.info("Skipping track %s", track.get_id())
            return None
        track_pred = self.track_prediction_from_raw(
            track.get_id(), prediction_frames, output, masses
        )
        track_pred.classify_time = time.time() - start

        return track_pred

    def track_prediction_from_raw(self, track_id, prediction_frames, output, masses):
        from classify.trackprediction import TrackPrediction

        track_prediction = TrackPrediction(
            track_id,
            self.labels,
            smooth_preds=self.params.smooth_predictions,
            multi_label=self.params.multi_label,
            parent_mappings=self.parent_mappings,
            scale_thresholds=self.scale_thresholds,
            thresholds_per_label=self.thresholds_per_label
        )
        track_prediction.classified_track(
            output,
            prediction_frames,
            masses,
        )
        if (
            len(prediction_frames) == 1
            and len(set(prediction_frames[0])) < self.params.square_width**2 / 4
        ):
            # if we don't have many frames to get a good prediction, lets assume only false-positive is a good prediction and filter the rest to a maximum of 0.5
            if track_prediction.predicted_tags() != "false-positive":
                track_prediction.cap_confidences(0.5)
        return track_prediction

    def predict_track(self, clip, track, **args):
        samples = self.frames_for_prediction(clip, track, **args)
        frames, preprocessed, masses = self.preprocess(clip, track, samples, **args)
        if preprocessed is None or len(preprocessed) == 0:
            return None, None, None
        pred = self.predict(preprocessed)
        return frames, pred, masses

    def frames_for_prediction(self, clip, track, **args):
        frames_per_classify = args.get("frames_per_classify", 25)
        max_predictions = args.get("num_predictions")
        if frames_per_classify > 1:
            predict_from_last = args.get("predict_from_last", None)
            dont_filter = args.get("dont_filter", False)

            # this might be a little slower as it checks some massess etc
            # but keeps it the same for all ways of classifying
            available_frames = None
            if predict_from_last is not None:
                available_frames = (
                    min(len(track.bounds_history), clip.frames_kept())
                    if clip.frames_kept() is not None
                    else len(track.bounds_history)
                )
                predict_from_last = min(predict_from_last, available_frames)

                logging.debug(
                    "Prediction from last available frames %s track is of length %s",
                    available_frames,
                    len(track.bounds_history),
                )
            from ml_tools.datasetstructures import get_segments

            if predict_from_last == 0:
                return []
            if predict_from_last is not None:
                regions = np.array(track.bounds_history[-available_frames:])
                start_frame = regions[0].frame_number
            else:
                start_frame = track.start_frame
                regions = np.array(track.bounds_history)

            segments, _ = get_segments(
                track.clip_id,
                track._id,
                start_frame,
                segment_frame_spacing=9,
                segment_width=self.params.square_width**2,
                regions=regions,
                ffc_frames=[] if dont_filter else clip.ffc_frames,
                repeats=1,
                min_frames=1,
                segment_types=self.params.segment_types,
                from_last=predict_from_last,
                max_segments=max_predictions,
                dont_filter=dont_filter,
                min_segments=args.get("min_segments"),
                rng=self.rng,
                # min_frames = args.get("min_frames")
            )
            return segments
        else:
            max_frames = max_predictions
            frames = [
                region
                for region in track.bounds_history
                if not region.blank and region.width > 0 and region.height > 0
            ]
            if max_frames is not None and len(frames) >= max_frames:
                frames = frames[-max_frames:]
            return frames
        # to do should really return some kind of common class

    def preprocess_frames(
        self,
        clip,
        track,
        samples,
    ):
        from ml_tools.preprocess import preprocess_single_frame, preprocess_frame

        data = []
        frames_used = []
        filtered_norm_limits = None
        thermal_norm_limits = None
        if self.params.diff_norm or self.params.thermal_diff_norm:
            thermal_norm_limits, filtered_norm_limits = self.get_limits(clip, track)

        for region in samples:
            frame = clip.get_frame(region.frame_number)
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
                "classifying single frame with preprocess %s size %s crop? %s f shape %s region %s",
                "None" if self.preprocess_fn is None else self.preprocess_fn.__module__,
                self.params.frame_size,
                True,
                frame.thermal.shape,
                region,
            )
            cropped_frame = preprocess_frame(
                frame,
                (self.params.frame_size, self.params.frame_size),
                region,
                clip.background,
                clip.crop_rectangle,
                calculate_filtered=False,
                filtered_norm_limits=filtered_norm_limits,
                thermal_norm_limits=thermal_norm_limits,
            )
            preprocessed = preprocess_single_frame(
                cropped_frame,
                self.params.channels,
                self.preprocess_fn,
                save_info=f"{region.frame_number} - {region}",
            )

            frames_used.append(region.frame_number)
            data.append(preprocessed)

        return frames_used, np.array(data), [region.mass]

    def get_limits(self, clip, track):
        min_diff = None
        max_diff = 0
        thermal_max_diff = None
        thermal_min_diff = None
        thermal_norm_limits = None
        filtered_norm_limits = None
        for region in reversed(track.bounds_history):
            if region.blank:
                continue
            if region.width == 0 or region.height == 0:
                logging.warn(
                    "No width or height for frame %s regoin %s",
                    region.frame_number,
                    region,
                )
                continue
            f = clip.get_frame(region.frame_number)
            if region.blank or region.width <= 0 or region.height <= 0 or f is None:
                continue

            f.float_arrays()

            if self.params.thermal_diff_norm:
                diff_frame = f.thermal - np.median(f.thermal)
                new_max = np.amax(diff_frame)
                new_min = np.amin(diff_frame)
                if thermal_min_diff is None or new_min < thermal_min_diff:
                    thermal_min_diff = new_min
                if thermal_max_diff is None or new_max > thermal_max_diff:
                    thermal_max_diff = new_max
            if self.params.diff_norm:
                diff_frame = region.subimage(f.filtered)
                # - region.subimage(
                #     clip.background
                # )
                new_max = np.amax(diff_frame)
                new_min = np.amin(diff_frame)
                if min_diff is None or new_min < min_diff:
                    min_diff = new_min
                if new_max > max_diff:
                    max_diff = new_max

        if self.params.thermal_diff_norm:
            thermal_norm_limits = (thermal_min_diff, thermal_max_diff)

        if self.params.diff_norm:
            filtered_norm_limits = (min_diff, max_diff)
        return thermal_norm_limits, filtered_norm_limits

    def preprocess_segments_v2(self, clip, track, segments, predict_from_last=None):
        from ml_tools.preprocess import preprocess_frame_v2, preprocess_movement
        track_data = {}
        masses = []
        preprocessed = {}
        for segment in segments:
            segment_data = []
            for region in segment.regions:

                if region.frame_number in track_data:
                    cropped_frame = track_data[region.frame_number]
                    if cropped_frame is None:
                        continue
                else:
                    frame = clip.get_frame(region.frame_number)
                    result = preprocess_frame_v2(
                        frame,
                        self.params.frame_size,
                        region,
                        clip.crop_rectangle,
                        enlarge=self.enlarge,
                        new_max=255.0,
                    )
                    if result is None:
                        track_data[region.frame_number] = None
                        continue
                    cropped_frame, _, _ = result
                    track_data[region.frame_number] = cropped_frame
                segment_data.append(cropped_frame)
            input_image = preprocess_movement(
                segment_data,
                self.params.square_width,
                self.params.frame_size * 2 if self.enlarge else self.params.frame_size,
                self.params.channels,
                self.preprocess_fn,
                sample=f"Clip-{clip.get_id()}-track-{track.get_id()}",
                pad_with=0,  # dont repeat frames
            )
            if input_image is None:
                logging.warn("No frames to predict on")
                continue
            preprocessed.setdefault("input_image", []).append(input_image)

            if self.multi_input:
                input_mask = get_frame_mask(segment.frame_indices)
                preprocessed.setdefault("input_mask", []).append(input_mask)
            masses.append(segment.mass)

        if len(preprocessed) > 0:
            if not self.multi_input:
                preprocessed = np.array(preprocessed["input_image"])
            else:
                preprocessed["input_image"] = np.array(preprocessed["input_image"])
                preprocessed["input_mask"] = np.array(preprocessed["input_mask"])

        return [s.frame_indices for s in segments], preprocessed, masses

    def preprocess_segments(self, clip, track, segments, predict_from_last=None):
        from ml_tools.preprocess import preprocess_frame, preprocess_movement

        track_data = {}
        unique_regions = {}
        frame_temp_medians = {}
        clip_thermals_at_zero = True
        for segment in segments:
            for region in segment.regions:
                if region.frame_number not in unique_regions:
                    unique_regions[region.frame_number] = region
                    frame = clip.get_frame(region.frame_number)
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
                    frame_temp_medians[region.frame_number] = np.median(frame.thermal)

                    if clip_thermals_at_zero:
                        # check that we have nice values other wise allow negatives when normalizing
                        sub_thermal = region.subimage(frame.thermal)
                        sub_thermal = (
                            np.float32(sub_thermal)
                            - frame_temp_medians[region.frame_number]
                        )
                        if np.median(sub_thermal) <= 0:
                            clip_thermals_at_zero = False
        # should really be over whole track buts let just do the indices we predict of
        #  seems to make little different to just doing a min max normalization

        thermal_norm_limits = None
        filtered_norm_limits = None
        if self.params.diff_norm or self.params.thermal_diff_norm:
            thermal_norm_limits, filtered_norm_limits = self.get_limits(clip, track)

        for region in unique_regions.values():
            # for frame_index in frame_indices:
            # region = track.bounds_history[frame_index - track.start_frame]

            frame = clip.get_frame(region.frame_number)
            # filtered is calculated slightly different for tracking, set to null so preprocess can recalc it
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
            cropped_frame = preprocess_frame(
                frame,
                (self.params.frame_size, self.params.frame_size),
                region,
                None,
                clip.crop_rectangle,
                calculate_filtered=False,
                filtered_norm_limits=filtered_norm_limits,
                thermal_norm_limits=thermal_norm_limits,
                median=frame_temp_medians[region.frame_number],
                clip_thermals_at_zero=clip_thermals_at_zero,
            )
            track_data[frame.frame_number] = cropped_frame
        features = None
        if self.params.mvm:
            from ml_tools.forestmodel import process_track as forest_process_track

            features = forest_process_track(
                clip, track, normalize=True, predict_from_last=predict_from_last
            )

        preprocessed = []
        masses = []
        for segment in segments:
            segment_frames = []
            for frame_i in segment.frame_indices:
                f = track_data[frame_i]
                # probably no need to copy
                segment_frames.append(f.copy())
            frames = preprocess_movement(
                segment_frames,
                self.params.square_width,
                self.params.frame_size,
                self.params.channels,
                self.preprocess_fn,
                sample=f"Clip-{clip.get_id()}-track-{track.get_id()}",
            )
            if frames is None:
                logging.warn("No frames to predict on")
                continue
            preprocessed.append(frames)
            masses.append(segment.mass)
        preprocessed = np.array(preprocessed)
        if self.params.mvm:
            features = features[np.newaxis, :]
            features = np.repeat(features, len(preprocessed), axis=0)
            preprocessed = [preprocessed, features]

        return [s.frame_indices for s in segments], preprocessed, masses


class NeuralInterpreter(Interpreter):
    TYPE = "Neural"

    def __init__(self, model_name, load_model=True):

        super().__init__(model_name)
        if load_model:
            self.load_model()

    def load_model(self):
        from openvino.inference_engine import IENetwork, IECore

        # can use to test on PC
        # device = "CPU"
        device = "MYRIAD"
        model_xml = self.model_file.with_suffix(".xml")
        model_bin = self.model_file.with_suffix(".bin")
        ie = IECore()
        ie.set_config({}, device)
        net = ie.read_network(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))
        self.input_shape = net.input_info[self.input_blob].input_data.shape

        net.batch_size = 1
        self.exec_net = ie.load_network(network=net, device_name=device)
        self.preprocess_fn = inc3_preprocess

    def predict(self, input_x):
        if input_x is None:
            return None
        input_x = np.float32(input_x)
        channels_last = input_x.shape[-1] == 3
        if channels_last:
            input_x = np.moveaxis(input_x, 3, 1)
        res = self.exec_net.infer(inputs={self.input_blob: input_x})
        res = res[self.out_blob]
        return res

    def shape(self):
        return 1, self.input_shape


class LiteInterpreter(Interpreter):
    TYPE = "TFLite"

    def __init__(self, model_name, run_over_network=False, load_model=True):
        super().__init__(model_name, run_over_network)

        if run_over_network or not load_model:
            return
        self.load_model()

    def load_model(self):
        from ai_edge_litert.interpreter import Interpreter

        model_name = self.model_file.with_suffix(".tflite")
        self.interpreter = Interpreter(str(model_name))

        self.interpreter.allocate_tensors()  # Needed before execution!

        self.output = self.interpreter.get_output_details()[
            0
        ]  # Model has single output.

        self.input = self.interpreter.get_input_details()[0]  # Model has single input.
        self.preprocess_fn = self.get_preprocess_fn()
        self.in_idx = self.input["index"]
        self.out_idx = self.output["index"]
        # inc3_preprocess

    def predict(self, input_x):
        if self.run_over_network:
            return self.predict_over_network(np.float32(input_x))
        input_x = np.float32(input_x)
        preds = []
        # only works on input of 1
        for data in input_x:
            self.interpreter.set_tensor(self.in_idx, data[np.newaxis, :])
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.out_idx)
            preds.append(pred[0])
        return preds

    def shape(self):
        return 1, self.input["shape"]


def inc3_preprocess(x):
    x /= 127.5
    x -= 1.0
    return x


def get_interpreter_from_path(model_file, run_over_network=False, load_model=True):
    logging.info("Loading %s", model_file)

    if model_file.suffix in [".keras", ".pb"]:
        from ml_tools.kerasmodel import KerasModel

        classifier = KerasModel(run_over_network=run_over_network)
        classifier.init_model(
            model_file, load_model=load_model
        )
    elif model_file.suffix == ".tflite":
        classifier = LiteInterpreter(
            model_file, run_over_network=run_over_network, load_model=load_model
        )
    elif model_file.suffix == ".pkl":
        from ml_tools.forestmodel import ForestModel

        classifier = ForestModel(model_file)
    return classifier


def guess_type(model_file):
    model_file = Path(model_file)
    if model_file.suffix in [".keras", ".pb"]:
        from ml_tools.kerasmodel import KerasModel

        return KerasModel.TYPE
    elif model_file.suffix == ".tflite":
        return LiteInterpreter.TYPE
    elif model_file.suffix == ".pkl":
        from ml_tools.forestmodel import ForestModel

        return ForestModel.TYPE


def get_interpreter(model, run_over_network=False, load_model=True, seed=None):
    if model.type is None:
        model.type = guess_type(model.model_file)

    logging.info(
        "Loading %s type %s over network: %s",
        model.model_file,
        model.type,
        model.run_over_network,
    )

    if model.type == LiteInterpreter.TYPE:
        classifier = LiteInterpreter(model.model_file, run_over_network, load_model)
    elif model.type == NeuralInterpreter.TYPE:
        classifier = NeuralInterpreter(model.model_file, run_over_network, load_model)
    elif model.type == "RandomForest":
        from ml_tools.forestmodel import ForestModel

        classifier = ForestModel(model.model_file, load_model=load_model)
    else:
        from ml_tools.kerasmodel import KerasModel
        classifier = KerasModel(run_over_network=run_over_network)
        classifier.init_model(
            model.model_file, weights=model.model_weights, load_model=load_model
        )
    classifier.id = model.id
    classifier.port = model.port

    if seed is not None:
        classifier.seed = seed
    return classifier


def get_contours(contour_image, frame_number):
    import cv2
    from ml_tools.imageprocessing import normalize

    contour_image, stats = normalize(contour_image, new_max=255)

    image = cv2.GaussianBlur(np.uint8(contour_image), (15, 15), 0)

    flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU

    _, image = cv2.threshold(image, 0, 255, flags)
    cv2.imwrite(f"contours/contours-{frame_number}.png", image)

    contours, _ = cv2.findContours(
        np.uint8(image),
        cv2.RETR_EXTERNAL,
        # cv2.CHAIN_APPROX_SIMPLE,
        cv2.CHAIN_APPROX_TC89_L1,
    )
    if len(contours) == 0:
        return 0

    contours = sorted(contours, key=lambda c: len(c), reverse=True)

    return len(contours[0])


class ModelMeta(Interpreter):
    def __init__(self, model_name):
        super().__init__(model_name)

    def predict(self):
        raise "No predict for model meta"

    def shape(self):
        raise "No shape for model meta"


def get_frame_mask(indices):
    """
    Normalises frame intervals uniformly against a maximum inter-frame distance of 9 frames.
    """
    # this comes from the random section logic where frames are selected at intervals of 4.32 frames apart
    #  allowing for a possible missed chunk  double this
    MAX_FRAME_DIST = 9
    indices = np.float32(indices)
    num_valid = len(indices)
    frame_delta = indices[1:] - indices[:-1]
    normalised_delta = np.minimum(frame_delta / MAX_FRAME_DIST, 1.0)
    # Use -1.0 as a strict geometric flag for empty padding slots
    mask_flat = np.concatenate([[0.0], normalised_delta, np.full(25 - num_valid, -1.0)])

    mask = mask_flat.reshape(5, 5, 1)
    return mask


label_paths_dl = "https://raw.githubusercontent.com/TheCacophonyProject/cacophony-web/main/api/classifications/label_paths.json"


def dl_mappings():
    import requests

    logging.info("Downloading mappings file from %s ", label_paths_dl)
    response = requests.get(label_paths_dl)
    response.raise_for_status()

    mapping_content = response.content.decode()
    with open("label_paths.json", "w") as f:
        f.write(mapping_content)
    return mapping_content


def get_mappings():
    labels_path = Path("label_paths.json")
    if not labels_path.exists():
        print("Doesnt exist so dling")
        label_paths = json.loads(dl_mappings())
    with open("label_paths.json", "r") as f:
        label_paths = json.load(f)
    return label_paths



# from tensorflow examples
def plot_confusion_matrix(cm, class_names, title="Confusion Matrix",totals_row = None):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    import matplotlib.pyplot as plt
    import itertools
    plt.clf()
    figure = plt.figure(figsize=(16, 16))
    tick_marks = np.arange(len(class_names))

    if totals_row is not None:
        plt.imshow(np.vstack((cm,totals_row)), interpolation="nearest", cmap=plt.cm.Blues)
    else:
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    plt.title(title)
    plt.colorbar()
    
    plt.xticks(tick_marks, class_names, rotation=90)
    ylabels = []
    for i, label in enumerate(class_names):
        ylabel = f"{label} ({np.sum(cm[i])})"
        ylabels.append(ylabel)
    if totals_row is not None:
        tick_marks = np.arange(len(class_names)+1)
        ylabels.append("totals")
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
    if totals_row is not None:
        i = len(cm)
        for j,count in enumerate(totals_row):
            color = "white" if count> threshold else "black"
            plt.text(j, i, count, horizontalalignment="center", color=color)   
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure
