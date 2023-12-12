from abc import ABC, abstractmethod
import time
import json
import logging
import numpy as np
from ml_tools.hyperparams import HyperParams
from pathlib import Path
from classify.trackprediction import TrackPrediction


class Interpreter(ABC):
    def __init__(self, model_file):
        self.load_json(model_file)

    def load_json(self, filename):
        """Loads model and parameters from file."""
        filename = Path(filename)
        filename = filename.with_suffix(".json")
        logging.info("Loading metadata from %s", filename)
        metadata = json.load(open(filename, "r"))

        self.labels = metadata["labels"]
        self.params = HyperParams()
        self.params.update(metadata.get("hyperparams", {}))
        self.data_type = metadata.get("type", "thermal")

        self.mapped_labels = metadata.get("mapped_labels")
        self.label_probabilities = metadata.get("label_probabilities")
        self.preprocess_fn = self.get_preprocess_fn()

    @abstractmethod
    def shape(self):
        """Num Inputs, Prediction shape"""
        ...

    @abstractmethod
    def predict(self, frames):
        """predict"""
        ...

    def get_preprocess_fn(self):
        model_name = self.params.model_name
        if model_name == "inceptionv3":
            # no need to use tf module, if train other model types may have to add
            #  preprocess definitions
            return inc3_preprocess
        elif model_name == "wr-resnet":
            return None
        else:
            import tensorflow as tf

            if pretrained_model == "resnet":
                return tf.keras.applications.resnet.preprocess_input
            elif pretrained_model == "nasnet":
                return tf.keras.applications.nasnet.preprocess_input
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
        logging.warn(
            "pretrained model %s has no preprocessing function", pretrained_model
        )
        return None
        logging.info("No preprocess defined for %s", model_name)
        return None

    def preprocess(self, clip, track, **args):
        scale = args.get("scale", None)
        num_predictions = args.get("num_predictions", None)
        predict_from_last = args.get("predict_from_last", None)
        segment_frames = args.get("segment_frames")
        frames_per_classify = args.get("frames_per_classify", 25)
        available_frames = (
            min(len(track.bounds_history), clip.frames_kept())
            if clip.frames_kept() is not None
            else len(track.bounds_history)
        )
        if predict_from_last is not None:
            predict_from_last = min(predict_from_last, available_frames)
        # this might be a little slower as it checks some massess etc
        # but keeps it the same for all ways of classifying
        if frames_per_classify > 1:
            if predict_from_last is not None and segment_frames is None:
                logging.debug(
                    "Prediction from last available frames %s track is of length %s",
                    available_frames,
                    len(track.bounds_history),
                )
                regions = track.bounds_history[-available_frames:]
                valid_regions = 0
                if available_frames > predict_from_last:
                    # want to get rid of any blank frames
                    predict_from_last = 0
                    for i, r in enumerate(
                        reversed(track.bounds_history[-available_frames:])
                    ):
                        if r.blank:
                            continue
                        valid_regions += 1
                        predict_from_last = i + 1
                        if valid_regions >= predict_from_last:
                            break
                logging.debug(
                    "After checking blanks have predict from last %s from last available frames %s track is of length %s",
                    predict_from_last,
                    available_frames,
                    len(track.bounds_history),
                )
            frames, preprocessed, masses = self.preprocess_segments(
                clip,
                track,
                num_predictions,
                predict_from_last,
                segment_frames=segment_frames,
                dont_filter=args.get("dont_filter", False),
            )
        else:
            frames, preprocessed, masses = self.preprocess_frames(
                clip, track, num_predictions, segment_frames=segment_frames
            )
        return frames, preprocessed, masses

    def classify_track(self, clip, track, segment_frames=None):
        start = time.time()
        prediction_frames, output, masses = self.predict_track(
            clip,
            track,
            segment_frames=segment_frames,
            frames_per_classify=self.params.square_width**2,
        )
        track_prediction = TrackPrediction(track.get_id(), self.labels)
        # self.model.predict(preprocessed)
        top_score = None
        smoothed_predictions = None
        if self.params.smooth_predictions:
            masses = np.array(masses)
            top_score = np.sum(masses)
            masses = masses[:, None]
            smoothed_predictions = output * masses
        track_prediction.classified_clip(
            output,
            smoothed_predictions,
            prediction_frames,
            top_score=top_score,
        )
        track_prediction.classify_time = time.time() - start
        return track_prediction

    def predict_track(self, clip, track, **args):
        frames, preprocessed, masses = self.preprocess(clip, track, **args)
        if preprocessed is None or len(preprocessed) == 0:
            return None, None, None
        pred = self.predict(preprocessed)
        return frames, pred, masses

    def preprocess_frames(
        self,
        clip,
        track,
        max_frames=None,
        segment_frames=None,
    ):
        from ml_tools.preprocess import preprocess_single_frame, preprocess_frame

        data = []
        frames_used = []
        filtered_norm_limits = None
        if self.params.diff_norm:
            min_diff = None
            max_diff = 0
            for i, region in enumerate(reversed(track.bounds_history)):
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
                if region.blank or region.width <= 0 or region.height <= 0:
                    continue

                f.float_arrays()
                diff_frame = region.subimage(f.thermal) - region.subimage(
                    clip.background
                )
                new_max = np.amax(diff_frame)
                new_min = np.amin(diff_frame)
                if min_diff is None or new_min < min_diff:
                    min_diff = new_min
                if new_max > max_diff:
                    max_diff = new_max
            filtered_norm_limits = (min_diff, max_diff)
        for i, region in enumerate(reversed(track.bounds_history)):
            if region.blank:
                continue
            if region.width == 0 or region.height == 0:
                logging.warn(
                    "No width or height for frame %s regoin %s",
                    region.frame_number,
                    region,
                )
                continue
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
                filtered_norm_limits=filtered_norm_limits,
            )
            preprocessed = preprocess_single_frame(
                cropped_frame,
                self.params.channels,
                self.preprocess_fn,
                save_info=f"{region.frame_number} - {region}",
            )

            frames_used.append(region.frame_number)
            data.append(preprocessed)
            if max_frames is not None and len(data) >= max_frames:
                break
        return frames_used, np.array(data), [region.mass]

    def preprocess_segments(
        self,
        clip,
        track,
        max_segments=None,
        predict_from_last=None,
        segment_frames=None,
        dont_filter=False,
    ):
        from ml_tools.preprocess import preprocess_frame, preprocess_movement

        track_data = {}
        segments = track.get_segments(
            self.params.square_width**2,
            ffc_frames=[] if dont_filter else clip.ffc_frames,
            repeats=1,
            segment_frames=segment_frames,
            segment_type=self.params.segment_type,
            from_last=predict_from_last,
            max_segments=max_segments,
            dont_filter=dont_filter,
        )
        frame_indices = set()
        for segment in segments:
            frame_indices.update(set(segment.frame_indices))
        frame_indices = list(frame_indices)
        frame_indices.sort()

        # should really be over whole track buts let just do the indices we predict of
        #  seems to make little different to just doing a min max normalization
        filtered_norm_limits = None
        if self.params.diff_norm:
            min_diff = None
            max_diff = 0
            for frame_index in frame_indices:
                region = track.bounds_history[frame_index - track.start_frame]
                f = clip.get_frame(region.frame_number)
                if region.blank or region.width <= 0 or region.height <= 0:
                    continue

                f.float_arrays()
                diff_frame = region.subimage(f.thermal) - region.subimage(
                    clip.background
                )
                new_max = np.amax(diff_frame)
                new_min = np.amin(diff_frame)
                if min_diff is None or new_min < min_diff:
                    min_diff = new_min
                if new_max > max_diff:
                    max_diff = new_max
            filtered_norm_limits = (min_diff, max_diff)
        for frame_index in frame_indices:
            region = track.bounds_history[frame_index - track.start_frame]

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
                clip.background,
                clip.crop_rectangle,
                filtered_norm_limits=filtered_norm_limits,
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

    def __init__(self, model_name):
        from openvino.inference_engine import IENetwork, IECore

        super().__init__(model_name)
        model_name = Path(model_name)
        # can use to test on PC
        # device = "CPU"
        device = "MYRIAD"
        model_xml = model_name.with_suffix(".xml")
        model_bin = model_name.with_suffix(".bin")
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

    def __init__(self, model_name):
        super().__init__(model_name)

        import tflite_runtime.interpreter as tflite

        model_name = Path(model_name)
        model_name = model_name.with_suffix(".tflite")
        self.interpreter = tflite.Interpreter(str(model_name))

        self.interpreter.allocate_tensors()  # Needed before execution!

        self.output = self.interpreter.get_output_details()[
            0
        ]  # Model has single output.
        self.input = self.interpreter.get_input_details()[0]  # Model has single input.
        self.preprocess_fn = self.get_preprocess_fn()
        # inc3_preprocess

    def predict(self, input_x):
        input_x = np.float32(input_x)
        preds = []
        # only works on input of 1
        for data in input_x:
            self.interpreter.set_tensor(self.input["index"], data[np.newaxis, :])
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.output["index"])
            preds.append(pred)
        return pred

    def shape(self):
        return 1, self.input["shape"]


def inc3_preprocess(x):
    x /= 127.5
    x -= 1.0
    return x


def get_interpreter(model):
    # model_name, type = os.path.splitext(model.model_file)

    logging.info(
        "Loading %s of type %s",
        model.model_file,
        model.type,
    )

    if model.type == LiteInterpreter.TYPE:
        classifier = LiteInterpreter(model.model_file)
    elif model.type == NeuralInterpreter.TYPE:
        classifier = NeuralInterpreter(model.model_file)
    elif model.type == "RandomForest":
        from ml_tools.forestmodel import ForestModel

        classifier = ForestModel(model.model_file)
    else:
        from ml_tools.kerasmodel import KerasModel

        classifier = KerasModel()
        classifier.load_model(model.model_file, weights=model.model_weights)

    return classifier
