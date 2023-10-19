from abc import ABC, abstractmethod

import json
import logging
import numpy as np
from ml_tools.hyperparams import HyperParams
from pathlib import Path


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

    @abstractmethod
    def shape(self):
        """Num Inputs, Prediction shape"""
        ...

    @abstractmethod
    def predict(self, frames):
        """predict"""
        ...

    def preprocess(self, clip, track, **args):
        scale = args.get("scale", None)
        num_predictions = args.get("num_predictions", None)
        predict_from_last = args.get("predict_from_last", None)
        segment_frames = args.get("segment_frames")
        frames_per_classify = args.get("frames_per_classify", 25)
        available_frames = (
            min(len(track.bounds_history), clip.frame_buffer.max_frames)
            if clip.frame_buffer.max_frames is not None
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
            )
        else:
            frames, preprocessed, masses = self.preprocess_frames(
                clip, track, num_predictions, segment_frames=segment_frames
            )
        return frames, preprocessed, masses

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
        from ml_tools.preprocess import preprocess_single_frame

        data = []
        frames_used = []

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
                "classifying single frame with preprocess %s size %s crop? %s f shape %s region %s",
                self.preprocess_fn.__module__,
                self.params.frame_size,
                crop,
                frame.thermal.shape,
                region,
            )
            preprocessed = preprocess_single_frame(
                frame,
                (
                    self.params.frame_size,
                    self.params.frame_size,
                ),
                region,
                self.preprocess_fn,
                save_info=f"{region.frame_number} - {region}",
            )

            frames_used.append(region.frame_number)
            data.append(preprocessed)
            if max_frames is not None and len(data) >= max_frames:
                break
        return frames_used, np.array(data), mass

    def preprocess_segments(
        self,
        clip,
        track,
        max_segments=None,
        predict_from_last=None,
        segment_frames=None,
    ):
        from ml_tools.preprocess import preprocess_frame, preprocess_movement

        track_data = {}
        segments = track.get_segments(
            clip.ffc_frames,
            self.params.square_width**2,
            repeats=1,
            segment_frames=segment_frames,
            segment_type=self.params.segment_type,
            from_last=predict_from_last,
            max_segments=max_segments,
        )
        frame_indices = set()
        for segment in segments:
            frame_indices.update(set(segment.frame_indices))
        frame_indices = list(frame_indices)
        frame_indices.sort()
        for frame_index in frame_indices:
            region = track.bounds_history[frame_index - track.start_frame]

            frame = clip.frame_buffer.get_frame(region.frame_number)
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
                self.params.red_type,
                self.params.green_type,
                self.params.blue_type,
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
        # input_x = np.transpose(input_x, axes=[3, 1, 2])
        # input_x = np.array([[rearranged_arr]])
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
        self.preprocess_fn = inc3_preprocess

    def predict(self, input_x):
        input_x = np.float32(input_x)

        self.interpreter.set_tensor(self.input["index"], input_x)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output["index"])
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
