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
        """Prediction shape"""
        ...

    @abstractmethod
    def predict(self, frames):
        """predict"""
        ...

    def predict_track(self, clip, track, **args):
        frames, preprocessed, mass = self.preprocess(clip, track, args)
        # print("preprocess is %s", preprocessed)
        if preprocessed is None or len(preprocessed) == 0:
            return None, None, None
        pred = self.predict(np.array(preprocessed))
        return frames, pred, mass

    def preprocess(self, clip, track, args):
        if self.TYPE == "RandomForest":
            return
        last_x_frames = args.get("last_x_frames", 1)
        scale = args.get("scale", None)

        frame_ago = 0
        # get non blank frames
        regions = []
        frames = []
        for r in reversed(track.bounds_history):
            if not r.blank:
                frame = clip.frame_buffer.get_frame_ago(frame_ago)
                if frame is None:
                    break
                frame = frame
                regions.append(r)
                frames.append(frame)
                assert frame.frame_number == r.frame_number
                if len(regions) == last_x_frames:
                    break
            frame_ago += 1
        if len(frames) == 0:
            return None, None, None

        from ml_tools.preprocess import preprocess_frame

        if self.data_type == "IR":
            logging.info("Preprocess IR scale %s last_x %s", scale, last_x_frames)
            from ml_tools.preprocess import (
                preprocess_ir,
            )

            preprocessed = []
            masses = []
            for region, frame in zip(regions, frames):
                if (
                    frame is None
                    or region.width == 0
                    or region.height == 0
                    or region.blank
                ):
                    continue
                params = self.params

                pre_f = preprocess_ir(
                    frame.copy(),
                    (
                        params.frame_size,
                        params.frame_size,
                    ),
                    region=region,
                    preprocess_fn=self.preprocess_fn,
                )
                if pre_f is None:
                    continue
                preprocessed.append(pre_f)
                masses.append(1)
            return [frame.frame_number for f in frames], preprocessed, masses
        elif self.data_type == "thermal":
            from ml_tools.preprocess import preprocess_movement, preprocess_frame

            frames_per_classify = args.get("frames_per_classify", 25)
            logging.info(
                "Preprocess thermal scale %s frames_per_classify %s last_x %s",
                scale,
                frames_per_classify,
                last_x_frames,
            )

            indices = np.random.choice(
                len(regions),
                min(frames_per_classify, len(regions)),
                replace=False,
            )
            indices.sort()
            frames = np.array(frames)[indices]
            regions = np.array(regions)[indices]

            refs = []
            segment_data = []
            mass = 0
            params = self.params

            for frame, region in zip(frames, regions):
                if region.blank:
                    continue
                f = preprocess_frame(
                    frame,
                    (params.frame_size, params.frame_size),
                    region,
                    clip.background,
                    clip.crop_rectangle,
                )
                # refs.append(np.median(frame.thermal))
                # thermal_reference = np.median(frame.thermal)
                # f = frame.crop_by_region(region)
                # mass += region.mass
                # f.resize_with_aspect(
                #     (params.frame_size, params.frame_size),
                #     clip.crop_rectangle,
                #     True,
                # )
                segment_data.append(f)
            preprocessed = preprocess_movement(
                segment_data,
                params.square_width,
                params.frame_size,
                red_type=params.red_type,
                green_type=params.green_type,
                blue_type=params.blue_type,
                preprocess_fn=self.preprocess_fn,
            )
            if preprocessed is None:
                return None, None, mass
            return [f.frame_number for f in frames], [preprocessed], [mass]


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
        return self.input_shape


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
        return self.input["shape"]


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
