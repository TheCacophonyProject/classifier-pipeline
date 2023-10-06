import argparse
import os
import cv2
from pathlib import Path
import numpy as np
import json
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="open vino model to use",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="CPU",
        help="Open vino device to use (CPU/MYRIAD)",
    )
    parser.add_argument(
        "source",
        help="an image to classify",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    image = cv2.imread(args.source)
    new_image = np.zeros((160, 160, 3))
    print("image is", image.shape)
    new_image[:, :, 0] = image[:, :, 2]
    new_image[:, :, 1] = image[:, :, 2]
    new_image[:, :, 2] = image[:, :, 0]
    new_image = image
    new_image = new_image / 128 - 1
    print(new_image.dtype, np.amin(new_image), np.amax(new_image))
    # preprocessing
    interpreter = NeuralInterpreter(args.model, args.device)
    results = interpreter.predict(new_image)
    max_i = np.argmax(results)
    print("Predicted results are ", np.round(100 * results), interpreter.labels[max_i])


class NeuralInterpreter:
    def __init__(self, model_name, device="CPU"):
        from openvino.inference_engine import IENetwork, IECore

        # super().__init__(model_name)
        model_name = Path(model_name)
        # can use to test on PC
        # device = "CPU"
        # device = "MYRIAD"
        model_meta = model_name.with_suffix(".txt")
        stats = json.load(open(model_meta, "r"))
        self.labels = stats.get("labels")
        model_xml = model_name.with_suffix(".xml")
        model_bin = model_name.with_suffix(".bin")
        ie = IECore()
        ie.set_config({}, device)
        net = ie.read_network(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))
        net.batch_size = 1
        self.input_shape = net.input_info[self.input_blob].input_data.shape

        self.exec_net = ie.load_network(network=net, device_name=device)

    def predict(self, input_x):
        if input_x is None:
            return None
        input_x = np.float16(input_x)
        # rearranged_arr = np.transpose(input_x, axes=[2, 0, 1])
        input_x = np.array([[input_x]])
        s = time.time()
        res = self.exec_net.infer(inputs={self.input_blob: input_x})

        res = res[self.out_blob]
        print("time take is", time.time() - s)
        return res[0]

    def shape(self):
        return self.input_shape


if __name__ == "__main__":
    main()
