import argparse
import os
import numpy as np

import shutil
import tensorflow_decision_forests as tfdf

import tensorflow as tf
from config.config import Config
import pickle
from pathlib import Path
import shutil
from ml_tools.interpreter import LiteInterpreter

MODEL_DIR = "../cptv-download/train/checkpoints"
MODEL_NAME = "training-most-recent.sav"
SAVED_DIR = "saved_model"
LITE_MODEL_NAME = "converted_model.tflite"


def run_model(args):
    model = LiteInterpreter(args.model)
    input_data = np.array(np.random.random_sample(model.shape()[1:]), dtype=np.float32)
    prediction = model.predict(input_data)
    print("model pass 1 predicted", prediction)
    input_data = np.array(np.random.random_sample(model.shape()[1:]), dtype=np.float32)
    prediction = model.predict(input_data)
    print("model pass 2 predicted", prediction)


def convert_model(args):
    print("Loading: ", args.model)
    args.model = Path(args.model)
    model_dir = args.model.parent
    lite_dir = model_dir / "tflite"
    import time

    a = time.time()
    if args.model.suffix == ".pb":
        # for some reason refuses to work with absolute path
        model = tf.keras.models.load_model(args.model.parent, compile=False)
    else:
        model = tf.keras.models.load_model(args.model, compile=False)
    print(time.time() - a, " to load model")
    # return
    model.trainable = False
    meta_file = args.model.with_suffix(".json")

    if args.weights:
        print("using weights ", args.weights)
        model.load_weights(args.weights).expect_partial()
    if args.convert:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # converter.target_spec.supported_ops = [
        #     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        #     tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        # ]
        # 8 bit ingeter
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
        print("saving model to ", out_dir / args.model.stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / args.model.stem).open("wb") as f:
            f.write(tflite_model)
        frozen_meta = out_dir / meta_file.name

    elif args.freeze or args.export:
        out_dir = Path(args.freeze)
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.export:
            model.summary()
            input_signature = get_input_sig(model)
            export_archive = tf.keras.export.ExportArchive()
            export_archive.track(model)
            export_archive.add_endpoint(
                name="predict", fn=model.call, input_signature=input_signature
            )
            export_archive.write_out(out_dir)

            print("saving model to", out_dir / "saved_model.pb")
            frozen_meta = out_dir / "saved_model.json"

        else:
            print("saving model to", out_dir / args.model.name)
            model.save(out_dir / args.model.name)
            frozen_meta = out_dir / meta_file.name

    if meta_file.exists():
        shutil.copy(meta_file, frozen_meta)


def get_input_sig(model):
    inputs = []
    for input in model.inputs:
        inputs.append(tf.TensorSpec(shape=input.shape, dtype=input.dtype))
    return inputs


def load_model(args):
    print("loading model ", args.model)
    model_dir = Path(args.model)
    if model_dir.is_file():
        model_dir = model_dir.parent
    lite_dir = model_dir / "tflite"
    print("loading", model_dir)
    model = tf.keras.models.load_model(str(model_dir))
    model.trainable = False
    if args.weights:
        print("using weights ", args.weights)
        model.load_weights(args.weights).expect_partial()
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--freeze",
        help="freeze model with weights here",
    )
    parser.add_argument(
        "-e",
        "--export",
        action="store_true",
        help="export model instead of saving",
    )
    parser.add_argument("-w", "--weights", help="Weights to use")

    parser.add_argument(
        "-c", "--convert", action="store_true", help="Convert frozen model to tflite"
    )
    parser.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="Test converted model with random data using tflite interpreter",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=MODEL_DIR,
        help="Directory where meta data of the model you want to convert is stored",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.run:
        run_model(args)
    else:
        convert_model(args)


if __name__ == "__main__":
    main()
