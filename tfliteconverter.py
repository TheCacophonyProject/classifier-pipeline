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
from piclassifier.piclassifier import LiteInterpreter

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
    print("converting to tflite: ", args.model)
    model_dir = Path(args.model)
    if model_dir.is_file():
        model_dir = model_dir.parent
    lite_dir = model_dir / "tflite"

    model = tf.keras.models.load_model(str(model_dir))
    model.trainable = False

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
        print("saving model to ", lite_dir / args.tflite_name)
        lite_dir.mkdir(parents=True, exist_ok=True)
        open(lite_dir / args.tflite_name, "wb").write(tflite_model)

        meta_file = model_dir / "metadata.txt"
        if meta_file.exists():
            lite_meta = lite_dir / args.tflite_name
            lite_meta = lite_meta.with_suffix(".txt")
            shutil.copy(meta_file, lite_meta)
    elif args.freeze:
        print("saving model to", model_dir / "frozen_model")
        model.save(model_dir / "frozen_model")
        meta_file = model_dir / "metadata.txt"
        if meta_file.exists():
            frozen_meta = model_dir / "frozen_model" / "metadata.txt"
            shutil.copy(meta_file, frozen_meta)


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
        action="store_true",
        help="freeze model with weights supplied into <model path> / frozen_model",
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
    parser.add_argument(
        "--tflite_name",
        default=LITE_MODEL_NAME,
        help="Name to save converted tflite model under, also used to run model",
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
