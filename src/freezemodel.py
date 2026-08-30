import argparse
import json
import os

# Works around https://github.com/tensorflow/tensorflow/issues/62217:
# "TypeError: this __dict__ descriptor does not support '_DictWrapper'
# objects" from wrapt's C extension during tf.function tracing (hit by
# model.export()/TFLite conversion). Must be set before tensorflow import.
os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "true")
import numpy as np

import shutil

from pathlib import Path
import shutil
from ml_tools.interpreter import LiteInterpreter

import tensorflow as tf

def run_model(args):
    model = LiteInterpreter(args.model)
    _, input_shape = model.shape()
    print("Model shape is ", input_shape)

    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    prediction = model.predict(input_data)
    print("model pass 1 predicted", prediction)
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    prediction = model.predict(input_data)
    print("model pass 2 predicted", prediction)


def convert_model(args):

    print("Loading: ", args.model)
    args.model = Path(args.model).expanduser()

    model_dir = args.model.parent
    lite_dir = model_dir / "tflite"
    import time

    a = time.time()
    if args.model.suffix == ".pb":
        # for some reason refuses to work with absolute path
        model = tf.keras.models.load_model(args.model.parent, compile=False)
    else:
        model = tf.keras.models.load_model(args.model, compile=False)

    from modelevaluate import has_activation, add_sigmoid_output

    if not has_activation(model):
        print("Added sigmoid output")
        model = add_sigmoid_output(model)
        model.summary()
    # if args.sigmoid:
    #     probabilities = tf.keras.layers.Activation("sigmoid", name="sigmoid_output")(
    #         model.output
    #     )

    #     # 5. Construct the final inference model
    #     model = tf.keras.Model(inputs=model.inputs, outputs=probabilities)
    #     print("Addied sigmoid")
    #     model.summary()
    print(time.time() - a, " to load model")
    # return
    model.trainable = False
    meta_file = args.model.with_suffix(".json")

    if args.weights:
        print("using weights ", args.weights)
        weights = Path(args.weights).expanduser()
        model.load_weights(weights)
    if args.convert:
        out_dir = Path(args.convert).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Keras 3 models break tf.lite.TFLiteConverter.from_keras_model()
        # (it relies on a legacy-Keras call-context shim that no longer
        # resolves), so export to a SavedModel dir and convert from that.
        saved_model_dir = out_dir / "saved_model_tmp"
        if saved_model_dir.exists():
            shutil.rmtree(saved_model_dir)
        export_archive = tf.keras.export.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint(
            name="serve",
            fn=model.call,
            input_signature=get_input_sig(model),
        )
        export_archive.write_out(str(saved_model_dir))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        # converter.target_spec.supported_ops = [
        #     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        #     tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        # ]
        # 8 bit ingeter
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
        shutil.rmtree(saved_model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_file = out_dir/"converted_model.tflite"
        print("saving model to ", model_file)

        with model_file.open("wb") as f:
            f.write(tflite_model)
        frozen_meta = model_file.with_suffix(".json")

    elif args.freeze or args.export:
        out_dir = Path(args.freeze).expanduser()
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

            print("saving model to", out_dir / "saved_model")
            frozen_meta = out_dir / "saved_model.json"

        else:
            print("saving model to", out_dir / "saved_model.keras")
            print(model.summary())

            model.save(out_dir / "saved_model.keras")
            frozen_meta = out_dir / "saved_model.json"

    if meta_file.exists():
        print("Copying",meta_file, " to " , frozen_meta)
        shutil.copy(meta_file, frozen_meta)

        if args.thresholds:
            thresholds = Path(args.thresholds).expanduser()
            with open(thresholds) as f:
                original_thresholds = json.load(f)
            clipped_thresholds = {
                label: 0.8 if value == 0.0 else min(max(value, 0.5), 0.8)
                for label, value in original_thresholds.items()
            }
            with open(frozen_meta) as f:
                meta = json.load(f)
            meta["thresholds"] = clipped_thresholds
            meta["original_thresholds"] = original_thresholds
            with open(frozen_meta, "w") as f:
                json.dump(meta, f, indent=4)


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

    parser.add_argument("-c", "--convert", help="Convert frozen model to tflite")
    parser.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="Test converted model with random data using tflite interpreter",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Directory where meta data of the model you want to convert is stored",
    )
    parser.add_argument(
        "-t",
        "--thresholds",
        help="JSON file of per label thresholds, values are clipped to 0.5 - 0.8 and saved into the frozen model meta data",
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
