import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import shutil
import tensorflow as tf


base_dir = "/home/zaza/Cacophony/classifier-pipeline/newmodel/train/checkpoints"
model_name = "training-most-recent.sav"
saved_dir = "saved_model2"


def freeze_model():
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:

        saver = tf.compat.v1.train.import_meta_graph(
            os.path.join(base_dir, model_name) + ".meta", clear_devices=True
        )
        saver.restore(sess, os.path.join(base_dir, model_name))

        try:
            from tensorflow.compat.v1.saved_model import simple_save
        except (ModuleNotFoundError, ImportError):
            from tensorflow.saved_model import simple_save

        inName = "X:0"
        outName = "state_out:0"

        inputTensor = loaded_graph.get_tensor_by_name(inName)
        inputTensor.set_shape([1, 5, 48, 48])
        # doing this lets us convert the model as it doesn't like unknown shapes
        outTensor = loaded_graph.get_tensor_by_name(outName)

        inputs = {inName: inputTensor}
        outputs = {outName: outTensor}

        # complains if directory isn't empty
        try:
            shutil.rmtree(os.path.join(base_dir, saved_dir))
        except:
            pass

        simple_save(sess, os.path.join(base_dir, saved_dir), inputs, outputs)


def run_model():
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    print(output_data)


def convert_model():
    converter = tf.lite.TFLiteConverter.from_saved_model(
        os.path.join(base_dir, saved_dir)
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        # tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_new_converter = True

    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)


def convert_model_concrete():

    model = tf.saved_model.load(export_dir=os.path.join(base_dir, saved_dir), tags=None)
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([0, 27, 5, 48, 48])

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        # tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_new_converter = True

    tflite_model = converter.convert()


def main():
    # run_model()
    freeze_model()
    convert_model()


if __name__ == "__main__":
    main()
