import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import shutil
import tensorflow as tf
from ml_tools.model import Model
from model_resnet import ResnetModel
from model_crnn import ModelCRNN_LQ
from model_crnn import ModelCRNN_HQ

from config.config import Config

from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY as DEFAULT_SIG_DEF,
)
from ml_tools import tools


base_dir = "/home/zaza/Cacophony/classifier-pipeline/newmodel/train/checkpoints"
model_name = "training-most-recent.sav"
saved_dir = "saved_model2"
# outName = "state_in:0"
# inputLayerName = "state_out:0"
# print("Checking outlayer", outName)
# outLayer = tf.get_default_graph().get_tensor_by_name(outName)
# print("Checking inlayer", inputLayerName)
# inputTensor = tf.get_default_graph().get_tensor_by_name(inputLayerName)
# inputs = {"state_in:0": self.state_in}
# outputs = {"state_out:0": self.state_out}

# print(tf.all_variables())
# tf.compat.v1.saved_model.simple_save(
#     self.session, "/home/zaza/Cacophony/classifier-pipeline/test_model", inputs, outputs
# )
# connect up nodes.
# self.attach_nodes()
# with tf.Graph().as_default() as graph:
#     print("GRAPH DEFS")
#     print(graph.as_graph_def())


def save_again():
    # """
    # Code from v1.6.0 of Tensorflow's label_image.py example
    # """
    # model_file = "/home/zaza/Cacophony/classifier-pipeline/models/model_hq-0.966.meta"

    # graph = tf.Graph()
    # graph_def = tf.GraphDef()

    # with tf.Session() as sess:
    #     # Restore the graph
    #     saver = tf.train.import_meta_graph(model_file)

    #     # Load weights
    #     saver.restore(
    #         sess, "/home/zaza/Cacophony/classifier-pipeline/models/model_hq-0.966"
    #     )
    # conf = Config.load_from_file()
    # if conf.train.model == ResnetModel.MODEL_NAME:
    #     model = ResnetModel([], conf.train)
    # elif conf.train.model == ModelCRNN_HQ.MODEL_NAME:
    #     model = ModelCRNN_HQ(
    #         labels=len([]), train_config=conf.train,
    #     )
    # else:
    #     model = ModelCRNN_LQ(
    #         labels=len([]), train_config=conf.train
    #     )
    # model.load(os.path.join(base_dir, model_name))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:

        saver = tf.compat.v1.train.import_meta_graph(
            os.path.join(base_dir, model_name) + ".meta", clear_devices=True
        )
        saver.restore(sess, os.path.join(base_dir, model_name))
    # with tf.Session() as sess:

    # with open(model_file, "rb") as f:
    #     graph_def.ParseFromString(f.read())
    # returns = None
    # with graph.as_default():
    #     returns = tf.import_graph_def(graph_def)

    # graph = load_graph()

        inputTensor = None
        # with tf.Session(graph=graph) as sess:
        # Read the layers
        try:
            from tensorflow.compat.v1.saved_model import simple_save
        except (ModuleNotFoundError, ImportError):
            from tensorflow.saved_model import simple_save
        # with tf.Graph().as_default() as graph:
        #     print("GRAPH DEFS")
        #     print(graph.as_graph_def())
        #     layers = [n.name for n in graph.as_graph_def().node]
        #     outName = layers.pop() + ":0"
        #     inputLayerName = layers.pop(0) + ":0"

        inName = "X:0"
        outName = "state_out:0"
        print("Checking outlayer", inName)
        inputTensor = loaded_graph.get_tensor_by_name(inName)
        print("input tensor,", inputTensor.shape)

        inputTensor.set_shape([1,5 ,48,48])
        print("input tensor,", inputTensor.shape)

        outTensor = loaded_graph.get_tensor_by_name(outName)

        # if inputTensor is None:
        #     print("Checking inlayer", inputLayerName)
        #     inputTensor = loaded_graph.get_tensor_by_name(inputLayerName)

        inputs = {inName: inputTensor}
        outputs = {outName: outTensor}
        shutil.rmtree(os.path.join(base_dir, saved_dir))
        simple_save(sess, os.path.join(base_dir, saved_dir), inputs, outputs)


# def savewith_serving():
#     builder = tf.saved_model.builder.SavedModelBuilder(export_dir)


# with tf.Session(graph=tf.Graph()) as sess:
#     builder.add_meta_graph_and_variables(
#         sess,
#         [tag_constants.SERVING],
#         signature_def_map=foo_signatures,
#         assets_collection=foo_assets,
#         strip_default_attrs=True,
#     )
# ...
# with tf.Session(graph=tf.Graph()) as sess:
#     ...
#     builder.add_meta_graph([tag_constants.SERVING], strip_default_attrs=True)
# ...
# builder.save()


def inference():
    loaded_graph = tf.Graph()
    # new_saver = tf.train.Saver()
    with tf.Session(graph=loaded_graph) as sess:
        new_saver = tf.train.import_meta_graph(
            os.path.join(base_dir, model_name) + ".meta"
        )
        new_saver.restore(sess, tf.train.latest_checkpoint(base_dir))

        # Get the tensors by their variable name
        # Note: the names of the following tensors have to be declared in your train graph for this to work. So just name them appropriately.
        # _accuracy = loaded_graph.get_tensor_by_name("accuracy:0")
        # _y = loaded_graph.get_tensor_by_name("y:0")
        # print(accuracy)
        # print(
        #     "Accuracy:", _accuracy.eval({_x: mnist.test.images, _y: mnist.test.labels})
        # )
        try:
            from tensorflow.compat.v1.saved_model import simple_save
        except (ModuleNotFoundError, ImportError):
            from tensorflow.saved_model import simple_save

        outName = "state_out:0"
# .set_shape([0, 27, 5, 48, 48])
        inputLayerName = "X:0"
        outLayer = loaded_graph.get_tensor_by_name(outName)
        x = loaded_graph.get_tensor_by_name("X:0")
        x.set_shape([1,None,5,48,48])
        print("x shape", x.shape)
        inputTensor = loaded_graph.get_tensor_by_name(inputLayerName)
        inputs = {inputLayerName: inputTensor}
        outputs = {outName: outLayer}
        simple_save(sess, os.path.join(base_dir, saved_dir), inputs, outputs)


def get_classifier():

    """
    Returns a classifier object, which is created on demand.
    This means if the ClipClassifier is copied to a new process a new Classifier instance will be created.
    """
    model = "/home/zaza/Cacophony/classifier-pipeline/models/model_hq-0.966"
    classifier = Model(train_config=False, session=tools.get_session(disable_gpu=True))
    classifier.load(model)

    return classifier


def print_tensors(pb_file):
    print("Model File: {}\n".format(pb_file))
    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name + "\t" + str(op.values()))


def convert_again():
    meta_path = "/home/zaza/Cacophony/classifier-pipeline/models/model_hq-0.966.meta"

    saver = tf.train.import_meta_graph(meta_path, clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess = tf.Session()
    saver.restore(
        sess, "/home/zaza/Cacophony/classifier-pipeline/models/model_hq-0.966"
    )
    output_node_names = "state_out"
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(",")
    )
    output_graph = "/home/zaza/Cacophony/classifier-pipeline/models/output_graph.pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    sess.close()


def convert_graph():
    meta_path = (
        "/home/zaza/Cacophony/classifier-pipeline/models/themodel/saved_model.meta"
    )

    output_node_names = ["state_out"]  # Output nodes

    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)

        # Load weights
        # saver.restore(sess,"/home/zaza/Cacophony/classifier-pipeline/models/themodel/saved_model")

        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            "/home/zaza/Cacophony/classifier-pipeline/models/themodel",
        )

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), output_node_names
        )

        # Save the frozen graph
        with open(
            "/home/zaza/Cacophony/classifier-pipeline/models/output_graph.pb", "wb"
        ) as f:
            f.write(frozen_graph_def.SerializeToString())


# def v2_save():
#     builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(base_dir, saved_dir))
#     with tf.Session(graph=tf.Graph()) as sess:
#         builder.add_meta_graph_and_variables(sess,
#                                            [tf.saved_model.tag_constants.TRAINING],
#                                            signature_def_map=foo_signatures,
#                                            assets_collection=foo_assets)
#         builder.save()

def run_model():
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
def main():
    run_model()
    tf.keras.backend.set_learning_phase(0)
    tf.compat.v1.enable_control_flow_v2()
    # tf.compat.v1.disable_eager_execution()
    # inference()
    # return
    save_again()
    # return
    # print_tensors("/home/zaza/Cacophony/classifier-pipeline/models/themodel/saved_model.pb")
    # return
    # Convert the model.
    # converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(base_dir,saved_dir))
    # tflite_model = converter.convert()
    # return

    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(base_dir, saved_dir))
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        # tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_new_converter = True

    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    return
    converter = tf.lite.TFLiteConverter.from_frozen_graph(os.path.join(base_dir, saved_dir,"saved_model.pb"),
                                                          input_arrays=['X'], 
                                                          output_arrays=['state_out','prediction'] 
    )
    tflite_model = converter.convert()
    return
    model = tf.saved_model.load(export_dir=os.path.join(base_dir, saved_dir), tags=None)
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([0, 27, 5, 48, 48])

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # converter.experimental_new_converter = True

    tflite_model = converter.convert()


if __name__ == "__main__":
    main()
