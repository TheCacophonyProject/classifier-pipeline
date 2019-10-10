import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import utils as saved_model_utils

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils

tensor_X = None
tensor_y = None
tensor_state_in = None
tensor_keep_prob = None
tensor_is_training = None
tensor_global_step = None


def get_tensor(sess, name):
    """
    Returns a reference to tensor by given name.
    :param name: name of tensor
    :param none_if_not_found: if true none is returned if tensor is not found otherwise an exception is thrown.
    :return: the tensor
    """
    try:
        return sess.graph.get_tensor_by_name(name + ":0")
    except Exception as e:
        raise e


def test_run(sess):
    global tensor_X, tensor_y, tensor_state_in, tensor_keep_prob, tensor_is_training, tensor_global_step
    p_frame = np.zeros((5, 48, 48), np.float32)

    state_out = get_tensor(sess, "state_out")
    tensor_state_in = get_tensor(sess, "state_in")
    tensor_X = get_tensor(sess, "X")
    tensor_y = get_tensor(sess, "y")
    prediction = get_tensor(sess, "prediction")
    tensor_keep_prob = get_tensor(sess, "keep_prob")
    tensor_is_training = get_tensor(sess, "training")
    tensor_global_step = get_tensor(sess, "global_step")
    state_shape = tensor_state_in.shape
    state = np.zeros([1, state_shape[1], state_shape[2]], dtype=np.float32)

    batch_X = p_frame[np.newaxis, np.newaxis, :]

    feed_dict = get_feed_dict(batch_X, state_in=state)
    pred, state = sess.run([prediction, state_out], feed_dict=feed_dict)
    pred = pred[0]
    print(pred)


def get_feed_dict(X, y=None, is_training=False, state_in=None):
    """
        Returns a feed dictionary for TensorFlow placeholders.
        :param X: The examples to classify
        :param y: (optional) the labels for each example
        :param is_training: (optional) boolean indicating if we are training or not.
        :param state_in: (optional) states from previous classification.  Used to maintain internal state across runs
        :return:
        """
    result = {
        tensor_X: X[:, 0:27],  # limit number of frames per segment passed to trainer
        tensor_keep_prob: 1.0,
        tensor_is_training: False,
        tensor_global_step: 0,
    }
    if y is not None:
        result[tensor_y] = y
    if state_in is not None:
        result[tensor_state_in] = state_in
    return result


def main():
    export_dir = "/home/zaza/TheCacophonyProject/classifier-pipeline/models/saved_model"
    # converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    # tflite_model = converter.convert()
    # return
    checkoint_name = "/home/zaza/TheCacophonyProject/cptv-download/train/checkpoints/training-most-recent.sav"

    with tf.Session(graph=tf.Graph()) as sess:
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(checkoint_name + ".meta")

        # return
        # return
        # path = tf.train.latest_checkpoint(checkoint_name)
        # if path is None:
        #     sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkoint_name)
        # print(tf.train.list_variables(checkoint_name))
        # test_run(sess)
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        # builder.add_meta_graph_and_variables(sess, [tf.saved_model.SERVING])
        state_out = get_tensor(sess, "state_out")
        tensor_state_in = get_tensor(sess, "state_in")
        tensor_X = get_tensor(sess, "X")
        state_out = get_tensor(sess, "state_out")
        tensor_state_in = get_tensor(sess, "state_in")
        tensor_X = get_tensor(sess, "X")
        tensor_y = get_tensor(sess, "y")
        prediction = get_tensor(sess, "prediction")
        print(prediction)
        tensor_keep_prob = get_tensor(sess, "keep_prob")
        print(tensor_keep_prob)
        tensor_is_training = get_tensor(sess, "training")
        tensor_global_step = get_tensor(sess, "global_step")
        # tensor_X = tf.reshape(tensor_X,[1, 1, 5, 48, 48])
        # tensor_X = tf.placeholder_with_default(tensor_X, shape=[1, 1, 5, 48, 48])
        # tensor_X.set_shape([1, 1, 4, 48, 48])
        print(tensor_X)

        # tensor_X = tf.identity(tensor_X, name="X")
        print(saved_model_utils.build_tensor_info(tensor_X))
        print("making signatures")
        inputs = {
            "state_in": saved_model_utils.build_tensor_info(tensor_state_in),
            # "X": saved_model_utils.build_tensor_info(tensor_X),
            "keep_prob": saved_model_utils.build_tensor_info(tensor_keep_prob),
            "is_training": saved_model_utils.build_tensor_info(tensor_is_training),
            "global_step": saved_model_utils.build_tensor_info(tensor_global_step),
        }

        outputs = {
            "state_out": saved_model_utils.build_tensor_info(state_out),
            "pred": saved_model_utils.build_tensor_info(prediction),
        }
        signature = signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=signature_constants.PREDICT_METHOD_NAME,
        )

        signature_map = {
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        }
        print("saving signatures")

        builder.add_meta_graph_and_variables(
            sess,
            tags=[tag_constants.SERVING],
            signature_def_map=signature_map,
            clear_devices=True,
        )
        builder.save()
        print("saved signatures")

    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    return


if __name__ == "__main__":
    main()
