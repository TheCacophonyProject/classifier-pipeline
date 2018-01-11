"""
Tools for visualising ML models.

"""

import numpy as np
import matplotlib.pyplot as plt

def show_saliency_map(model, X_in, y_in):
    """
    Plots saliency map for single segment via pyplot.
    :param model: the model to use to generate the saliency maps
    :param X_in: segment of shape [frames, height, width, channels]
    :param y_in: label index
    """

    X = np.asarray(X_in, dtype=np.float32)
    y = np.asarray(y_in, dtype=np.int32)

    saliency = compute_saliency_map(X[np.newaxis, :, :, :, :], y[np.newaxis], model)[0]
    saliency = saliency / (np.max(saliency) + 1e-8)

    cols = X.shape[0]

    rows = 8

    for frame in range(cols):

        # plot original image

        # plot all 4 channels plus original
        for channel in range(4):

            plt.subplot(rows, cols, (cols * channel) + frame + 1)
            frame_data = np.float32(X[frame, channel, :, :])
            if channel in [2, 3]:
                # for motion vectors it's better to use magnitude when drawing them
                frame_data = np.abs(frame_data)

            plt.imshow(frame_data, aspect='auto', vmin=-1, vmax=10)
            plt.axis('off')

            plt.subplot(rows, cols, (cols * (channel + 4)) + frame + 1)
            plt.imshow(saliency[frame, channel, :, :], vmin=0.0, vmax=1.0, cmap=plt.cm.hot, aspect='auto')
            plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.gcf().set_size_inches(cols * 3, rows * 3)
    plt.show()

def show_segment(X_in):
    """
    Displays all channels of a segment
    :param X_in: segment of shape [frames, height, width, channels]
    """

    X = np.asarray(X_in, dtype=np.float32)

    cols = X.shape[0]

    rows = 4

    for frame in range(cols):

        # plot original image

        # plot all 4 channels plus original
        for channel in range(4):

            plt.subplot(rows, cols, (cols * channel) + frame + 1)
            frame_data = np.float32(X[frame, channel, :, :])
            if channel in [2, 3]:
                # for motion vectors it's better to use magnitude when drawing them
                frame_data = np.abs(frame_data) * 5

            plt.imshow(frame_data, aspect='auto', vmin=-10, vmax=10)
            plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.gcf().set_size_inches(cols * 3, rows * 3)
    plt.show()



def compute_saliency_map(X_in, y_in, model):
    """
    Compute a class saliency map for segment using the model for image X and label y.
    :param X_in: segment of shape [batch, frames, channels, height, width]
    :param y_in: label index
    :param model: the model to use classify the segment
    :return: the saliency map of shape [frames, height, width]
    """

    global tf
    import tensorflow as tf

    correct_scores = tf.gather_nd(model.pred,
                                  tf.stack((tf.range(X_in.shape[0], dtype="int64"), model.y), axis=1))

    feed_dict = model.get_feed_dict(X_in, y_in)

    print(correct_scores.shape, model.X.shape)

    grads = tf.abs(tf.gradients(correct_scores, model.X)[0])
    saliency = model.session.run([grads], feed_dict=feed_dict)[0]

    return saliency
