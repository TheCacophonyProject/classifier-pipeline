"""
Tools for visualising ML models.

"""

import numpy as np
import matplotlib.pyplot as plt
import itertools


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

            plt.imshow(frame_data, aspect="auto", vmin=-1, vmax=10)
            plt.axis("off")

            plt.subplot(rows, cols, (cols * (channel + 4)) + frame + 1)
            plt.imshow(
                saliency[frame, channel, :, :],
                vmin=0.0,
                vmax=1.0,
                cmap=plt.cm.hot,
                aspect="auto",
            )
            plt.axis("off")

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

            plt.imshow(frame_data, aspect="auto", vmin=-10, vmax=10)
            plt.axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.gcf().set_size_inches(cols * 3, rows * 3)
    plt.show()


def plot_confidence_by_class(predictions, true_class, labels):
    """
    Plots confidence for each class label for both correct and incorrect predictions.
    :param predictions: prediction distribution for each example. Of shape [n, classes]
    :param true_class: true class index for each example.  Of shape [n]
    :param labels: names of labels for each class.  List of strings of length [n]
    :return: a figure
    """
    class_conf_correct = []
    class_conf_incorrect = []
    for i in range(len(labels)):
        class_conf_correct.append([])
        class_conf_incorrect.append([])

    pred_class = [np.argmax(prediction) for prediction in predictions]
    pred_conf = [predictions[i][x] for i, x in enumerate(pred_class)]

    for i, conf in enumerate(pred_conf):
        if true_class[i] == pred_class[i]:
            class_conf_correct[true_class[i]].append(conf)
        else:
            class_conf_incorrect[pred_class[i]].append(conf)

    fig = plt.figure(1, figsize=(8, 6))
    ax = plt.gca()

    plt.grid(True)
    ax.set_axisbelow(True)

    y_pos = np.arange(len(labels))
    bar_width = 0.35

    values = [np.mean(class_conf_correct[i]) for i in range(len(labels))]
    r1 = plt.bar(y_pos, values, bar_width, alpha=0.9, label="Correct")

    values = [np.mean(class_conf_incorrect[i]) for i in range(len(labels))]
    r2 = plt.bar(y_pos + bar_width, values, bar_width, alpha=0.9, label="Incorrect")

    plt.xticks(y_pos, labels)
    plt.ylabel("Confidence")
    plt.title("Confidence by Class on Correct Segments.")
    ax.set_ylim([0.0, 1.0])
    plt.legend()

    plt.tight_layout()
    return fig


def fig_to_numpy(figure):
    """Converts a matplotlib figure to a numpy array."""
    figure.canvas.draw()
    data = figure.canvas.tostring_rgb()
    ncols, nrows = figure.canvas.get_width_height()
    return np.fromstring(data, dtype=np.uint8).reshape(nrows, ncols, 3)


def plot_confusion_matrix(confusion_matrix, classes):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    plt.title("Classification Confusion Matrix")

    thresh = confusion_matrix.max() / 2.0
    for i, j in itertools.product(
        range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])
    ):
        plt.text(
            j,
            i,
            "{:.2f}".format(confusion_matrix[i, j]),
            horizontalalignment="center",
            color="white" if confusion_matrix[i, j] > thresh else "black",
        )

    ax.set_xticklabels([""] + classes, rotation=45)
    ax.set_yticklabels([""] + classes)
    ax.xaxis.set_tick_params(labeltop=False, labelbottom=True)
    return fig


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

    correct_scores = tf.gather_nd(
        model.prediction,
        tf.stack((tf.range(X_in.shape[0], dtype="int64"), model.y), axis=1),
    )

    feed_dict = model.get_feed_dict(X_in, y_in)

    print(correct_scores.shape, model.X.shape)

    grads = tf.abs(tf.gradients(correct_scores, model.X)[0])
    saliency = model.session.run([grads], feed_dict=feed_dict)[0]

    return saliency
