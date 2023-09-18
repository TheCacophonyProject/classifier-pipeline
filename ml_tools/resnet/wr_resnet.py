import tensorflow as tf


# https://arxiv.org/pdf/1605.07146.pdf
def WRResNet(input_shape=(128, 512, 1), depth=22, k=4):
    filters = [16, 16 * k, 32 * k, 64 * k]
    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.Input(input_shape)
    n = int((depth - 4) / 6)
    for stage, f in enumerate(filters):
        if stage == 0:
            X = tf.keras.layers.Conv2D(
                f,
                (3, 3),
                strides=1,
                padding="same",
                name=f"conv1_{stage+1}",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
            )(X_input)
        else:
            X = wr_block(
                X, 3, (f, f), stage=stage + 1, block="b", stride=stage, depth=n
            )
    #
    X = tf.keras.layers.BatchNormalization(axis=3, name="final_bn")(X)
    X = tf.keras.layers.Activation("relu")(X)
    # LME?
    # X = tf.keras.layers.GlobalAveragePooling2D()(X)

    # X = tf.keras.layers.Flatten()(X)
    # X = tf.keras.layers.Dense(classes, activation="sigmoid", name="prediction")(X)
    model = tf.keras.Model(inputs=X_input, outputs=X, name="WRResNet")
    return model


def wr_block(X, f, filters, stage, block, stride=1, depth=1):
    s_block = f"{block}0"

    X = basic_block(X, f, filters, stage, s_block, stride)
    for d in range(depth - 1):
        s_block = f"{block}{d+1}"
        X = basic_block(X, f, filters, stage, s_block, 1)
    return X


def basic_block(X, f, filters, stage, block, stride=1):
    # defining name basis
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve Filters
    F1, F2 = filters
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    # First component of main path
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        filters=F1,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding="same",
        name=conv_name_base + "2a",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)

    X = tf.keras.layers.Dropout(rate=0.1)(X)
    # , training=istraining_ph)

    # Second component of main path
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        filters=F2,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        name=conv_name_base + "2b",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)
    if X.shape[-1] == X_shortcut.shape[-1]:
        X_shortcut = tf.keras.layers.Identity()(X_shortcut)
    else:
        X_shortcut = tf.keras.layers.Conv2D(
            X.shape[-1], strides=(stride, stride), kernel_size=1
        )(X_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation("relu")(X)
    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        F1,
        (1, 1),
        strides=(s, s),
        name=conv_name_base + "2a",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)

    # Second component of main path
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        F2,
        (f, f),
        strides=(1, 1),
        padding="same",
        name=conv_name_base + "2b",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)

    ##### SHORTCUT PATH ####
    X_shortcut = tf.keras.layers.Conv2D(
        F3,
        (1, 1),
        strides=(s, s),
        padding="valid",
        name=conv_name_base + "1",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation("relu")(X)

    return X


# https://arxiv.org/pdf/1812.01187.pdf birdnet used also
def basic_block_tweaked(X, f, filters, stage, block, stride=1):
    # defining name basis
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve Filters
    F1, F2 = filters

    # Save the input value. You'll need this later to add back to the main path.

    X_shortcut = X
    # First component of main path
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2a0")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        strides=1,
        padding="same",
        name=conv_name_base + "2a0",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)

    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        filters=F1,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        name=conv_name_base + "21",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)

    X = tf.keras.layers.Dropout(rate=0.1)(X)
    # , training=istraining_ph)

    # Second component of main path
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        filters=F2,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        name=conv_name_base + "2b",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)

    X_shortcut = tf.keras.layers.AveragePooling2D(pool_size=stride, strides=stride)(
        X_shortcut
    )

    X_shortcut = tf.keras.layers.Conv2D(X.shape[-1], strides=1, kernel_size=1)(
        X_shortcut
    )
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation("relu")(X)
    return X


def main():
    WRResNet()


if __name__ == "__main__":
    main()
