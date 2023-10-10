import tensorflow as tf


# https://arxiv.org/pdf/1605.07146.pdf
def WRResNet(X_input, depth=22, k=4):
    filters = [16, 16 * k, 32 * k, 64 * k]
    # Define the input as a tensor with shape input_shape
    # X_input = tf.keras.Input(input_shape, name="wr-input")
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


def main():
    WRResNet()


if __name__ == "__main__":
    main()
