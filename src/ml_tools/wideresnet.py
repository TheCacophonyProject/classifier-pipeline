import tensorflow as tf
from tensorflow.keras import layers, models


def wrn_block(x, filters, stride):
    # Standard Wide ResNet Residual block with Pre-Activation
    shortcut = x

    # First Conv
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(
        x
    )

    # Second Conv
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3, 3), strides=1, padding="same", use_bias=False)(x)

    # Projection shortcut if dimensions or strides change
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, (1, 1), strides=stride, padding="same", use_bias=False
        )(shortcut)

    return layers.add([x, shortcut])


def wide_resnet_backbone(inputs, depth=16, k=2):
    """
    WRN-depth-k feature extractor, returns the final feature map (no pooling or head)
    depth = 6n + 4 where n is the number of blocks per group
    k = Width Multiplier (e.g., k=2 makes it twice as wide as standard ResNet)
    """
    assert (depth - 4) % 6 == 0, "Depth must be 6n + 4"
    n = (depth - 4) // 6

    # Stem: 1x Stride (Keeps spatial details high!)
    x = layers.Conv2D(16, (3, 3), strides=1, padding="same", use_bias=False)(inputs)

    # Block Groups at stride 1, 2, 2 -> 4x total downsampling
    # e.g. 160x160 -> 160x160 -> 80x80 -> 40x40 so each 32x32 tile becomes 8x8
    for filters, stride in [(16 * k, 1), (32 * k, 2), (64 * k, 2)]:
        for b in range(n):
            x = wrn_block(x, filters, stride if b == 0 else 1)

    # Final Representation
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return models.Model(inputs, x, name="wideresnet")


def create_wide_resnet(input_shape=(160, 160, 3), depth=16, k=2, num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = wide_resnet_backbone(inputs, depth=depth, k=k).output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


if __name__ == "__main__":
    # Build a WRN-16-2 variant
    model = create_wide_resnet(input_shape=(160, 160, 3), depth=16, k=2)
    model.summary()
