from classes import img_size
from prepare_yolo import num_anchors, grid_size
from tensorflow.keras import layers, Model, regularizers


input_shape = (*img_size, 3)


def residual_block(x, filters):
    shortcut = x

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(alpha=0.1)(x)

    return x


def yolo():
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(2)(x)

    x = residual_block(x, 256)
    x = layers.MaxPooling2D(2)(x)

    x = residual_block(x, 512)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(1024, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(1024, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.3)(x)

    output_filters = num_anchors * 5
    outputs = layers.Conv2D(output_filters, 1, activation=None, name='yolo_output', kernel_regularizer=regularizers.l2(1e-4))(x)

    t = layers.Reshape((grid_size[0], grid_size[1], num_anchors, 5))(outputs)

    tx = layers.Activation('sigmoid')(t[..., 0])
    ty = layers.Activation('sigmoid')(t[..., 1])
    tw = t[..., 2]
    th = t[..., 3]
    conf = layers.Activation('sigmoid')(t[..., 4])

    output = layers.Concatenate(axis=-1)([tx, ty, tw, th, conf])
    outputs = layers.Reshape((grid_size[0], grid_size[1], num_anchors * 5))(output)

    model = Model(inputs, outputs, name='yolo')
    return model
