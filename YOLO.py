from tensorflow.keras import layers, Model
from classes import get_id_classes, get_size


input_shape = (*get_size(), 3)
num_classes = len(get_id_classes())


grid_size = (14, 14)
num_anchors = 3


def yolo():
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(2)(x)

    output_filters = num_anchors * (5 + num_classes)
    outputs = layers.Conv2D(output_filters, 1, activation='sigmoid', name='yolo_output')(x)

    model = Model(inputs, outputs, name='yolo')
    return model
