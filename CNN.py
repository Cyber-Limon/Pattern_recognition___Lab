import tensorflow as tf
from classes import num_classes, img_size
from tensorflow.keras import layers, models, regularizers


input_shape = (*img_size, 3)


def focal_loss(y_true, y_pred, class_alphas, gamma=2.0):
    if len(y_true.shape) == 1:
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=20)

    y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)

    ce_loss = -y_true * tf.math.log(y_pred)

    p = tf.reduce_sum(y_true * y_pred, axis=-1)
    focal_weight = tf.pow(1.0 - p, gamma)
    focal_weight_expanded = tf.expand_dims(focal_weight, axis=-1)

    alpha = tf.constant(class_alphas, dtype=tf.float32)
    alpha = tf.reshape(alpha, [1, len(class_alphas)])

    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    focal_loss_value = alpha_factor * focal_weight_expanded * ce_loss

    return tf.reduce_mean(focal_loss_value)


def cnn():
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.4),

        layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.4),

        layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.4),

        layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.6),

        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.6),

        layers.Dense(num_classes, activation='softmax')])

    return model
