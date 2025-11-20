from YOLO import yolo
import tensorflow as tf
from evaluate_yolo import evaluate
from classes import num_classes, img_size
from prepare_yolo import grid_size, num_anchors
from prepare_yolo import prepare_yolo_dataset, grid_size
from check_dataset import download_dataset, check_dataset


if not check_dataset():
    print("--- Датасет не найден ---")
    download_dataset()
else:
    print("--- Датасет найден ---")


def loss(y_true, y_pred):
    batch_size = tf.shape(y_true)[0]
    y_true_reshaped = tf.reshape(y_true, [batch_size, *grid_size, 75])

    return tf.keras.losses.MSE(y_true_reshaped, y_pred)


def yolo_loss(y_true, y_pred):
    batch_size = tf.shape(y_true)[0]

    y_true_reshaped = tf.reshape(y_true, [batch_size, *grid_size, num_anchors, 5 + num_classes])
    y_pred_reshaped = tf.reshape(y_pred, [batch_size, *grid_size, num_anchors, 5 + num_classes])

    true_xy = y_true_reshaped[..., 0:2]
    true_wh = y_true_reshaped[..., 2:4]
    true_conf = y_true_reshaped[..., 4:5]
    true_class = y_true_reshaped[..., 5:]

    pred_xy = y_pred_reshaped[..., 0:2]
    pred_wh = y_pred_reshaped[..., 2:4]
    pred_conf = y_pred_reshaped[..., 4:5]
    pred_class = y_pred_reshaped[..., 5:]


    obj_mask = true_conf
    noobj_mask = 1 - obj_mask

    xy_loss = tf.reduce_sum(obj_mask * tf.square(true_xy - pred_xy))
    wh_loss = tf.reduce_sum(obj_mask * tf.square(tf.sqrt(tf.maximum(true_wh, 1e-8)) - tf.sqrt(tf.maximum(pred_wh, 1e-8))))
    obj_conf_loss = tf.reduce_sum(obj_mask * tf.square(true_conf - pred_conf))
    noobj_conf_loss = tf.reduce_sum(noobj_mask * tf.square(true_conf - pred_conf))
    class_loss = tf.reduce_sum(obj_mask * tf.square(true_class - pred_class))


    coord_weight = 5.0
    noobj_weight = 0.5

    total_loss = (coord_weight * (xy_loss + wh_loss) +
                  obj_conf_loss +
                  noobj_weight * noobj_conf_loss +
                  class_loss)

    total_loss = total_loss / tf.cast(batch_size, tf.float32)

    return total_loss


def main():
    (x_train, y_train), (x_test, y_test) = prepare_yolo_dataset()

    model = yolo()
    model.compile(optimizer='adam', loss=yolo_loss)
    model.summary()
    model.fit(x_train, y_train, batch_size=10, epochs=100, validation_split=0.2)
    model.evaluate(x_test, y_test)
    evaluate(model=model, test_images=x_test, test_true_boxes=y_test)

    model.save('yolo_model.keras')
    print("--- Модель YOLO сохранена ---")


if __name__ == "__main__":
    main()
