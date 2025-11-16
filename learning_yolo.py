from YOLO import yolo
import tensorflow as tf
from prepare_yolo import prepare_yolo_dataset, grid_size
from evaluate_yolo import evaluate
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


def main():
    (x_train, y_train), (x_test, y_test) = prepare_yolo_dataset()

    model = yolo()
    model.compile(optimizer='adam', loss=loss)
    model.summary()
    model.fit(x_train, y_train, batch_size=10, epochs=10, validation_split=0.2)
    model.evaluate(x_test, y_test)
    evaluate(model=model, test_images=x_test, test_true_boxes=y_test)

    model.save('yolo_model.keras')
    print("--- Модель YOLO сохранена ---")


if __name__ == "__main__":
    main()
