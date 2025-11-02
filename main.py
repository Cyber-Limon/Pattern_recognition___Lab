import tensorflow as tf
from check_dataset import download_dataset, check_dataset
from prepare_dataset import prepare_dataset
from CNN import cnn


if not check_dataset():
    print("--- Датасет не найден ---")
    download_dataset()
else:
    print("--- Датасет найден ---")


def main():
    (x_train, y_train), (x_test, y_test), class_dict = prepare_dataset()

    num_classes = len(class_dict)

    model = cnn(num_classes=num_classes)
    model.summary()
    model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2)
    print(f'Оценка модели {model.evaluate(x_test, y_test)}')

    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
