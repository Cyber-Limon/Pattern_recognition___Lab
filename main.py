import tensorflow as tf
from CNN import cnn
from prepare_dataset import prepare_dataset
from check_dataset import download_dataset, check_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator



if not check_dataset():
    print("--- Датасет не найден ---")
    download_dataset()
else:
    print("--- Датасет найден ---")


def main():
    (x_train, y_train), (x_test, y_test), id_classes = prepare_dataset()

    """datagen = ImageDataGenerator(rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True,
                                 zoom_range=0.2)"""

    num_classes = len(id_classes)

    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    model = cnn(num_classes=num_classes)
    model.summary()
    model.fit(x_train, y_train, batch_size=10, epochs=15, validation_split=0.2)
    model.evaluate(x_test, y_test)

    model.save('cnn_model.keras')
    print("--- Модель CNN сохранена ---")

    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
