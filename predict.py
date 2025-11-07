import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from prepare_dataset import get_id_classes, get_size


def load_model():
    try:
        model = tf.keras.models.load_model('cnn_model.keras')
        return model
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None


def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return False

    image = cv2.imread(image_path)
    image = cv2.resize(image, get_size())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image


def predict_aircraft(model, image_path, confidence=0.7):
    image = load_image(image_path)
    predictions = model.predict(image, verbose=0)
    class_id = np.argmax(predictions[0])

    if np.max(predictions[0]) < confidence:
        return "Самолет не обнаружен", np.max(predictions[0]), image[0]

    class_name = None
    for name, idx in get_id_classes().items():
        if idx == class_id:
            class_name = name
            break

    return class_name, np.max(predictions[0]), image[0]


def show(image, class_name, confidence):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(f"Класс: {class_name}\nТочность: {confidence:.2%}", fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    model = load_model()
    if model is None:
        print("--- Модель не найдена ---")
        return

    while True:
        print("Введите путь к изображению (или 'exit' для выхода):")
        image_path = input().strip().strip('"').strip("'")

        if image_path.lower() == 'exit':
            break

        try:
            class_name, confidence, image = predict_aircraft(model, image_path)
            show(image, class_name, confidence)
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
