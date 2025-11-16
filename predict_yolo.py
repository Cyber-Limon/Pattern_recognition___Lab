import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from prepare_yolo import grid_size, num_anchors
from classes import id_classes, img_size
from evaluate_yolo import thresholds
from learning_yolo import loss


def load_model():
    try:
        model = tf.keras.models.load_model('yolo_model.keras', custom_objects={'loss': loss})
        return model
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None


def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return None

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    processed_image = cv2.resize(original_image, img_size)
    processed_image = processed_image / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)

    return original_image, processed_image


def extract_predictions(yolo_output, confidence_threshold):
    grid_h, grid_w = grid_size
    yolo_reshaped = yolo_output.reshape(grid_h, grid_w, num_anchors, 25)

    detections = []

    for i in range(grid_h):
        for j in range(grid_w):
            for a in range(num_anchors):
                data = yolo_reshaped[i, j, a]

                obj_confidence = data[4]

                if obj_confidence < confidence_threshold:
                    continue

                x_center, y_center, width, height = data[0:4]

                x_center_abs = (x_center + j) / grid_w * img_size[0]
                y_center_abs = (y_center + i) / grid_h * img_size[1]
                width_abs = width * img_size[0]
                height_abs = height * img_size[1]

                x1 = int(x_center_abs - width_abs / 2)
                y1 = int(y_center_abs - height_abs / 2)
                x2 = int(x_center_abs + width_abs / 2)
                y2 = int(y_center_abs + height_abs / 2)

                x1 = max(0, min(x1, img_size[0] - 1))
                y1 = max(0, min(y1, img_size[1] - 1))
                x2 = max(0, min(x2, img_size[0] - 1))
                y2 = max(0, min(y2, img_size[1] - 1))

                class_probs = data[5:25]
                class_id = np.argmax(class_probs)
                class_confidence = np.max(class_probs)

                total_confidence = obj_confidence * class_confidence

                class_name = None
                for name, idx in id_classes.items():
                    if idx == class_id:
                        class_name = name
                        break

                detection = {'bbox': [x1, y1, x2, y2],
                             'class_name': class_name,
                             'confidence': total_confidence,
                             'class_id': class_id}

                detections.append(detection)

    return detections


def draw_boxes(image, detections):
    image_with_boxes = image.copy()

    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']

        cv2.rectangle(image_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        text = f"{class_name}: {confidence:.3f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        cv2.rectangle(image_with_boxes, (bbox[0], bbox[1] - text_size[1] - 5), (bbox[0] + text_size[0], bbox[1]), (0, 0, 255), -1)

        cv2.putText(image_with_boxes, text, (bbox[0], bbox[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image_with_boxes


def show_detection(image, detections, confidence_threshold):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)

    title = f"YOLO Детекция\n"
    title += f"Порог уверенности: {confidence_threshold}\n"
    title += f"Найдено объектов: {len(detections)}"

    plt.title(title, fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if detections:
        print(f"\nНайдено объектов: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"{i + 1}. {det['class_name']} - уверенность: {det['confidence']:.3f}")
    else:
        print("Объекты не обнаружены")


def predict_yolo(model, image_path, confidence):
    result = load_image(image_path)
    if result is None:
        return "Ошибка загрузки", 0, None

    original_image, processed_image = result

    predictions = model.predict(processed_image, verbose=0)
    yolo_output = predictions[0]

    detections = extract_predictions(yolo_output, confidence)

    image_with_boxes = draw_boxes(original_image, detections)

    return detections, image_with_boxes


def main():
    model = load_model()
    if model is None:
        print("--- Модель не найдена ---")
        return

    while True:
        print("\nВведите путь к изображению (или 'exit' для выхода):")
        image_path = input().strip().strip('"').strip("'")

        if image_path.lower() == 'exit':
            break

        try:
            for confidence in thresholds:
                print(f"\n--- Порог уверенности: {confidence} ---")

                detections, image_with_boxes = predict_yolo(model, image_path, confidence)

                if image_with_boxes is not None:
                    show_detection(image_with_boxes, detections, confidence)

        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
