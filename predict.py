import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.utils.image_dataset_utils import load_image

from learning_yolo import yolo_loss
from learning_cnn import focal_loss
from prepare_yolo import num_anchors, grid_size, anchors
from classes import id_classes, img_size, confidence_thresholds


def load_models():
    models = {}

    try:
        models['yolo'] = tf.keras.models.load_model('yolo_model.keras', custom_objects={'yolo_loss': yolo_loss})
        print("Модель YOLO загружена")
    except Exception as e:
        print(f"Модель YOLO не загружена: {e}")
        return None

    try:
        models['cnn'] = tf.keras.models.load_model('cnn_model.keras', custom_objects={'yolo_loss': focal_loss})
        print("Модель CNN загружена")
    except Exception as e:
        print(f"Модель CNN не загружена: {e}")
        return None

    return models


def extract_aircraft(image, yolo_output, confidence):
    grid_w, grid_h = grid_size
    yolo_reshaped = yolo_output.reshape(*grid_size, num_anchors, 5)

    all_boxes = []
    all_scores = []
    all_padded = []

    for i in range(grid_h):
        for j in range(grid_w):
            for a in range(num_anchors):
                data = yolo_reshaped[i, j, a]
                obj_confidence = data[4]

                if obj_confidence < confidence:
                    continue

                x_center, y_center, width, height = data[0:4]

                x_center = (x_center + j) / grid_w * image.shape[1]
                y_center = (y_center + i) / grid_h * image.shape[0]
                width = anchors[a][0] * np.exp(width) * image.shape[1]
                height = anchors[a][1] * np.exp(height) * image.shape[0]

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                x1 = max(0, min(x1, image.shape[1] - 1))
                y1 = max(0, min(y1, image.shape[0] - 1))
                x2 = max(0, min(x2, image.shape[1] - 1))
                y2 = max(0, min(y2, image.shape[0] - 1))

                padding = 5
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(image.shape[1], x2 + padding)
                y2_pad = min(image.shape[0], y2 + padding)

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(float(obj_confidence))
                all_padded.append([x1_pad, y1_pad, x2_pad, y2_pad])

    if len(all_boxes) == 0:
        return [], []

    keep = nms(all_boxes, all_scores)

    rois = []
    detections = []

    for idx in keep:
        x1, y1, x2, y2 = all_boxes[idx]
        x1_pad, y1_pad, x2_pad, y2_pad = all_padded[idx]
        obj_confidence = all_scores[idx]

        roi = image[y1_pad:y2_pad, x1_pad:x2_pad]

        if roi.shape[0] < 20 or roi.shape[1] < 20:
            continue

        roi_resized = cv2.resize(roi, img_size)
        roi_normalized = roi_resized / 255.0

        detection_info = {'bbox': [x1, y1, x2, y2],
                          'bbox_padded': [x1_pad, y1_pad, x2_pad, y2_pad],
                          'roi': roi_normalized,
                          'detection_confidence': obj_confidence,
                          'class_name': None,
                          'classification_confidence': 0.0}


        rois.append(roi_normalized)
        detections.append(detection_info)

    return rois, detections


def classify_aircraft(cnn_model, rois, detections):
    if not rois:
        return detections

    roi_batch = np.array(rois)

    predictions = cnn_model.predict(roi_batch, verbose=0)

    for i, detection in enumerate(detections):
        class_probs = predictions[i]
        predicted_class_idx = np.argmax(class_probs)
        classification_confidence = np.max(class_probs)

        class_name = None
        for name, idx in id_classes.items():
            if idx == predicted_class_idx:
                class_name = name
                break

        detection['class_name'] = class_name
        detection['classification_confidence'] = classification_confidence
        detection['final_confidence'] = detection['detection_confidence'] * classification_confidence

    return detections


def draw_detections(image, detections, confidence):
    image_with_boxes = image.copy()

    sorted_detections = sorted(detections, key=lambda x: x['final_confidence'], reverse=True)

    for i, detection in enumerate(sorted_detections):
        if detection['final_confidence'] < confidence:
            continue

        bbox = detection['bbox']
        class_name = detection['class_name']
        final_confidence = detection['final_confidence']

        if final_confidence > 0.7:
            color = (0, 255, 0)
        elif final_confidence > 0.5:
            color = (255, 255, 0)
        else:
            color = (255, 0, 0)

        cv2.rectangle(image_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        text = f"{class_name}: {final_confidence:.3f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        cv2.rectangle(image_with_boxes,
                      (bbox[0], bbox[1] - text_size[1] - 5),
                      (bbox[0] + text_size[0], bbox[1]),
                      color, -1)

        cv2.putText(image_with_boxes, text,
                    (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image_with_boxes


def show_results(image, detections, confidence):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)

    valid_detections = [d for d in detections if d['final_confidence'] >= confidence]

    if valid_detections:
        print(f"\nНайдено самолетов: {len(valid_detections)}")
        for i, det in enumerate(valid_detections):
            print(f"{i + 1}. {det['class_name']} - "
                  f"Детекция: {det['detection_confidence']:.3f}, "
                  f"Классификация: {det['classification_confidence']:.3f}, "
                  f"Итог: {det['final_confidence']:.3f}")
    else:
        print("Самолеты не обнаружены")

    title = f"Детекция и классификация самолетов\n"
    title += f"Порог уверенности: {confidence}\n"
    title += f"Найдено самолетов: {len(valid_detections)}"

    plt.title(title, fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def image_processing(models, image_path, confidence):
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return None, None

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None, None

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(original_image, img_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    yolo_predictions = models['yolo'].predict(image, verbose=0)
    yolo_output = yolo_predictions[0]

    rois, detections = extract_aircraft(original_image, yolo_output, confidence)

    if rois:
        detections = classify_aircraft(models['cnn'], rois, detections)

    return original_image, detections


def nms(boxes, scores):
    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        inds = np.where(iou < confidence_thresholds[1])[0]
        order = order[inds + 1]

    return keep



def main():
    models = load_models()

    if models is None:
        print("Не удалось загрузить модели")
        return


    while True:
        print("\nВведите путь к изображению (или 'exit' для выхода):")
        image_path = input().strip().strip('"').strip("'")

        if image_path.lower() == 'exit':
            break

        try:
            for confidence in confidence_thresholds:
                print(f"\n--- Порог уверенности: {confidence} ---")

                original_image, detections = image_processing(models, image_path, confidence)

                if original_image is None:
                    continue

                result_image = draw_detections(original_image, detections, confidence)
                show_results(result_image, detections, confidence)

        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()