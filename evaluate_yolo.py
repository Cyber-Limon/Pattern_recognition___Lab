from classes import img_size
from prepare_yolo import grid_size, num_anchors


thresholds = [0.75, 0.5, 0.25]


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def extract_true_boxes(yolo_data):
    true_boxes = []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for a in range(num_anchors):
                if yolo_data[i, j, a, 4] == 1:
                    x, y, w, h = yolo_data[i, j, a, 0:4]

                    x_abs = (x + j) / grid_size[0] * img_size[0]
                    y_abs = (y + i) / grid_size[1] * img_size[1]
                    w_abs = w * img_size[0]
                    h_abs = h * img_size[1]

                    x1 = max(0, x_abs - w_abs / 2)
                    y1 = max(0, y_abs - h_abs / 2)
                    x2 = min(img_size[0], x_abs + w_abs / 2)
                    y2 = min(img_size[1], y_abs + h_abs / 2)

                    true_boxes.append({'bbox': [x1, y1, x2, y2]})

    return true_boxes


def extract_pred_boxes(pred_data, confidence=thresholds[1]):
    pred_boxes = []
    pred_reshaped = pred_data.reshape(*grid_size, num_anchors, 5)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for a in range(num_anchors):
                if pred_reshaped[i, j, a, 4] > confidence:
                    x, y, w, h = pred_reshaped[i, j, a, 0:4]

                    x_abs = (x + j) / grid_size[0] * img_size[0]
                    y_abs = (y + i) / grid_size[1] * img_size[1]
                    w_abs = w * img_size[0]
                    h_abs = h * img_size[1]

                    x1 = max(0, x_abs - w_abs / 2)
                    y1 = max(0, y_abs - h_abs / 2)
                    x2 = min(img_size[0], x_abs + w_abs / 2)
                    y2 = min(img_size[1], y_abs + h_abs / 2)

                    confidence = pred_reshaped[i, j, a, 4]

                    pred_boxes.append({'bbox': [x1, y1, x2, y2], 'confidence': confidence})

    return pred_boxes


def evaluate(model, test_images, test_true_boxes, iou_threshold=thresholds[1]):
    predictions = model.predict(test_images)

    total_objects = 0
    true_positives = 0
    false_positives = 0

    for i in range(len(test_images)):
        true_boxes = extract_true_boxes(test_true_boxes[i])
        pred_boxes = extract_pred_boxes(predictions[i])

        total_objects += len(true_boxes)

        used_predictions = set()

        for true_box in true_boxes:
            best_iou = 0
            best_pred_idx = -1

            for j, pred_box in enumerate(pred_boxes):
                if j in used_predictions:
                    continue

                iou = calculate_iou(true_box['bbox'], pred_box['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j

            if best_iou > iou_threshold and best_pred_idx != -1:
                true_positives += 1
                used_predictions.add(best_pred_idx)

        false_positives += len(pred_boxes) - len(used_predictions)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / total_objects if total_objects > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Результаты ---")
    print(f"- Правильно найденные самолеты: {true_positives}")
    print(f"- Неправильно найденные самолеты: {false_positives}")
    print(f"- Ненайденные самолеты: {total_objects - true_positives}")
    print(f"- Всего самолетов: {total_objects}")
    print(f"- Точность: {precision:.3f}")
    print(f"- Полнота: {recall:.3f}")
    print(f"- F1-Score: {f1_score:.3f}")

    return precision, recall, f1_score
