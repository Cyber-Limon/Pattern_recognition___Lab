import os
import xml.etree.ElementTree as et
from evaluate_yolo import calculate_iou
from classes import confidence_thresholds
from prepare_yolo import get_train_test_image
from predict import load_models, image_processing


def load_test():
    annotation_path = os.path.join('dataset', 'Annotations', 'Horizontal Bounding Boxes')
    _, test_images = get_train_test_image()

    annotations = {}
    for image_name in test_images:
        xml_path = os.path.join(annotation_path, f"{image_name}.xml")

        try:
            tree = et.parse(xml_path)
            root = tree.getroot()
            true_boxes = []
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                true_boxes.append({'bbox': [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                                            int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)],
                                   'class_name': obj.find('name').text})
            annotations[image_name] = true_boxes

        except:
            continue

    return annotations, test_images


def coincidences(true_boxes, pred_boxes, confidence):
    compared, used_pred = [], set()

    for i, true_box in enumerate(true_boxes):
        for j, pred_box in enumerate(pred_boxes):

            if j in used_pred:
                continue

            iou = calculate_iou(true_box['bbox'], pred_box['bbox'])

            if iou >= confidence:
                compared.append({'true_class': true_box['class_name'], 'pred_class': pred_box['class_name']})
                used_pred.add(j)
                break

    return compared, len(pred_boxes) - len(compared), len(true_boxes) - len(compared)


def evaluate_end_to_end():
    models = load_models()

    if models is None:
        print("Не удалось загрузить модели")
        return

    annotations, test_images = load_test()

    for confidence in confidence_thresholds:
        total_true, correct_det, correct_cls, total_fn, total_fp = 0, 0, 0, 0, 0

        for image in test_images:
            if image not in annotations:
                continue

            true_boxes = annotations[image]
            _, detections = image_processing(models, os.path.join('dataset', 'JPEGImages', f"{image}.jpg"), confidence)

            if not detections:
                continue

            pred_boxes = [{'bbox': d['bbox'], 'class_name': d['class_name']}
                          for d in detections if d['final_confidence'] >= confidence]

            matches, fp, fn = coincidences(true_boxes, pred_boxes, confidence)

            total_true += len(true_boxes)
            correct_det += len(matches)
            correct_cls += sum(1 for m in matches if m['true_class'] == m['pred_class'])
            total_fp += fp
            total_fn += fn

        if total_true > 0:
            print(f"\n--- Порог {confidence} ---")
            print(f"- End-to-End Accuracy: {correct_cls/total_true:.3f}")
            print(f"- Точность детекции: {correct_det/total_true:.3f}")
            print(f"- Точность классификации: {correct_cls/correct_det:.3f}")
            print(f"- Всего самолетов: {total_true}")
            print(f"- Правильные детекции: {correct_det}")
            print(f"- Правильные классификации: {correct_cls}")
            print(f"- Неправильно найденные самолеты: {total_fp}")
            print(f"- Ненайденные самолеты: {total_fn}")


if __name__ == "__main__":
    evaluate_end_to_end()
