import os
import cv2
import numpy as np
import xml.etree.ElementTree as et
from sklearn.cluster import KMeans
from prepare_cnn import get_train_test_image, load_images
from classes import num_classes, id_classes, img_size


grid_size = (14, 14)
num_anchors = 3


def anchors():
    bbox_sizes = []
    annotations_path = os.path.join('dataset', 'Annotations', 'Horizontal Bounding Boxes')

    if not os.path.exists(annotations_path):
        print(f"Ошибка: путь {annotations_path} не найден")
        return None

    xml_files = [f for f in os.listdir(annotations_path) if f.endswith('.xml')]

    for xml_file in xml_files:
        tree = et.parse(os.path.join(annotations_path, xml_file))
        root = tree.getroot()

        size_elem = root.find('size')
        image_width = int(size_elem.find('width').text)
        image_height = int(size_elem.find('height').text)

        if image_width == 0 or image_height == 0:
            print(f"Предупреждение: нулевые размеры в {xml_file}")
            continue

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            bbox_sizes.append([width, height])

    bbox_array = np.array(bbox_sizes)
    kmeans = KMeans(n_clusters=num_anchors, random_state=42, n_init=10)
    kmeans.fit(bbox_array)

    anchors = kmeans.cluster_centers_.tolist()
    anchors.sort(key=lambda anchor: anchor[0] * anchor[1])
    anchors = [[round(width, 3), round(height, 3)] for width, height in anchors]

    print("\n--- РЕЗУЛЬТАТЫ ---")
    for i, (width, height) in enumerate(anchors):
        print(f"Якорь {i + 1}: [{width:.3f}, {height:.3f}]")

    return anchors


anchors = anchors()


def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height):
    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height

    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    return x_center, y_center, width, height


def get_grid_cell(x_center, y_center, grid_w, grid_h):
    grid_x = int(x_center * grid_w)
    grid_y = int(y_center * grid_h)

    grid_x = max(0, min(grid_x, grid_w - 1))
    grid_y = max(0, min(grid_y, grid_h - 1))

    return grid_x, grid_y


def find_best_anchor(width, height):
    best_iou = best_id = 0

    for i, (anchor_w, anchor_h) in enumerate(anchors):
        intersection = min(width, anchor_w) * min(height, anchor_h)
        union = width * height + anchor_w * anchor_h - intersection

        iou = intersection / union if union > 0 else 0

        if iou > best_iou:
            best_iou = iou
            best_id = i

    return best_id


def build_yolo_target(annotation_path, image_name):
    grid_h, grid_w = grid_size
    target = np.zeros((grid_h, grid_w, num_anchors, 5 + num_classes))

    xml_path = os.path.join(annotation_path, f"{image_name}.xml")

    try:
        tree = et.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        image_width = int(size.find('width').text)
        image_height = int(size.find('height').text)

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            class_name = obj.find('name').text

            x_center, y_center, width, height = convert_bbox_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height)

            grid_x, grid_y = get_grid_cell(x_center, y_center, grid_w, grid_h)

            best_anchor_idx = find_best_anchor(width, height)

            target[grid_y, grid_x, best_anchor_idx, 0:4] = [x_center, y_center, width, height]
            target[grid_y, grid_x, best_anchor_idx, 4] = 1.0
            target[grid_y, grid_x, best_anchor_idx, 5 + id_classes[class_name]] = 1.0

    except Exception as e:
        print(f"Ошибка обработки {xml_path}: {e}")

    return target


def prepare_yolo_dataset():
    train_images, test_images = get_train_test_image()

    x_train = load_images(train_images)
    x_test = load_images(test_images)

    annotation_path = os.path.join('dataset', 'Annotations', 'Horizontal Bounding Boxes')

    y_train = []
    for image_name in train_images:
        target = build_yolo_target(annotation_path, image_name)
        y_train.append(target)
    y_test = []
    for image_name in test_images:
        target = build_yolo_target(annotation_path, image_name)
        y_test.append(target)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)
