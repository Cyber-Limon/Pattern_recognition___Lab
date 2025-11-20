import os
import cv2
import numpy as np
from collections import Counter
import xml.etree.ElementTree as et
from classes import id_classes, img_size


def get_train_test_image():
    imagesets_path = os.path.join('dataset', 'ImageSets', 'Main')

    with open(os.path.join(imagesets_path, 'test.txt'), 'r') as f:
        train_image = [line.strip() for line in f.readlines()]
    with open(os.path.join(imagesets_path, 'train.txt'), 'r') as f:
        test_image = [line.strip() for line in f.readlines()]

    return train_image, test_image


def load_images(list_image):
    images = []
    jpegimages_path = os.path.join('dataset', 'JPEGImages')

    for image_name in list_image:
        image_path = os.path.join(jpegimages_path, f"{image_name}.jpg")
        image = cv2.imread(image_path)

        if image is not None:
            image = cv2.resize(image, img_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        else:
            print(f"Не удалось загрузить: {image_path}")

    return np.array(images)


def get_main_class(annotation_path, image_name):
    xml_path = os.path.join(annotation_path, f"{image_name}.xml")

    try:
        tree = et.parse(xml_path)
        root = tree.getroot()

        classes = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            classes.append(class_name)

        if classes:
            most_common = Counter(classes).most_common(1)[0][0]
            return most_common

    except Exception as e:
        print(f"Ошибка чтения {xml_path}: {e}")
        return None


def get_labels_for_files(annotation_path, list_image):
    labels = []
    for image_name in list_image:
        class_name = get_main_class(annotation_path, image_name)
        labels.append(class_name)
    return labels


def prepare_dataset():
    train_image, test_image = get_train_test_image()

    x_train = load_images(train_image)
    x_test = load_images(test_image)

    annotations_path = os.path.join('dataset', 'Annotations', 'Horizontal Bounding Boxes')

    y_train = get_labels_for_files(annotations_path, train_image)
    y_test = get_labels_for_files(annotations_path, test_image)

    y_train_ids = [id_classes[cls] for cls in y_train]
    y_test_ids = [id_classes[cls] for cls in y_test]

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(f"Подготовленные данные:")
    print(f"- Train: {len(x_train)} изображений")
    print(f"- Test: {len(x_test)} изображений")
    print(f"- Классы: {list(id_classes)}")

    return (x_train, np.array(y_train_ids)), (x_test, np.array(y_test_ids)), id_classes
