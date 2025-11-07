import os
import cv2
import numpy as np
import xml.etree.ElementTree as et
from collections import Counter


dataset_path = 'dataset'
img_size = (200, 200)
id_classes = {'A1':   0, 'A2':   1, 'A3':   2, 'A4':   3, 'A5':   4,
              'A6':   5, 'A7':   6, 'A8':   7, 'A9':   8, 'A10':  9,
              'A11': 10, 'A12': 11, 'A13': 12, 'A14': 13, 'A15': 14,
              'A16': 15, 'A17': 16, 'A18': 17, 'A19': 18, 'A20': 19}

def get_train_test_image():
    imagesets_path = os.path.join(dataset_path, 'ImageSets', 'Main')

    with open(os.path.join(imagesets_path, 'test.txt'), 'r') as f:
        train_image = [line.strip() for line in f.readlines()]
    with open(os.path.join(imagesets_path, 'train.txt'), 'r') as f:
        test_image = [line.strip() for line in f.readlines()]

    return train_image, test_image


def load_images(list_image):
    images = []
    jpegimages_path = os.path.join(dataset_path, 'JPEGImages')

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


def get_id_classes():
    return id_classes


def get_size():
    return img_size


def prepare_dataset():
    train_image, test_image = get_train_test_image()

    x_train = load_images(train_image)
    x_test = load_images(test_image)

    annotations_path = os.path.join(dataset_path, 'Annotations', 'Horizontal Bounding Boxes')

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
