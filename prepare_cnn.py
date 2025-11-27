import os
import cv2
import xml.etree.ElementTree as et
from classes import id_classes, img_size
from prepare_yolo import get_train_test_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def extract_aircraft_rois():
    os.makedirs('aircraft', exist_ok=True)

    for class_name in id_classes.keys():
        os.makedirs(os.path.join('aircraft', 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join('aircraft', 'test', class_name), exist_ok=True)

    annotation_path = os.path.join('dataset', 'Annotations', 'Horizontal Bounding Boxes')
    jpegimages_path = os.path.join('dataset', 'JPEGImages')

    train_images, test_images = get_train_test_image()

    train_count = 0
    test_count = 0
    train_objects = 0
    test_objects = 0

    for image_name in train_images:
        objects_count = process_image_rois(image_name, annotation_path, jpegimages_path, 'train')
        train_objects += objects_count
        train_count += 1

    for image_name in test_images:
        objects_count = process_image_rois(image_name, annotation_path, jpegimages_path, 'test')
        test_objects += objects_count
        test_count += 1

    print(f"\n--- Датасет самолетов создан ---")
    print(f"Обработано тренировочных изображений: {train_count}")
    print(f"Обработано тестовых изображений: {test_count}")
    print(f"Всего самолетов для тренировки: {train_objects}")
    print(f"Всего самолетов для тестирования: {test_objects}")
    print(f"Общее количество самолетов: {train_objects + test_objects}")


def process_image_rois(image_name, annotation_path, jpegimages_path, split):
    xml_file = os.path.join(annotation_path, f"{image_name}.xml")
    image_file = os.path.join(jpegimages_path, f"{image_name}.jpg")

    if not os.path.exists(xml_file) or not os.path.exists(image_file):
        return None

    objects_count = 0

    try:
        tree = et.parse(xml_file)
        root = tree.getroot()

        image = cv2.imread(image_file)
        if image is None:
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i, obj in enumerate(root.findall('object')):
            bndbox = obj.find('bndbox')
            class_name = obj.find('name').text

            if bndbox is None or class_name not in id_classes:
                continue

            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            padding = 5
            xmin_pad = max(0, xmin - padding)
            ymin_pad = max(0, ymin - padding)
            xmax_pad = min(image.shape[1], xmax + padding)
            ymax_pad = min(image.shape[0], ymax + padding)

            roi = image[ymin_pad:ymax_pad, xmin_pad:xmax_pad]
            if roi.shape[0] < 20 or roi.shape[1] < 20:
                continue

            roi_resized = cv2.resize(roi, img_size)

            save_path = os.path.join('aircraft', split, class_name, f"{image_name}_{i}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(roi_resized, cv2.COLOR_RGB2BGR))

            objects_count += 1

    except Exception as e:
        print(f"Ошибка обработки {image_name}: {e}")

    return objects_count


def load_classification_dataset(batch_size):
    class_order = list(id_classes.keys())

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=15,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True,
                                       brightness_range=[0.9, 1.1],
                                       fill_mode='nearest',
                                       validation_split=0.2)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(os.path.join('aircraft', 'train'),
                                                        target_size=img_size,
                                                        batch_size=batch_size,
                                                        class_mode='sparse',
                                                        classes=class_order,
                                                        subset='training',
                                                        shuffle=True)

    validation_generator = train_datagen.flow_from_directory(os.path.join('aircraft', 'train'),
                                                             target_size=img_size,
                                                             batch_size=batch_size,
                                                             class_mode='sparse',
                                                             classes=class_order,
                                                             subset='validation',
                                                             shuffle=True)

    test_generator = test_datagen.flow_from_directory(os.path.join('aircraft', 'test'),
                                                      target_size=img_size,
                                                      batch_size=batch_size,
                                                      class_mode='sparse',
                                                      classes=class_order,
                                                      shuffle=False)

    print("\n--- Результаты ---")
    print(f"- Найдено классов: {train_generator.num_classes}")
    print(f"- Тренировочные примеры: {train_generator.samples}")
    print(f"- Валидационные примеры: {validation_generator.samples}")
    print(f"- Тестовые примеры: {test_generator.samples}")

    return train_generator, validation_generator, test_generator


def check_dataset_aircraft():
    if not os.path.exists('aircraft'):
        print("--- Датасет самолетов не создан ---")
        return False, None

    print("\n--- Датасет самолетов ---")

    class_counts = {}
    for split in ['train', 'test']:
        split_path = os.path.join('aircraft', split)
        total_images = 0

        print(f"\n{split.upper()}:")
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
                print(f"{class_name}: {count} изображений")
                total_images += count

                if class_name not in class_counts:
                    class_counts[class_name] = 0

                class_counts[class_name] += count

        print(f"Всего в {split}: {total_images} изображений")

    return True, class_counts
