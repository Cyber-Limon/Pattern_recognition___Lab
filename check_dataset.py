import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset():
    os.makedirs('dataset', exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("khlaifiabilel/military-aircraft-recognition-dataset", path='dataset', unzip=True)

    print("--- Датасет скачан ---")


def check_dataset():
    folders = [os.path.join('dataset', 'JPEGImages'),
               os.path.join('dataset', 'Annotations'),
               os.path.join('dataset', 'ImageSets', 'Main')]

    for folder in folders:
        if not os.path.exists(folder):
            print("--- Датасет не найден ---")
            return False

    print("--- Датасет найден ---")
    return True
