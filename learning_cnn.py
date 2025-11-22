from CNN import cnn
from check_dataset import check_dataset, download_dataset
from prepare_cnn import extract_aircraft_rois, load_classification_dataset, check_dataset_stats


def main():
    if not check_dataset():
        download_dataset()

    if not check_dataset_stats():
        extract_aircraft_rois()

    batch_size = 32

    train, val, test = load_classification_dataset(batch_size)

    model = cnn()
    model.summary()
    model.fit(train, epochs=1, validation_data=val)
    model.evaluate(test)

    model.save('cnn_model.keras')
    print("--- Модель CNN сохранена ---")


if __name__ == "__main__":
    main()
