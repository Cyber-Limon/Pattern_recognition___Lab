from CNN import cnn, focal_loss
from check_dataset import check_dataset, download_dataset
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from prepare_cnn import extract_aircraft_rois, load_classification_dataset, check_dataset_aircraft


def calculate_class_alphas(class_counts):
    total = sum(class_counts.values())
    class_alphas = []

    for count in class_counts.values():
        class_alphas.append(1.0 - (count / total))

    return class_alphas


def main():
    if not check_dataset():
        download_dataset()

    check, class_counts = check_dataset_aircraft()
    if not check:
        extract_aircraft_rois()

    class_alphas = calculate_class_alphas(class_counts)
    def loss(y_true, y_pred):
        return focal_loss(y_true, y_pred, class_alphas)

    batch_size = 32
    train, val, test = load_classification_dataset(batch_size)

    callbacks = [EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)]

    model = cnn()
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.summary()
    model.fit(train, epochs=30, validation_data=val, callbacks=callbacks)
    model.evaluate(test)

    model.save('cnn_model.keras')
    print("--- Модель CNN сохранена ---")


if __name__ == "__main__":
    main()
