from tensorflow.keras import layers, models
from prepare_dataset import get_size


input_shape=(*get_size(), 3)

def cnn(num_classes):
    model = models.Sequential([layers.Input(shape=input_shape),
                               layers.Conv2D(64, (3, 3), activation='relu'),
                               layers.MaxPooling2D((2, 2)),
                               layers.Conv2D(128, (3, 3), activation='relu'),
                               layers.MaxPooling2D((2, 2)),
                               layers.Conv2D(128, (3, 3), activation='relu'),
                               layers.Flatten(),
                               layers.Dense(256, activation='relu'),
                               layers.Dropout(0.3),
                               layers.Dense(256, activation='relu'),
                               layers.Dense(num_classes, activation='softmax')])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
