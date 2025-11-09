from tensorflow.keras import layers, models
from classes import id_classes, img_size


input_shape=(*img_size, 3)
num_classes = len(id_classes())


def cnn():
    model = models.Sequential([layers.Input(shape=input_shape),
                               layers.Conv2D(64, 3, activation='relu'),
                               layers.MaxPooling2D(2),
                               layers.Conv2D(128, 3, activation='relu'),
                               layers.MaxPooling2D(2),
                               layers.Conv2D(256, 3, activation='relu'),
                               layers.Flatten(),
                               layers.Dense(256, activation='relu'),
                               layers.Dropout(0.3),
                               layers.Dense(256, activation='relu'),
                               layers.Dense(num_classes, activation='softmax')])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
