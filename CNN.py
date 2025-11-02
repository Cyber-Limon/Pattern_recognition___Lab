from tensorflow.keras import layers, models


input_shape=(224, 224, 3)

def cnn(num_classes):
    model = models.Sequential([layers.Input(shape=input_shape),
                               layers.Conv2D(32, (3, 3), activation='relu'),
                               layers.MaxPooling2D((2, 2)),
                               layers.Conv2D(64, (3, 3), activation='relu'),
                               layers.MaxPooling2D((2, 2)),
                               layers.Conv2D(64, (3, 3), activation='relu'),
                               layers.Flatten(),
                               layers.Dense(64, activation='relu'),
                               layers.Dropout(0.5),
                               layers.Dense(num_classes, activation='softmax')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
