from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    SIZE = 128
    batch_size = 64
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        'cell_images2/uninfected_train/',
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        class_mode='input'
    )

    validation_generator = datagen.flow_from_directory(
        'cell_images2/uninfected_test/',
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        class_mode='input'
    )

    anomaly_generator = datagen.flow_from_directory(
        'cell_images2/parasitized/',
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        class_mode='input'
    )

    return train_generator, validation_generator, anomaly_generator
