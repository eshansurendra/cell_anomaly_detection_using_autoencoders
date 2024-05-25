from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt

SIZE = 128

def build_and_train_model(train_generator, validation_generator):
    model = Sequential()
    
    # Encoder
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    # Decoder
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    model.summary()

    history = model.fit(
        train_generator,
        steps_per_epoch=500 // train_generator.batch_size,
        epochs=1000,
        validation_data=validation_generator,
        validation_steps=75 // validation_generator.batch_size,
        shuffle=True
    )

    encoder_model = Sequential()
    encoder_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3), weights=model.layers[0].get_weights()))
    encoder_model.add(MaxPooling2D((2, 2), padding='same'))
    encoder_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', weights=model.layers[2].get_weights()))
    encoder_model.add(MaxPooling2D((2, 2), padding='same'))
    encoder_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', weights=model.layers[4].get_weights()))
    encoder_model.add(MaxPooling2D((2, 2), padding='same'))
    encoder_model.summary()

    return model, encoder_model, history
