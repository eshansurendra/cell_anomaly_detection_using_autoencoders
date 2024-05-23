# train.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configuration options
SIZE = 128
BATCH_SIZE = 64
EPOCHS = 100  # Consider increasing for better results

def create_model():
    """Defines the convolutional autoencoder model architecture."""
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
    return model

def create_data_generators(train_dir, val_dir):
    """Creates data generators for training and validation."""
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(SIZE, SIZE),
        batch_size=BATCH_SIZE,
        class_mode='input'
    )
    validation_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(SIZE, SIZE),
        batch_size=BATCH_SIZE,
        class_mode='input'
    )
    return train_generator, validation_generator

def train_model(model, train_generator, validation_generator):
    """Compiles and trains the autoencoder model."""
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    model.summary()
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        shuffle=True
    )
    return history

def plot_training_history(history):
    """Visualizes the training and validation loss over epochs."""
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def save_model(model, model_path='autoencoder_model.h5'):
    """Saves the trained autoencoder model to a file."""
    model.save(model_path)

if __name__ == "__main__":
    # 1. Data Preparation
    train_data_dir = 'cell_images2/uninfected_train/'
    val_data_dir = 'cell_images2/uninfected_test/'
    train_generator, validation_generator = create_data_generators(train_data_dir, val_data_dir)

    # 2. Model Creation
    autoencoder_model = create_model()

    # 3. Model Training
    history = train_model(autoencoder_model, train_generator, validation_generator)

    # 4. Visualization of Training
    plot_training_history(history)

    # 5. Model Saving
    save_model(autoencoder_model) 
