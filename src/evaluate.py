import numpy as np
from sklearn.neighbors import KernelDensity
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt

SIZE = 128

def calculate_density_and_recon_error(batch_images, model, encoder_model):
    encoder_output_shape = encoder_model.output_shape
    out_vector_shape = encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3]

    encoded_images = encoder_model.predict(batch_images)
    encoded_images_vector = [np.reshape(img, (out_vector_shape)) for img in encoded_images]

    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(encoded_images_vector)

    density_list = []
    recon_error_list = []
    for im in range(0, batch_images.shape[0]):
        img = batch_images[im]
        img = img[np.newaxis, :, :, :]
        encoded_img = encoder_model.predict([img])
        encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img]
        density = kde.score_samples(encoded_img)[0]
        reconstruction = model.predict([img])
        reconstruction_error = model.evaluate([reconstruction], [img], batch_size=1)[0]
        density_list.append(density)
        recon_error_list.append(reconstruction_error)

    average_density = np.mean(np.array(density_list))
    stdev_density = np.std(np.array(density_list))
    average_recon_error = np.mean(np.array(recon_error_list))
    stdev_recon_error = np.std(np.array(recon_error_list))

    return average_density, stdev_density, average_recon_error, stdev_recon_error

def check_anomaly(img_path, model, encoder_model, uninfected_values, anomaly_values):
    density_threshold = uninfected_values[0] - 2 * uninfected_values[1]
    reconstruction_error_threshold = uninfected_values[2] + 2 * uninfected_values[3]

    img = Image.open(img_path)
    img = np.array(img.resize((128, 128), Image.ANTIALIAS))
    plt.imshow(img)
    img = img / 255.
    img = img[np.newaxis, :, :, :]
    encoder_output_shape = encoder_model.output_shape
    out_vector_shape = encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3]
    encoded_img = encoder_model.predict([img])
    encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img]
    density = kde.score_samples(encoded_img)[0]

    reconstruction = model.predict([img])
    reconstruction_error = model.evaluate([reconstruction], [img], batch_size=1)[0]

    if density < density_threshold or reconstruction_error > reconstruction_error_threshold:
        print("The image is an anomaly")
    else:
        print("The image is NOT an anomaly")
