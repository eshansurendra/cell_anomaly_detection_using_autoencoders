# visualize.py 

import matplotlib.pyplot as plt
import random 
import glob
from main import preprocess_image, load_model 

def visualize_reconstructions(autoencoder_model, image_paths, num_images=5):
    """Visualizes original and reconstructed images from a directory."""
    fig, axs = plt.subplots(2, num_images, figsize=(15, 6))
    random_indices = random.sample(range(len(image_paths)), num_images)
    for i, idx in enumerate(random_indices):
        img_path = image_paths[idx]
        original_img = preprocess_image(img_path)
        reconstructed_img = autoencoder_model.predict(original_img)
        axs[0, i].imshow(original_img[0])
        axs[0, i].axis('off')
        axs[1, i].imshow(reconstructed_img[0])
        axs[1, i].axis('off')
    plt.show()

if __name__ == "__main__":
    model = load_model() 
    image_dir = 'cell_images2/uninfected_train/images/' # Example directory 
    image_paths = glob.glob(image_dir + '*') 
    visualize_reconstructions(model, image_paths) 
