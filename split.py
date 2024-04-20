import os
import shutil
import random

# Define paths with raw string literals
source_dir = r'\cell_images2\Uninfected'
train_dir = r'\cell_images2\uninfected_train'
test_dir = r'\cell_images2\uninfected_test'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get list of images in the source directory
images = os.listdir(source_dir)

# Remove "Thumbs.db" from the list of images
images = [img for img in images if img != 'Thumbs.db']

# Shuffle the list of images
random.shuffle(images)

# Determine split indices
split_index = int(0.8 * len(images))

# Split images into train and test sets
train_images = images[:split_index]
test_images = images[split_index:]

# Move images to respective directories
for img in train_images:
    shutil.move(os.path.join(source_dir, img), os.path.join(train_dir, img))

for img in test_images:
    shutil.move(os.path.join(source_dir, img), os.path.join(test_dir, img))

print("Splitting completed successfully.")
