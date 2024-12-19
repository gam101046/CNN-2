from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
import shutil

# Augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Base and output paths
base_path = '/Users/gam/Desktop/DEEP/CNN/DATASET'
output_path = '/Users/gam/Desktop/DEEP/CNN/DATASET/Augmented_dataset'
os.makedirs(output_path, exist_ok=True)

# Define the target number of images per class (use the max of original counts)
max_images = 4000  # Can be dynamic based on the maximum count from the dataset that we get from step1

# Categories with their original image counts
categories = {'Non Drowsy': 2000, 'Splitted_dataset': 0, 'Drowsy': 2000, 'Use mobile phone': 2000}

for category, total_images in categories.items():
    category_path = os.path.join(base_path, category)
    images = os.listdir(category_path)
    category_output_path = os.path.join(output_path, category)
    os.makedirs(category_output_path, exist_ok=True)

    print(f"\n[INFO] Processing category: {category}")

    # Copy original images to the augmented dataset
    for img_name in images:
        src = os.path.join(category_path, img_name)
        dst = os.path.join(category_output_path, img_name)
        shutil.copy(src, dst)
        print(f"[COPY] Copied {img_name} to {category_output_path}")

    # Augment images if necessary
    if total_images < max_images:
        num_augments = max_images - total_images
        print(f"[INFO] Starting augmentation for {category} to add {num_augments} images...")

        count = 0
        for img_name in images:
            img_path = os.path.join(category_path, img_name)
            img = np.array(Image.open(img_path))

            # Ensure the image has 3 channels (RGB) for augmentation
            if img.ndim == 2:  # Grayscale image, convert to RGB
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                img = np.repeat(img, 3, axis=-1)  # Convert to 3-channel (RGB)
            
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Generate augmented images one by one until the required number is reached
            for batch in datagen.flow(img, batch_size=1, save_to_dir=category_output_path, save_prefix='aug', save_format='jpeg'):
                count += 1
                aug_img_name = f"aug_{count}_{img_name}"
                print(f"[AUGMENT] Generated {aug_img_name} for {category}")

                # Stop once the required number of augmentations is reached
                if count >= num_augments:
                    break

            if count >= num_augments:
                break

        print(f"[INFO] Finished augmentation for {category}: {count} augmented images generated.")
    else:
        print(f"[INFO] No augmentation needed for {category} as it already has {total_images} images.")

print("[INFO] Augmentation process complete!")
