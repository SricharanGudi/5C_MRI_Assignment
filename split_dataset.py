import os
import shutil
import random

# Base path to your dataset
base_path = r"D:\5C Task\kaggle_3m"

# Directories for train and val images/masks
train_image_dir = os.path.join(base_path, 'train_images')
train_mask_dir = os.path.join(base_path, 'train_masks')
val_image_dir = os.path.join(base_path, 'val_images')
val_mask_dir = os.path.join(base_path, 'val_masks')

# Create directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

# List all image files in the training directory
image_files = [f for f in os.listdir(train_image_dir) if f.endswith(('.jpg', '.png'))]

# Shuffle the dataset for random splitting
random.shuffle(image_files)

# Define the train-val split ratio
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

# Split into train and validation lists
train_images = image_files[:split_index]
val_images = image_files[split_index:]

# Function to move files based on lists
def move_files(files, src_dir, dest_dir):
    for file in files:
        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(dest_dir, file)

        # Check if the source file exists before moving
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
        else:
            print(f"File not found: {src_file}")

# Move training data (though it's still in the same location)
move_files(train_images, train_image_dir, train_image_dir)
move_files(train_images, train_mask_dir, train_mask_dir)

# Move validation data
move_files(val_images, train_image_dir, val_image_dir)
move_files(val_images, train_mask_dir, val_mask_dir)

print(f"Total training images: {len(train_images)}")
print(f"Total validation images: {len(val_images)}")
