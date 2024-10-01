import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

class BrainMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", "_mask.jpg"))  # Assuming masks have a similar naming convention
        
        # Load images
        image = np.array(Image.open(image_path).convert("L"))  # Convert to grayscale
        mask = np.array(Image.open(mask_path).convert("L"))
        
        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        return torch.Tensor(image).unsqueeze(0), torch.Tensor(mask).unsqueeze(0)

def get_dataloaders(batch_size=8):
    # Update paths to your training and validation datasets
    train_image_dir = r"D:\5C Task\kaggle_3m\train_images"
    train_mask_dir = r"D:\5C Task\kaggle_3m\train_masks"
    val_image_dir = r"D:\5C Task\kaggle_3m\val_images"
    val_mask_dir = r"D:\5C Task\kaggle_3m\val_masks"
    
    train_dataset = BrainMRIDataset(train_image_dir, train_mask_dir)
    val_dataset = BrainMRIDataset(val_image_dir, val_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return {"train": train_loader, "val": val_loader}
