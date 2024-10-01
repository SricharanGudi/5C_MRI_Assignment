import torch
import segmentation_models_pytorch as smp

# Define U-Net++ Model
model_unetpp = smp.UnetPlusPlus(
    encoder_name="resnet34",  # Encoder backbone
    encoder_weights="imagenet",  # Use pre-trained weights
    in_channels=1,  # Grayscale MRI images
    classes=1,  # Binary segmentation
)
