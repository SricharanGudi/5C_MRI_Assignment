import cv2
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast, Compose
)

def augment(image, mask):
    transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=45, p=0.5),
        RandomBrightnessContrast(p=0.2)
    ])
    augmented = transform(image=image, mask=mask)
    return augmented['image'], augmented['mask']

# Load an example image and mask
image = cv2.imread('test_image.jpg')
mask = cv2.imread('test_mask.jpg', cv2.IMREAD_GRAYSCALE)

if image is None or mask is None:
    print("Error: Image or mask not found.")
else:
    # Apply augmentations
    augmented_image, augmented_mask = augment(image, mask)

    # Save the augmented image and mask
    cv2.imwrite('augmented_image.jpg', augmented_image)
    cv2.imwrite('augmented_mask.jpg', augmented_mask)
    print("Augmented image and mask saved.")
