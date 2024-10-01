import numpy as np
import cv2

# Load an image
image = cv2.imread('test_image.jpg')
height, width = image.shape[:2]

# Create a simple synthetic mask (e.g., a circle in the center)
mask = np.zeros((height, width), dtype=np.uint8)
cv2.circle(mask, (width // 2, height // 2), 50, (255), -1)  # Circle in the center

# Save the synthetic mask
cv2.imwrite('test_mask.jpg', mask)
print("Synthetic mask created and saved as test_mask.jpg")
