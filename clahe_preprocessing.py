import cv2

def apply_clahe(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

# Load the local image (replace 'test_image.jpg' with your actual file)
image = cv2.imread('test_image.jpg')

if image is None:
    print("Error: Image not found.")
else:
    clahe_image = apply_clahe(image)
    cv2.imwrite('output_image.jpg', clahe_image)
    print("CLAHE applied and output image saved as output_image.jpg")
