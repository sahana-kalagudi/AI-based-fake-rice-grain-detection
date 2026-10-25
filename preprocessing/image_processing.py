import cv2
import numpy as np

def preprocess_image(image_path):
    # Read image
    image = cv2.imread(image_path)

    # Resize image
    image = cv2.resize(image, (224, 224))

    # Save original resized image
    original = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge Detection
    edges = cv2.Canny(blur, 50, 150)

    # Convert back to RGB for CNN
    processed = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)

    # Normalize
    processed = processed.astype("float32") / 255.0

    return processed, original, gray, blur, edges