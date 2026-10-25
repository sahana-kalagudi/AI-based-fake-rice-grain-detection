 
# ============================================================
# AI FAKE RICE DETECTION - PREDICTION
# ============================================================

print("========================================")
print("       PREDICT.PY STARTED")
print("========================================")


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

print("\nLoading required libraries...")

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import glob

print("Libraries loaded successfully!")


# ============================================================
# 2. LOAD TRAINED CNN MODEL
# ============================================================

MODEL_PATH = "model/rice_model.keras"

print("\n----------------------------------------")
print("Loading CNN model...")
print("Model path:", MODEL_PATH)
print("----------------------------------------")

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found!")
    print("Expected location:")
    print(MODEL_PATH)

    input("\nPress Enter to exit...")
    exit()

try:
    model = tf.keras.models.load_model(MODEL_PATH)

    print("CNN model loaded successfully!")

except Exception as e:

    print("\nERROR while loading CNN model:")
    print(e)

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# 3. FIND IMAGE
# ============================================================

IMAGE_FOLDER = "captured_images"

print("\n----------------------------------------")
print("Searching for rice images...")
print("Folder:", IMAGE_FOLDER)
print("----------------------------------------")


extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]

image_files = []

if os.path.exists(IMAGE_FOLDER):

    for ext in extensions:

        image_files.extend(
            glob.glob(
                os.path.join(
                    IMAGE_FOLDER,
                    "**",
                    ext
                ),
                recursive=True
            )
        )

else:

    print("\nWARNING: captured_images folder does not exist.")

    os.makedirs(IMAGE_FOLDER)

    print("Created captured_images folder.")


# ============================================================
# 4. CHECK IMAGE
# ============================================================

if len(image_files) == 0:

    print("\n========================================")
    print("ERROR: NO IMAGE FOUND")
    print("========================================")

    print("\nPlease put a rice image inside:")

    print("captured_images/")

    print("\nExample:")
    print("captured_images/rice.jpg")

    print("\n========================================")

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# 5. SELECT FIRST IMAGE
# ============================================================

img_path = image_files[0]

print("\nImage found!")
print("Image:", img_path)


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

print("\n----------------------------------------")
print("Processing image...")
print("----------------------------------------")

try:

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img = image.img_to_array(img)

    img = np.expand_dims(
        img,
        axis=0
    )

    img = img / 255.0

    print("Image processing completed!")

except Exception as e:

    print("\nERROR while processing image:")
    print(e)

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# 7. CNN PREDICTION
# ============================================================

print("\n----------------------------------------")
print("Running CNN prediction...")
print("----------------------------------------")

try:

    prediction = model.predict(
        img,
        verbose=0
    )[0][0]

    prediction = float(prediction)

except Exception as e:

    print("\nERROR during prediction:")
    print(e)

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# 8. DISPLAY RAW CNN OUTPUT
# ============================================================

print("\nRaw CNN output:", prediction)


# ============================================================
# 9. CLASSIFICATION
# ============================================================

if prediction >= 0.5:

    result = "REAL RICE"

    # CNN probability for Real Rice
    confidence = prediction * 100

else:

    result = "FAKE RICE"

    # As requested for your project
    confidence = 0.0


# ============================================================
# 10. FINAL RESULT
# ============================================================

print("\n")
print("========================================")
print("       AI FAKE RICE DETECTION")
print("========================================")

print("Image      :", img_path)

print("----------------------------------------")

print("Prediction :", result)

print("Confidence :", round(confidence, 2), "%")

print("----------------------------------------")

print("========================================")

print("\nPrediction completed successfully!")


# ============================================================
# 11. KEEP TERMINAL OPEN
# ============================================================

input("\nPress Enter to exit...")

