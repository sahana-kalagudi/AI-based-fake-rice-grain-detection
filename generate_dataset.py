import os
import cv2
import numpy as np

# -------------------------------------------------
# Folders
# -------------------------------------------------

INPUT_FOLDER = "dataset/fake_original"
OUTPUT_FOLDER = "dataset/fake"

# Number of generated images from EACH original image
IMAGES_PER_ORIGINAL = 100

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------------------------
# Find images
# -------------------------------------------------

files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

print("Original images found:", len(files))

if len(files) == 0:
    print("ERROR: No images found in dataset/fake_original")
    exit()

# -------------------------------------------------
# Generate augmented images
# -------------------------------------------------

count = 0

for filename in files:

    path = os.path.join(INPUT_FOLDER, filename)

    original = cv2.imread(path)

    if original is None:
        print("Could not read:", filename)
        continue

    for i in range(IMAGES_PER_ORIGINAL):

        img = original.copy()

        # 1. Random rotation
        angle = np.random.uniform(-25, 25)

        height, width = img.shape[:2]

        center = (width // 2, height // 2)

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            np.random.uniform(0.85, 1.15)
        )

        img = cv2.warpAffine(
            img,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REFLECT
        )

        # 2. Random brightness
        brightness = np.random.randint(-30, 31)

        img = cv2.convertScaleAbs(
            img,
            alpha=np.random.uniform(0.85, 1.15),
            beta=brightness
        )

        # 3. Random horizontal flip
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        # 4. Random vertical flip
        if np.random.rand() > 0.7:
            img = cv2.flip(img, 0)

        # 5. Random blur
        if np.random.rand() > 0.7:
            kernel = np.random.choice([3, 5])
            img = cv2.GaussianBlur(
                img,
                (kernel, kernel),
                0
            )

        # 6. Random noise
        if np.random.rand() > 0.7:

            noise = np.random.normal(
                0,
                8,
                img.shape
            ).astype(np.int16)

            img = np.clip(
                img.astype(np.int16) + noise,
                0,
                255
            ).astype(np.uint8)

        # Save
        output_name = f"fake_{count + 1:04d}.jpg"

        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_name
        )

        cv2.imwrite(output_path, img)

        count += 1

print()
print("====================================")
print("DATASET GENERATION COMPLETED")
print("====================================")
print("Original images :", len(files))
print("Generated images:", count)
print("Saved in        :", OUTPUT_FOLDER)
print("====================================")