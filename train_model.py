import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

DATASET_DIR = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "rice_model.keras")


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

fake_dir = os.path.join(DATASET_DIR, "fake")
real_dir = os.path.join(DATASET_DIR, "real")

if not os.path.exists(fake_dir):
    raise FileNotFoundError(
        f"Fake rice folder not found: {fake_dir}"
    )

if not os.path.exists(real_dir):
    raise FileNotFoundError(
        f"Real rice folder not found: {real_dir}"
    )


# ============================================================
# COUNT IMAGES
# ============================================================

def count_images(folder):

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    count = 0

    for filename in os.listdir(folder):

        if filename.lower().endswith(valid_extensions):
            count += 1

    return count


fake_count = count_images(fake_dir)
real_count = count_images(real_dir)

total_images = fake_count + real_count


print("\n====================================")
print("        DATASET INFORMATION")
print("====================================")

print("Fake rice images :", fake_count)
print("Real rice images :", real_count)
print("Total images     :", total_images)

print("====================================")


if fake_count == 0:
    raise ValueError(
        "No images found in dataset/fake"
    )

if real_count == 0:
    raise ValueError(
        "No images found in dataset/real"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0,

    validation_split=0.20,

    rotation_range=20,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.15,

    shear_range=0.10,

    horizontal_flip=True,

    fill_mode="nearest"
)


# ============================================================
# TRAINING DATA
# ============================================================

train_data = train_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    classes=["fake", "real"],

    subset="training",

    shuffle=True,

    seed=42
)


# ============================================================
# VALIDATION DATA
# ============================================================

validation_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0,

    validation_split=0.20
)


validation_data = validation_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    classes=["fake", "real"],

    subset="validation",

    shuffle=False,

    seed=42
)


# ============================================================
# CLASS MAPPING
# ============================================================

print("\n====================================")
print("          CLASS MAPPING")
print("====================================")

print(train_data.class_indices)

print("\nExpected:")
print("Fake Rice = 0")
print("Real Rice = 1")

print("====================================")


if train_data.class_indices != {
    "fake": 0,
    "real": 1
}:

    raise ValueError(
        "ERROR: Class mapping is incorrect!"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

# Give more importance to Fake Rice because
# the dataset contains far fewer fake images.

class_weight = {

    0: total_images / (2 * fake_count),

    1: total_images / (2 * real_count)

}


print("\n====================================")
print("          CLASS WEIGHTS")
print("====================================")

print(
    "Fake Rice weight :",
    round(class_weight[0], 2)
)

print(
    "Real Rice weight :",
    round(class_weight[1], 2)
)

print("====================================")


# ============================================================
# CNN MODEL
# ============================================================

model = Sequential([

    Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),


    # ========================================================
    # BLOCK 1
    # ========================================================

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling2D(
        pool_size=(2, 2)
    ),


    # ========================================================
    # BLOCK 2
    # ========================================================

    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling2D(
        pool_size=(2, 2)
    ),


    # ========================================================
    # BLOCK 3
    # ========================================================

    Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling2D(
        pool_size=(2, 2)
    ),


    # ========================================================
    # BLOCK 4
    # ========================================================

    Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling2D(
        pool_size=(2, 2)
    ),


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    GlobalAveragePooling2D(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n====================================")
print("             CNN MODEL")
print("====================================")

model.summary()


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=7,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.000001,

    verbose=1
)


checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n====================================")
print("          TRAINING STARTED")
print("====================================\n")


history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS,

    class_weight=class_weight,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n====================================")
print("       FINAL MODEL EVALUATION")
print("====================================")


loss, accuracy = model.evaluate(

    validation_data,

    verbose=1
)


print(
    "\nValidation Loss     :",
    loss
)

print(
    "Validation Accuracy :",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n====================================")
print("     MODEL TRAINING COMPLETED")
print("====================================")

print("\nModel saved at:")
print(MODEL_PATH)

print("\nClass mapping:")
print("Fake Rice = 0")
print("Real Rice = 1")

print("\nDataset used:")
print("Fake Rice =", fake_count)
print("Real Rice =", real_count)

print("\nClass weighting was used to handle")
print("the imbalance between Fake and Real Rice.")

print("====================================")