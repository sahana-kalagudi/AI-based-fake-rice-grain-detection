import os
import cv2
import numpy as np
import tensorflow as tf


# predictor.py is inside the model folder,
# so rice_model.keras is in the same folder.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rice_model.keras"
)

print("Loading model from:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)


def predict_rice(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Unable to read the uploaded image.")

    # Resize to the CNN input size
    img = cv2.resize(img, (224, 224))

    # OpenCV BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # CNN prediction
    prediction = float(
        model.predict(img, verbose=0)[0][0]
    )

    # Model mapping:
    # fake = 0
    # real = 1

    if prediction >= 0.5:
        label = "Real Rice"
        confidence = round(prediction * 100, 2)
    else:
        label = "Fake Rice"
        confidence = 0.0

    return label, confidence