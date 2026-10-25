import cv2
import tensorflow as tf
import numpy as np
import os


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "model/rice_model.keras"

print("Loading CNN model...")

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found!")
    print("Expected:", MODEL_PATH)
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully!")


# ============================================================
# OPEN CAMERA
# ============================================================

print("Opening camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")
print("Press SPACE to capture.")
print("Press Q to quit.")


# ============================================================
# CAMERA LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break


    # --------------------------------------------------------
    # SHOW CLEAN CAMERA
    # --------------------------------------------------------

    cv2.imshow(
        "AI Fake Rice Detection",
        frame
    )


    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # SPACE → CAPTURE
    # ========================================================

    if key == 32:

        print("\nImage captured!")

        os.makedirs(
            "captured_images",
            exist_ok=True
        )

        image_path = os.path.join(
            "captured_images",
            "camera_rice.jpg"
        )

        cv2.imwrite(
            image_path,
            frame
        )

        print("Saved:", image_path)


        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

        img = cv2.resize(
            frame,
            (224, 224)
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = img.astype(
            np.float32
        )

        img = img / 255.0

        img = np.expand_dims(
            img,
            axis=0
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        print("Running CNN prediction...")

        prediction = model.predict(
            img,
            verbose=0
        )[0][0]

        prediction = float(prediction)

        print(
            "Raw CNN output:",
            prediction
        )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if prediction >= 0.5:

            result = "REAL RICE"
            confidence = prediction * 100

        else:

            result = "FAKE RICE"
            confidence = 0.0


        # ====================================================
        # TERMINAL RESULT
        # ====================================================

        print("\n========================================")
        print("          PREDICTION RESULT")
        print("========================================")

        print(
            "Prediction :",
            result
        )

        print(
            "Confidence :",
            round(confidence, 2),
            "%"
        )

        print("========================================")


        # ====================================================
        # RESULT WINDOW
        # ====================================================

        result_frame = frame.copy()

        if result == "REAL RICE":

            display_text = (
                "REAL RICE - "
                + str(round(confidence, 2))
                + "%"
            )

        else:

            display_text = "FAKE RICE - 0.0%"


        cv2.putText(
            result_frame,
            display_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )


        cv2.imshow(
            "Prediction Result",
            result_frame
        )

        cv2.waitKey(0)

        cv2.destroyWindow(
            "Prediction Result"
        )


    # ========================================================
    # Q → QUIT
    # ========================================================

    elif key == ord("q"):

        break


# ============================================================
# CLOSE CAMERA
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nCamera closed.")