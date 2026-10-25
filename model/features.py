import cv2
import numpy as np


def analyze_features(image_path):
    """
    Analyze visible rice-grain features using OpenCV.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read image.")

    # Resize for consistent processing
    image = cv2.resize(image, (500, 500))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to separate rice from background
    _, threshold = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Ignore very small objects/noise
    contours = [
        c for c in contours
        if cv2.contourArea(c) > 100
    ]

    if not contours:
        return {
            "grain_count": 0,
            "average_area": 0,
            "average_length": 0,
            "average_width": 0,
            "broken_grains": 0,
            "dark_pixels": 0
        }

    areas = []
    lengths = []
    widths = []
    broken_grains = 0

    for contour in contours:

        area = cv2.contourArea(contour)
        areas.append(area)

        x, y, w, h = cv2.boundingRect(contour)

        length = max(w, h)
        width = min(w, h)

        lengths.append(length)
        widths.append(width)

        # A very irregular/short grain may indicate breakage
        if length < 45 or width < 8:
            broken_grains += 1

    # Dark-pixel percentage
    dark_pixels = np.mean(gray < 80) * 100

    return {
        "grain_count": len(contours),
        "average_area": round(float(np.mean(areas)), 2),
        "average_length": round(float(np.mean(lengths)), 2),
        "average_width": round(float(np.mean(widths)), 2),
        "broken_grains": broken_grains,
        "dark_pixels": round(float(dark_pixels), 2)
    }


def build_quality(features, prediction_result=None):
    """
    Build a simple quality assessment from the extracted features.
    """

    score = 100
    criteria = []

    grain_count = features.get("grain_count", 0)
    broken_grains = features.get("broken_grains", 0)
    dark_pixels = features.get("dark_pixels", 0)

    # No grains detected
    if grain_count == 0:
        return {
            "quality": "Unable to determine",
            "score": 0,
            "criteria": ["No rice grains could be detected."]
        }

    # Broken grain criterion
    broken_percentage = (
        broken_grains / grain_count
    ) * 100

    if broken_percentage > 30:
        score -= 30
        criteria.append(
            f"High broken-grain percentage ({broken_percentage:.1f}%)."
        )
    elif broken_percentage > 15:
        score -= 15
        criteria.append(
            f"Moderate broken-grain percentage ({broken_percentage:.1f}%)."
        )
    else:
        criteria.append(
            f"Low broken-grain percentage ({broken_percentage:.1f}%)."
        )

    # Dark/abnormal surface criterion
    if dark_pixels > 20:
        score -= 25
        criteria.append(
            f"High dark-pixel percentage ({dark_pixels:.1f}%)."
        )
    elif dark_pixels > 10:
        score -= 10
        criteria.append(
            f"Some dark/discolored regions detected ({dark_pixels:.1f}%)."
        )
    else:
        criteria.append("Low visible discoloration.")

    score = max(0, min(100, score))

    if score >= 80:
        quality = "High Quality"
    elif score >= 60:
        quality = "Medium Quality"
    else:
        quality = "Low Quality"

    return {
        "quality": quality,
        "score": score,
        "criteria": criteria
    }


def build_reasons(features, prediction_result=None):
    """
    Generate human-readable reasons for the result.
    """

    reasons = []

    grain_count = features.get("grain_count", 0)
    broken_grains = features.get("broken_grains", 0)
    dark_pixels = features.get("dark_pixels", 0)

    if grain_count == 0:
        reasons.append("No clear rice grains were detected.")

    else:

        broken_percentage = (
            broken_grains / grain_count
        ) * 100

        if broken_percentage > 30:
            reasons.append(
                "Many grains appear broken or irregular."
            )
        elif broken_percentage > 15:
            reasons.append(
                "Some broken or irregular grains were detected."
            )
        else:
            reasons.append(
                "Most detected grains have a relatively consistent shape."
            )

        if dark_pixels > 20:
            reasons.append(
                "A high amount of dark/discolored regions was detected."
            )
        elif dark_pixels > 10:
            reasons.append(
                "Some darker regions are visible on the rice."
            )
        else:
            reasons.append(
                "No significant dark discoloration was detected."
            )

    # CNN prediction reason
    if prediction_result:

        result = prediction_result.get("result", "")

        if result == "Real Rice":
            reasons.append(
                "The CNN model classified the image as Real Rice."
            )

        elif result == "Fake Rice":
            reasons.append(
                "The CNN model classified the image as Fake Rice."
            )

    return reasons