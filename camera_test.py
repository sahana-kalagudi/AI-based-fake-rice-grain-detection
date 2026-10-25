import cv2
import os

# Create folder if it doesn't exist
os.makedirs("captured_images", exist_ok=True)

cap = cv2.VideoCapture(0)

print("Press 'C' to capture image")
print("Press 'Q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Rice Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        cv2.imwrite("captured_images/rice.jpg", frame)
        print("Image Saved Successfully!")
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()