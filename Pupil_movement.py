import cv2
import numpy as np

def get_pupil_center(eye_region, threshold=50):
    """Detect the pupil center in the eye region using thresholding."""
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_eye, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 10:  # Filter small noise
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
    return None

def analyze_gaze(eye_center_x, pupil_center, eye_width):
    """Determine if the eye is looking left or right."""
    if pupil_center is None:
        return "Unknown"
    pupil_x, _ = pupil_center
    if pupil_x < eye_center_x - eye_width * 0.15:  # Adjusted for sensitivity
        return "Left"
    elif pupil_x > eye_center_x + eye_width * 0.15:
        return "Right"
    return "Center"

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Process up to two eyes
        eye_region = frame[ey:ey+eh, ex:ex+ew]
        eye_center_x = ex + ew // 2
        eye_center_y = ey + eh // 2

        # Get pupil center
        pupil = get_pupil_center(eye_region)
        if pupil:
            pupil = (pupil[0] + ex, pupil[1] + ey)  # Adjust to global coordinates
            cv2.circle(frame, pupil, 3, (0, 0, 255), -1)  # Red for pupil

        # Analyze gaze
        gaze = analyze_gaze(eye_center_x, pupil, ew)
        label = f"Eye {i+1}: {gaze}"
        cv2.putText(frame, label, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw eye center and bounding box
        cv2.circle(frame, (eye_center_x, eye_center_y), 3, (255, 0, 0), -1)  # Blue for eye center
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 1)

    cv2.imshow("Pupil Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()