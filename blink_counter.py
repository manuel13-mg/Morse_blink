import subprocess
import cv2
import mediapipe as mp
import time
import json
import math
import os

# Run eye_loc.py first
print("Running eye_loc.py to get eye locations...")
subprocess.run(["python", "eye_loc.py"])

# Load eye centers from generated file
if not os.path.exists("eye_location.json"):
    print("❌ eye_location.json not found. Make sure eye_loc.py works.")
    exit()

with open("eye_location.json", "r") as f:
    loc_data = json.load(f)

left_eye_center = tuple(loc_data["left_eye"])
right_eye_center = tuple(loc_data["right_eye"])

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Blink counters
short_blinks = 0
medium_blinks = 0
long_blinks = 0

blink_started = False
blink_start_time = None

# EAR threshold
EAR_THRESHOLD = 0.18

# Indices for EAR calculation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_EAR(landmarks, eye_indices, img_w, img_h):
    points = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]
    vertical1 = euclidean(points[1], points[5])
    vertical2 = euclidean(points[2], points[4])
    horizontal = euclidean(points[0], points[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = get_EAR(landmarks, LEFT_EYE, w, h)
        right_ear = get_EAR(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2

        # Detect blink
        if avg_ear < EAR_THRESHOLD:
            if not blink_started:
                blink_started = True
                blink_start_time = time.time()
        else:
            if blink_started:
                blink_duration = time.time() - blink_start_time
                blink_started = False

                if blink_duration < 0.3:
                    short_blinks += 1
                elif 0.3 <= blink_duration <= 2.0:
                    medium_blinks += 1
                else:
                    long_blinks += 1

                print(f"Blink duration: {blink_duration:.2f}s")

    # Display (optional)
    cv2.imshow("Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# Final results
print("\nBlink Counts:")
print(f"Short Blinks (<0.3s): {short_blinks}")
print(f"Medium Blinks (0.3-2s): {medium_blinks}")
print(f"Long Blinks (>2s): {long_blinks}")
