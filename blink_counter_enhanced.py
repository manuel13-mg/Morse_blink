import subprocess
import cv2
import mediapipe as mp
import time
import json
import math
import os
from collections import deque

# Run eye_loc.py to get eye positions
print("Running eye_loc.py to get eye locations...")
subprocess.run(["python", "eye_loc.py"])

if not os.path.exists("eye_location.json"):
    print("❌ eye_location.json not found.")
    exit()

with open("eye_location.json", "r") as f:
    loc_data = json.load(f)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Constants
FULLY_CLOSED_RATIO = 0.55
CLOSED_FRAMES_REQUIRED = 3
COOLDOWN_DURATION = 0.5

# Blink counters
short_blinks = medium_blinks = long_blinks = 0
left_blinks = right_blinks = 0

# State tracking
blink_started = False
blink_start_time = None
cooldown_start_time = 0

left_closed_frames = 0
right_closed_frames = 0

# EAR smoothing
ear_history = deque(maxlen=5)

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_EAR(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    v1 = euclidean(pts[1], pts[5])
    v2 = euclidean(pts[2], pts[4])
    h_len = euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h_len)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# EAR calibration
print("Calibrating EAR... Keep eyes open.")
ear_samples = []
start_time = time.time()

while time.time() - start_time < 2:
    success, frame = cap.read()
    if not success:
        continue
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        left_ear = get_EAR(landmarks, LEFT_EYE, w, h)
        right_ear = get_EAR(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2
        ear_samples.append(avg_ear)

EAR_THRESHOLD = sum(ear_samples) / len(ear_samples) * 0.75
FULLY_CLOSED_THRESHOLD = EAR_THRESHOLD * FULLY_CLOSED_RATIO
print(f"✅ EAR Threshold: {EAR_THRESHOLD:.3f}, Full Close: {FULLY_CLOSED_THRESHOLD:.3f}")

# Detection loop
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
        ear_history.append(avg_ear)

        smoothed_ear = sum(ear_history) / len(ear_history)
        current_time = time.time()
        in_cooldown = (current_time - cooldown_start_time) < COOLDOWN_DURATION

        # --- Full blink detection (both eyes closed)
        if smoothed_ear < EAR_THRESHOLD:
            if not blink_started:
                blink_started = True
                blink_start_time = current_time
        else:
            if blink_started:
                duration = current_time - blink_start_time
                blink_started = False
                if duration < 0.3:
                    short_blinks += 1
                elif 0.3 <= duration <= 2.0:
                    medium_blinks += 1
                else:
                    long_blinks += 1
                print(f"[Total Blink] Duration: {duration:.2f}s")

        # --- Left blink (only left eye fully closed)
        if left_ear < FULLY_CLOSED_THRESHOLD and right_ear >= EAR_THRESHOLD:
            left_closed_frames += 1
        else:
            if left_closed_frames >= CLOSED_FRAMES_REQUIRED and not in_cooldown:
                left_blinks += 1
                print("[Left Blink] Detected")
                cooldown_start_time = current_time
            left_closed_frames = 0

        # --- Right blink (only right eye fully closed)
        if right_ear < FULLY_CLOSED_THRESHOLD and left_ear >= EAR_THRESHOLD:
            right_closed_frames += 1
        else:
            if right_closed_frames >= CLOSED_FRAMES_REQUIRED and not in_cooldown:
                right_blinks += 1
                print("[Right Blink] Detected")
                cooldown_start_time = current_time
            right_closed_frames = 0

    cv2.imshow("Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Summary
print("\n🔎 Final Blink Summary:")
print(f"Short Blinks (<0.3s): {short_blinks}")
print(f"Medium Blinks (0.3–2s): {medium_blinks}")
print(f"Long Blinks (>2s): {long_blinks}")
print(f"👁️ Left Eye Blinks: {left_blinks}")
print(f"👁️ Right Eye Blinks: {right_blinks}")
