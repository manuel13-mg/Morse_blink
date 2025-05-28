import cv2
import mediapipe as mp
import json
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Landmark indices for eye corners
LEFT_EYE_LANDMARKS = [33, 133]
RIGHT_EYE_LANDMARKS = [362, 263]

# Webcam
cap = cv2.VideoCapture(0)

# Accumulators for averaging
left_eye_sum = [0, 0]
right_eye_sum = [0, 0]
valid_frames = 0

for _ in range(100):
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        face_landmarks = results.multi_face_landmarks[0]

        # Get eye corner coordinates
        left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                    for i in LEFT_EYE_LANDMARKS]
        right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                     for i in RIGHT_EYE_LANDMARKS]

        # Average each eye pair (center point)
        left_center = ((left_eye[0][0] + left_eye[1][0]) // 2,
                       (left_eye[0][1] + left_eye[1][1]) // 2)
        right_center = ((right_eye[0][0] + right_eye[1][0]) // 2,
                        (right_eye[0][1] + right_eye[1][1]) // 2)

        # Add to sum
        left_eye_sum[0] += left_center[0]
        left_eye_sum[1] += left_center[1]
        right_eye_sum[0] += right_center[0]
        right_eye_sum[1] += right_center[1]

        valid_frames += 1

# Cleanup
cap.release()

# Final average locations
if valid_frames > 0:
    final_left_eye = (left_eye_sum[0] // valid_frames, left_eye_sum[1] // valid_frames)
    final_right_eye = (right_eye_sum[0] // valid_frames, right_eye_sum[1] // valid_frames)

    print(f"Final Averaged Left Eye Location: {final_left_eye}")
    print(f"Final Averaged Right Eye Location: {final_right_eye}")
else:
    print("No valid face detected in any frame.")


# Save locations
if valid_frames > 0:
    loc_data = {
        "left_eye": final_left_eye,
        "right_eye": final_right_eye
    }

    with open("eye_location.json", "w") as f:
        json.dump(loc_data, f)
