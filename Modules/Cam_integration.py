import cv2
import dlib
import mediapipe as mp
import numpy as np

class BlinkDetector:
    def __init__(self, use_mediapipe=True):
        self.use_mediapipe = use_mediapipe
        if use_mediapipe:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        else:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def get_landmarks(self, frame):
        if self.use_mediapipe:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 1)
            if rects:
                shape = self.predictor(gray, rects[0])
                return [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        return []

    def calculate_blink_ratio(self, landmarks):
        if self.use_mediapipe:
            left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
            left_ratio = self._get_ratio(left_eye)
            right_ratio = self._get_ratio(right_eye)
            return (left_ratio + right_ratio) / 2.0
        return 0

    def _get_ratio(self, eye):
        hor_len = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
        ver_len = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
        return hor_len / ver_len if ver_len != 0 else 0