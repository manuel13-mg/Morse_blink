# Set environment variables before imports to suppress TensorFlow warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=Warning)

import cv2
import numpy as np
import json
import time
from datetime import datetime
import mediapipe as mp
import dlib
from scipy.spatial import distance
import tensorflow as tf
import logging

# Suppress TensorFlow and related logs
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import sys
if not sys.warnoptions:
    sys.warnoptions = ['ignore']

from tf_keras.models import Sequential, load_model
from tf_keras.layers import Dense, Dropout
from tf_keras.optimizers import Adam
from collections import deque
import pickle

class BlinkDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print("Downloading dlib shape predictor model...")
            self._download_shape_predictor()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        self.LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
        self.base_ear_thresh = 0.21
        self.current_ear_thresh = self.base_ear_thresh
        self.ear_history = deque(maxlen=30)
        self.EYE_AR_CONSEC_FRAMES = 1
        self.counter = 0
        self.blink_detected = False
        self.blink_start_time = 0
        self.brightness_history = deque(maxlen=10)
        self.use_enhancement = False
    
    def _download_shape_predictor(self):
        import urllib.request
        import bz2
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        try:
            urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
            with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2", 'rb') as f_in:
                with open("shape_predictor_68_face_landmarks.dat", 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove("shape_predictor_68_face_landmarks.dat.bz2")
            print("Shape predictor model downloaded successfully!")
        except Exception as e:
            print(f"Failed to download shape predictor: {e}. Please ensure internet connection or provide the file manually.")
            raise
    
    def enhance_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        self.brightness_history.append(brightness)
        avg_brightness = np.mean(self.brightness_history)
        self.use_enhancement = avg_brightness < 80
        if self.use_enhancement:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            if avg_brightness < 50:
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=25)
            return enhanced
        return frame
    
    def eye_aspect_ratio_dlib(self, eye_landmarks):
        try:
            A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
            C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
            if C == 0:
                return 0
            ear = (A + B) / (2.0 * C)
            return ear
        except IndexError:
            return 0
    
    def eye_aspect_ratio_mediapipe(self, eye_landmarks):
        try:
            A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
            C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
            if C == 0:
                return 0
            ear = (A + B) / (2.0 * C)
            return ear
        except IndexError:
            return 0
    
    def get_eye_landmarks_mediapipe(self, landmarks, eye_indices, image_width, image_height):
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            eye_points.append([x, y])
        return np.array(eye_points)
    
    def adapt_threshold(self, current_ear):
        if current_ear is not None:
            self.ear_history.append(current_ear)
            if len(self.ear_history) >= 10:
                recent_ears = list(self.ear_history)[-10:]
                mean_ear = np.mean(recent_ears)
                std_ear = np.std(recent_ears)
                adaptive_thresh = max(0.15, min(0.25, mean_ear - 2*std_ear))
                self.current_ear_thresh = 0.7 * self.current_ear_thresh + 0.3 * adaptive_thresh
    
    def detect_blink_dlib(self, frame):
        try:
            enhanced_frame = self.enhance_frame(frame)
            gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            if len(faces) > 0:
                face = faces[0]
                landmarks = self.predictor(gray, face)
                left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in self.LEFT_EYE_POINTS])
                right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in self.RIGHT_EYE_POINTS])
                left_ear = self.eye_aspect_ratio_dlib(left_eye)
                right_ear = self.eye_aspect_ratio_dlib(right_eye)
                ear = (left_ear + right_ear) / 2.0
                return ear, True
            return None, False
        except Exception as e:
            print(f"dlib blink detection failed: {e}")
            return None, False
    
    def detect_blink_mediapipe(self, frame):
        try:
            enhanced_frame = self.enhance_frame(frame)
            rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w = enhanced_frame.shape[:2]
                    left_eye = self.get_eye_landmarks_mediapipe(face_landmarks, self.LEFT_EYE_EAR_INDICES, w, h)
                    right_eye = self.get_eye_landmarks_mediapipe(face_landmarks, self.RIGHT_EYE_EAR_INDICES, w, h)
                    left_ear = self.eye_aspect_ratio_mediapipe(left_eye)
                    right_ear = self.eye_aspect_ratio_mediapipe(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    return ear, True
            return None, False
        except Exception as e:
            print(f"MediaPipe blink detection failed: {e}")
            return None, False
    
    def detect_blink(self, frame):
        blink_info = None
        current_ear = None
        ear, dlib_success = self.detect_blink_dlib(frame)
        if not dlib_success:
            ear, mp_success = self.detect_blink_mediapipe(frame)
            if not mp_success:
                print("Warning: Both dlib and MediaPipe face detection failed. Check lighting or camera.")
                return None, None
        current_ear = ear
        self.adapt_threshold(current_ear)
        if ear is not None and ear < self.current_ear_thresh:
            self.counter += 1
            if not self.blink_detected:
                self.blink_start_time = time.time()
                self.blink_detected = True
        else:
            if self.counter >= self.EYE_AR_CONSEC_FRAMES and self.blink_detected:
                blink_duration = time.time() - self.blink_start_time
                if 0.05 < blink_duration < 3.0:
                    blink_info = {
                        'duration': blink_duration,
                        'intensity': max(0.01, self.current_ear_thresh - min(ear if ear else 0, self.current_ear_thresh)),
                        'timestamp': time.time(),
                        'min_ear': ear if ear else 0,
                        'enhanced': self.use_enhancement
                    }
            self.counter = 0
            self.blink_detected = False
        return blink_info, current_ear

class MorseCodeDecoder:
    def __init__(self):
        self.morse_code_dict = {
            '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
            '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
            '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
            '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
            '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
            '--..': 'Z', '.----': '1', '..---': '2', '...--': '3',
            '....-': '4', '.....': '5', '-....': '6', '--...': '7',
            '---..': '8', '----.': '9', '-----': '0', '--..--': ',',
            '.-.-.-': '.', '..--..': '?', '.----.': "'", '-.-.--': '!',
            '-..-.': '/', '-.--.': '(', '-.--.-': ')', '.-...': '&',
            '---...': ':', '-.-.-.': ';', '-...-': '=', '.-.-.': '+',
            '-....-': '-', '..--.-': '_', '.-..-.': '"', '...-..-': '$',
            '.--.-.': '@', '....-...': 'SOS'
        }
    
    def decode(self, morse_sequence):
        return self.morse_code_dict.get(morse_sequence, '?')

class UserManager:
    def __init__(self):
        self.users_dir = "users"
        self.users_file = os.path.join(self.users_dir, "users.json")
        self.ensure_directories()
        self.users = self.load_users()
    
    def ensure_directories(self):
        if not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir)
    
    def load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {self.users_file} is corrupted. Starting with empty user list.")
                return {}
        return {}
    
    def save_users(self):
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")
    
    def add_user(self, username):
        if username not in self.users:
            self.users[username] = {
                'created_date': datetime.now().isoformat(),
                'model_path': os.path.join(self.users_dir, f"{username}_model"),
                'trained': False
            }
            self.save_users()
            return True
        return False
    
    def get_user(self, username):
        return self.users.get(username)
    
    def mark_user_trained(self, username):
        if username in self.users:
            self.users[username]['trained'] = True
            self.save_users()
    
    def delete_user_model(self, username):
        if username in self.users:
            model_path = self.users[username]['model_path']
            model_file = f"{model_path}_model.h5"
            data_file = f"{model_path}_data.pkl"
            try:
                if os.path.exists(model_file):
                    os.remove(model_file)
                    print(f"Deleted model file: {model_file}")
                if os.path.exists(data_file):
                    os.remove(data_file)
                    print(f"Deleted data file: {data_file}")
                self.users[username]['trained'] = False
                self.save_users()
                return True
            except PermissionError as e:
                print(f"Permission error deleting model for {username}: {e}")
                return False
            except Exception as e:
                print(f"Error deleting model for {username}: {e}")
                return False
        print(f"User {username} not found.")
        return False
    
    def list_users(self):
        return list(self.users.keys())

class BlinkClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.dot_threshold = 0.4
    
    def create_model(self):
        model = Sequential([
            Dense(32, activation='relu', input_shape=(4,)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def prepare_features(self, blink_data):
        features = []
        for blink in blink_data:
            feature = [
                blink['duration'],
                blink['intensity'],
                blink.get('min_ear', 0.2),
                1.0 / (blink['duration'] + 0.001)
            ]
            features.append(feature)
        return np.array(features)
    
    def train(self, dot_blinks, dash_blinks):
        print(f"Training with {len(dot_blinks)} dots and {len(dash_blinks)} dashes")
        dot_durations = [b['duration'] for b in dot_blinks]
        dash_durations = [b['duration'] for b in dash_blinks]
        if dot_durations and dash_durations:
            dot_mean = np.mean(dot_durations)
            dash_mean = np.mean(dash_durations)
            print(f"Dot mean duration: {dot_mean:.3f}s, Dash mean duration: {dash_mean:.3f}s")
            if dash_mean <= dot_mean + 0.1:
                print("Warning: Dash durations too close to dot durations.")
        dot_features = self.prepare_features(dot_blinks)
        dash_features = self.prepare_features(dash_blinks)
        X = np.vstack([dot_features, dash_features])
        y = np.hstack([np.zeros(len(dot_features)), np.ones(len(dash_features))])
        if np.isnan(X).any():
            print("Warning: NaN values found in features, replacing with 0")
            X = np.nan_to_num(X)
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        if dot_durations and dash_durations:
            self.dot_threshold = (max(dot_durations) + min(dash_durations)) / 2
            print(f"Backup threshold set to: {self.dot_threshold:.3f}s")
        self.model = self.create_model()
        try:
            self.model.fit(
                X_scaled, y,
                epochs=50,
                batch_size=min(8, len(X)),
                verbose=0,
                validation_split=0.2 if len(X) > 5 else 0
            )
            loss, accuracy = self.model.evaluate(X_scaled, y, verbose=0)
            print(f"Training completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            return loss, accuracy
        except Exception as e:
            print(f"Model training failed: {e}")
            self.model = None
            return 0, 0.5
    
    def predict(self, blink_data):
        if self.model is not None and self.scaler is not None:
            try:
                features = self.prepare_features([blink_data])
                features_scaled = self.scaler.transform(features)
                prediction = self.model.predict(features_scaled, verbose=0)[0][0]
                return 'dash' if prediction > 0.5 else 'dot'
            except Exception as e:
                print(f"Model prediction failed: {e}, using threshold")
        return 'dash' if blink_data['duration'] > self.dot_threshold else 'dot'
    
    def save_model(self, filepath):
        try:
            if self.model is not None:
                self.model.save(f"{filepath}_model.h5")
            model_data = {
                'scaler': self.scaler,
                'dot_threshold': self.dot_threshold,
                'has_model': self.model is not None
            }
            with open(f"{filepath}_data.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        try:
            with open(f"{filepath}_data.pkl", 'rb') as f:
                model_data = pickle.load(f)
            if not isinstance(model_data, dict) or 'scaler' not in model_data:
                print(f"Error: {filepath}_data.pkl is corrupted.")
                return False
            self.scaler = model_data['scaler']
            self.dot_threshold = model_data['dot_threshold']
            if model_data.get('has_model'):
                try:
                    self.model = load_model(f"{filepath}_model.h5")
                    print("Neural network model loaded")
                except:
                    print("Neural network model not found, using threshold method")
                    self.model = None
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

class MorseCodeCommunicator:
    def __init__(self):
        self.blink_detector = BlinkDetector()
        self.morse_decoder = MorseCodeDecoder()
        self.user_manager = UserManager()
        self.current_user = None
        self.classifier = BlinkClassifier()
        self.current_morse_sequence = ""
        self.blink_buffer = deque(maxlen=100)
        self.last_blink_time = 0
        self.last_letter_time = 0
        self.LETTER_PAUSE = 5.0
        self.SPACE_PAUSE = 5.0
    
    def train_user(self, username):
        print(f"\n=== Training Model for {username} ===")
        print("Position yourself in front of the camera with good lighting.")
        try:
            cap = None
            for cam_index in [0, 1]:  # Try default and secondary camera
                cap = cv2.VideoCapture(cam_index)
                if cap.isOpened():
                    break
            if not cap or not cap.isOpened():
                raise Exception("Could not open any camera. Check connections.")
            print("\nTesting camera and face detection...")
            for i in range(30):
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Could not read from camera")
                blink_info, current_ear = self.blink_detector.detect_blink(frame)
                if current_ear is not None:
                    cv2.putText(frame, f"EAR: {current_ear:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Face detected! Press SPACE to continue", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No face detected - adjust position", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Camera Test', frame)
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break
            else:
                if current_ear is None:
                    raise Exception("No face detected. Check lighting and camera.")
            if cv2.getWindowProperty('Camera Test', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow('Camera Test')
            print("\n--- Collecting DOT blinks ---")
            print("Perform SHORT, QUICK blinks (0.1-0.4 seconds)")
            dot_blinks = []
            if not self.collect_training_data(cap, dot_blinks, "dot", 15):
                raise Exception("Dot collection cancelled")
            print("\n--- Collecting DASH blinks ---")
            print("Perform LONG, DELIBERATE blinks (0.5-3.0 seconds)")
            dash_blinks = []
            if not self.collect_training_data(cap, dash_blinks, "dash", 15):
                raise Exception("Dash collection cancelled")
            dot_durations = [b['duration'] for b in dot_blinks]
            dash_durations = [b['duration'] for b in dash_blinks]
            print(f"\nCollected {len(dot_blinks)} dots and {len(dash_blinks)} dashes")
            print(f"Dot durations: {min(dot_durations):.3f}s - {max(dot_durations):.3f}s (avg: {np.mean(dot_durations):.3f}s)")
            print(f"Dash durations: {min(dash_durations):.3f}s - {max(dash_durations):.3f}s (avg: {np.mean(dash_durations):.3f}s)")
            if len(dot_blinks) >= 8 and len(dash_blinks) >= 8:
                print("\nTraining model...")
                loss, accuracy = self.classifier.train(dot_blinks, dash_blinks)
                if loss is None or accuracy is None:
                    raise Exception("Training failed")
                user_info = self.user_manager.get_user(username)
                self.classifier.save_model(user_info['model_path'])
                self.user_manager.mark_user_trained(username)
                print(f"Training completed! Accuracy: {accuracy:.2%}")
                return True
            else:
                print(f"Insufficient data: {len(dot_blinks)} dots, {len(dash_blinks)} dashes")
                return False
        except Exception as e:
            print(f"Training error: {e}")
            return False
        finally:
            if cap is not None:
                cap.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass
    
    def collect_training_data(self, cap, data_list, blink_type, target_count):
        collecting = False
        collected_count = 0
        cooldown_time = 0
        last_duration = None
        while collected_count < target_count:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera")
                return False
            current_time = time.time()
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Collecting {blink_type.upper()} blinks: {collected_count}/{target_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if not collecting and current_time > cooldown_time:
                cv2.putText(frame, "Press SPACE to start collecting next blink",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"{'Quick blink (0.1-0.4s)' if blink_type == 'dot' else 'Long blink (0.5-3.0s)'}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            elif collecting:
                cv2.putText(frame, f"Perform {blink_type.upper()} blink NOW!",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Press SPACE to cancel this attempt",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            elif current_time <= cooldown_time:
                remaining = int(cooldown_time - current_time) + 1
                cv2.putText(frame, f"Wait {remaining} seconds...",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            if last_duration is not None:
                cv2.putText(frame, f"Last blink: {last_duration:.3f}s",
                            (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press ESC to cancel training",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if collecting:
                blink_info, current_ear = self.blink_detector.detect_blink(frame)
                if current_ear is not None:
                    cv2.putText(frame, f"EAR: {current_ear:.3f}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if blink_info:
                    duration = blink_info['duration']
                    last_duration = duration
                    if blink_type == "dot" and 0.05 <= duration <= 0.4:
                        data_list.append(blink_info)
                        collected_count += 1
                        cooldown_time = current_time + 1.5
                        print(f"Collected {blink_type.upper()} #{collected_count}: {duration:.3f}s")
                    elif blink_type == "dash" and 0.5 <= duration <= 3.0:
                        data_list.append(blink_info)
                        collected_count += 1
                        cooldown_time = current_time + 2.0
                        print(f"Collected {blink_type.upper()} #{collected_count}: {duration:.3f}s")
                    elif blink_type == "dot" and duration > 0.4:
                        print(f"Too long for DOT ({duration:.3f}s). Try a quicker blink (0.1-0.4s).")
                    elif blink_type == "dot" and duration < 0.05:
                        print(f"Too short for DOT ({duration:.3f}s). Try a slightly longer blink (0.1-0.4s).")
                    elif blink_type == "dash" and duration < 0.5:
                        print(f"Too short for DASH ({duration:.3f}s). Hold eyes closed longer (0.5-3.0s).")
                    elif blink_type == "dash" and duration > 3.0:
                        print(f"Too long for DASH ({duration:.3f}s). Try a shorter blink (0.5-3.0s).")
                    else:
                        print(f"Invalid duration for {blink_type.upper()} ({duration:.3f}s).")
                    collecting = False
                    cooldown_time = max(cooldown_time, current_time + 1.0)
            cv2.imshow('Training', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and current_time > cooldown_time:
                collecting = not collecting
                if collecting:
                    print(f"Ready to collect {blink_type} blink #{collected_count + 1}")
            elif key == 27:
                print("Training cancelled")
                return False
        return True
    
    def load_user_model(self, username):
        user_info = self.user_manager.get_user(username)
        if user_info and user_info['trained']:
            if self.classifier.load_model(user_info['model_path']):
                self.current_user = username
                print(f"Model loaded for user: {username}")
                return True
            print(f"Failed to load model for user: {username}")
            return False
        print(f"User {username} needs training first.")
        return False
    
    def communicate(self):
        if not self.current_user:
            print("No user selected!")
            return
        print(f"\n=== Communication Mode - User: {self.current_user} ===")
        print("Short blinks = dots (0.1-0.4s), Long blinks = dashes (0.5-3.0s)")
        print("3-second delay after each blink")
        print("Wait 5 seconds after last blink to complete a letter")
        print("Wait another 5 seconds to add a space")
        print("Blink 'M' to quit")
        print("Press 'q' to quit, 'c' to clear")
        cap = None
        try:
            for cam_index in [0, 1]:
                cap = cv2.VideoCapture(cam_index)
                if cap.isOpened():
                    break
            if not cap or not cap.isOpened():
                raise Exception("Could not open webcam")
            message = ""
            blink_cooldown_end = 0
            BLINK_COOLDOWN = 3.0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                current_time = time.time()
                h, w = frame.shape[:2]
                cv2.putText(frame, f"Message: {message}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Current: {self.current_morse_sequence}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit, 'c' to clear, Blink 'M' to quit",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                message_upper = message.upper().replace(' ', '')
                if message_upper.endswith('M'):
                    cv2.putText(frame, "EXIT DETECTED! Exiting...", (10, h-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                if self.current_morse_sequence and self.last_blink_time > 0:
                    time_since_last_blink = current_time - self.last_blink_time
                    time_until_letter_complete = self.LETTER_PAUSE - time_since_last_blink
                    if time_until_letter_complete > 0:
                        cv2.putText(frame, f"Letter completes in: {time_until_letter_complete:.1f}s",
                                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    else:
                        cv2.putText(frame, "Letter will complete soon...",
                                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                elif self.last_letter_time > 0 and not self.current_morse_sequence:
                    time_since_last_letter = current_time - self.last_letter_time
                    time_until_space = self.SPACE_PAUSE - time_since_last_letter
                    if time_until_space > 0:
                        cv2.putText(frame, f"Space in: {time_until_space:.1f}s (keep eyes open)",
                                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        progress = (self.SPACE_PAUSE - time_until_space) / self.SPACE_PAUSE
                        bar_width = 200
                        bar_height = 10
                        cv2.rectangle(frame, (10, 140), (10 + bar_width, 140 + bar_height), (50, 50, 50), -1)
                        cv2.rectangle(frame, (10, 140), (10 + int(bar_width * progress), 140 + bar_height), (0, 255, 255), -1)
                        cv2.putText(frame, f"Space Progress: {progress*100:.0f}%",
                                    (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if current_time < blink_cooldown_end:
                    remaining_time = blink_cooldown_end - current_time
                    cv2.putText(frame, f"Cooldown: {remaining_time:.1f}s",
                                (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    cv2.putText(frame, "Ignoring blinks...",
                                (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                    center = (w - 50, 100)
                    radius = 25
                    cv2.circle(frame, center, radius, (50, 50, 50), -1)
                    angle = int(360 * (1 - remaining_time / BLINK_COOLDOWN))
                    cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (255, 165, 0), 3)
                    cv2.putText(frame, f"{int(remaining_time) + 1}",
                                (center[0] - 10, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Ready for blinks",
                                (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    center = (w - 50, 100)
                    cv2.circle(frame, center, 25, (0, 255, 0), 3)
                    cv2.putText(frame, "OK",
                                (center[0] - 15, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if current_time >= blink_cooldown_end:
                    blink_info, current_ear = self.blink_detector.detect_blink(frame)
                    if blink_info:
                        blink_type = self.classifier.predict(blink_info)
                        if blink_type == 'dot':
                            self.current_morse_sequence += '.'
                        elif blink_type == 'dash':
                            self.current_morse_sequence += '-'
                        blink_cooldown_end = current_time + BLINK_COOLDOWN
                        self.last_blink_time = current_time
                        self.last_letter_time = 0
                        print(f"Detected {blink_type} ({blink_info['duration']:.3f}s): {self.current_morse_sequence}")
                if (self.current_morse_sequence and
                    self.last_blink_time > 0 and
                    current_time >= blink_cooldown_end and
                    (current_time - self.last_blink_time) > self.LETTER_PAUSE):
                    letter = self.morse_decoder.decode(self.current_morse_sequence)
                    if letter != '?':
                        message += letter
                        print(f"Letter decoded: {self.current_morse_sequence} -> {letter}")
                        if message.upper().replace(' ', '').endswith('M'):
                            print("\n=== EXIT command detected! ===")
                            break
                    else:
                        print(f"Unknown morse code: {self.current_morse_sequence}")
                    self.current_morse_sequence = ""
                    self.last_letter_time = current_time
                elif (not self.current_morse_sequence and
                      self.last_letter_time > 0 and
                      current_time >= blink_cooldown_end and
                      (current_time - self.last_letter_time) > self.SPACE_PAUSE):
                    if message and not message.endswith(' ') and self.last_letter_time > 0:
                        message += " "
                        print("Space added")
                    self.last_letter_time = 0
                cv2.imshow('Communication', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    message = ""
                    self.current_morse_sequence = ""
                    self.last_blink_time = 0
                    self.last_letter_time = 0
                    blink_cooldown_end = 0
                    print("Message cleared")
        except Exception as e:
            print(f"Communication error: {e}")
        finally:
            if cap is not None:
                cap.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass
        if message.strip():
            print(f"\nFinal message: {message}")

def main():
    print("=== Blink-Based Morse Code Communication System ===")
    communicator = MorseCodeCommunicator()
    while True:
        print("\n=== Main Menu ===")
        print("1. List existing users")
        print("2. Select user")
        print("3. Create new user")
        print("4. Start communication")
        print("5. Retrain existing user")
        print("6. Exit")
        choice = input("\nEnter your choice (1-6): ").strip()
        if choice == '1':
            users = communicator.user_manager.list_users()
            if users:
                print("\nExisting users:")
                for i, user in enumerate(users, 1):
                    user_info = communicator.user_manager.get_user(user)
                    status = "Trained" if user_info['trained'] else "Not trained"
                    print(f"{i}. {user} ({status})")
            else:
                print("No users found.")
        elif choice == '2':
            users = communicator.user_manager.list_users()
            if not users:
                print("No users found. Create a new user first.")
                continue
            print("\nAvailable users:")
            for i, user in enumerate(users, 1):
                user_info = communicator.user_manager.get_user(user)
                status = "Trained" if user_info['trained'] else "Not trained"
                print(f"{i}. {user} ({status})")
            try:
                user_choice = int(input("\nSelect user number: ")) - 1
                if 0 <= user_choice < len(users):
                    username = users[user_choice]
                    if communicator.load_user_model(username):
                        print(f"User {username} selected successfully!")
                else:
                    print("Invalid user number.")
            except ValueError:
                print("Please enter a valid number.")
        elif choice == '3':
            username = input("Enter new username: ").strip()
            if username:
                if communicator.user_manager.add_user(username):
                    print(f"User {username} created successfully!")
                    train_choice = input("Train the model now? (y/n): ").strip().lower()
                    if train_choice == 'y':
                        if communicator.train_user(username):
                            communicator.load_user_model(username)
                else:
                    print("User already exists!")
            else:
                print("Username cannot be empty.")
        elif choice == '4':
            if communicator.current_user:
                communicator.communicate()
            else:
                print("Please select a user first!")
        elif choice == '5':
            users = communicator.user_manager.list_users()
            if not users:
                print("No users found.")
                continue
            print("\nSelect user to retrain:")
            for i, user in enumerate(users, 1):
                user_info = communicator.user_manager.get_user(user)
                status = "Trained" if user_info['trained'] else "Not trained"
                print(f"{i}. {user} ({status})")
            try:
                user_choice = int(input("\nSelect user number to retrain: ")) - 1
                if 0 <= user_choice < len(users):
                    username = users[user_choice]
                    print(f"\nDeleting existing model for {username}...")
                    if not communicator.user_manager.delete_user_model(username):
                        print(f"Cannot proceed with training due to deletion failure.")
                        continue
                    print(f"Training new model for {username}...")
                    if communicator.train_user(username):
                        if communicator.load_user_model(username):
                            print(f"User {username} retrained successfully!")
                    else:
                        print(f"Failed to retrain user {username}.")
                else:
                    print("Invalid user number.")
            except ValueError:
                print("Please enter a valid number.")
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
