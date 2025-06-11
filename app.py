import cv2
import numpy as np
import json
import os
import time
import pickle
import threading
from datetime import datetime
from collections import deque
from scipy.spatial import distance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

try:
    import winsound
    AUDIO_LIB = 'winsound'
except ImportError:
    try:
        import beepy
        AUDIO_LIB = 'beepy'
    except ImportError:
        AUDIO_LIB = None
        print("[Warning] No audio library found (winsound or beepy). Audio feedback will be disabled.")

class BlinkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            raise Exception(f"Fatal: Failed to initialize MediaPipe FaceMesh. Error: {e}")

        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 373, 374, 384, 385, 386, 387, 388, 466, 263]
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_EYE_EAR_INDICES = [145, 159, 160, 161, 144, 133]
        self.RIGHT_EYE_EAR_INDICES = [374, 386, 387, 388, 373, 362]
        self.EYE_AR_THRESH = 0.23
        self.EYE_AR_CONSEC_FRAMES = 2
        self.is_blinking = False
        self.blink_start_time = 0
        self.frame_counter = 0

    def _eye_aspect_ratio(self, eye_landmarks):
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def _get_eye_landmarks(self, landmarks, eye_indices, image_width, image_height):
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            eye_points.append([x, y])
        return np.array(eye_points)

    def calibrate_ear_threshold(self, cap, num_frames=100):
        print("\n[Calibration] Starting EAR threshold calibration...")
        print("Please keep your eyes open and look directly at the camera.")
        time.sleep(2)
        ear_values = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            progress_text = f"Calibrating... {i+1}/{num_frames}"
            cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Calibration', frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w = frame.shape[:2]
                    left_eye = self._get_eye_landmarks(face_landmarks, self.LEFT_EYE_EAR_INDICES, w, h)
                    right_eye = self._get_eye_landmarks(face_landmarks, self.RIGHT_EYE_EAR_INDICES, w, h)
                    left_ear = self._eye_aspect_ratio(left_eye)
                    right_ear = self._eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    ear_values.append(ear)
            if cv2.waitKey(1) & 0xFF == 27:
                print("[Calibration] Canceled by user.")
                break
        cv2.destroyAllWindows()
        if ear_values:
            avg_ear = np.mean(ear_values)
            self.EYE_AR_THRESH = avg_ear * 0.8
            print(f"[Calibration] Complete. Average EAR: {avg_ear:.3f}, Threshold set to: {self.EYE_AR_THRESH:.3f}")
        else:
            print("[Calibration] Failed. No face detected. Using default threshold.")

    def detect_blink(self, frame):
        try:
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            blink_info = None
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                left_eye = self._get_eye_landmarks(face_landmarks, self.LEFT_EYE_EAR_INDICES, w, h)
                right_eye = self._get_eye_landmarks(face_landmarks, self.RIGHT_EYE_EAR_INDICES, w, h)
                left_ear = self._eye_aspect_ratio(left_eye)
                right_ear = self._eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                if ear < self.EYE_AR_THRESH:
                    self.frame_counter += 1
                    if not self.is_blinking and self.frame_counter >= self.EYE_AR_CONSEC_FRAMES:
                        self.is_blinking = True
                        self.blink_start_time = time.time()
                else:
                    if self.is_blinking:
                        self.is_blinking = False
                        blink_duration = time.time() - self.blink_start_time
                        blink_info = {
                            'duration': blink_duration,
                            'intensity': 1.0 - ear,
                            'left_ear': left_ear,
                            'right_ear': right_ear,
                            'timestamp': time.time()
                        }
                    self.frame_counter = 0
            return blink_info
        except Exception as e:
            return None

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
            '.--.-.': '@', '...---...': 'SOS'
        }

    def decode(self, morse_sequence):
        return self.morse_code_dict.get(morse_sequence, '?')

class BlinkClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None

    def _create_model(self):
        model = Sequential([
            Dense(32, activation='relu', input_shape=(3,)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _prepare_features(self, blink_data):
        features = []
        for blink in blink_data:
            feature = [
                blink['duration'],
                blink['intensity'],
                abs(blink['left_ear'] - blink['right_ear'])
            ]
            features.append(feature)
        return np.array(features)

    def train(self, dot_blinks, dash_blinks):
        print("[Training] Preparing data...")
        try:
            dot_features = self._prepare_features(dot_blinks)
            dash_features = self._prepare_features(dash_blinks)
            X = np.vstack([dot_features, dash_features])
            y = np.hstack([np.zeros(len(dot_features)), np.ones(len(dash_features))])
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.model = self._create_model()
            print("[Training] Starting model training...")
            history = self.model.fit(
                X_scaled, y,
                epochs=100,
                batch_size=8,
                validation_split=0.2,
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
            )
            print("[Training] Model training complete.")
            loss, accuracy = self.model.evaluate(X_scaled, y, verbose=0)
            return loss, accuracy
        except Exception as e:
            print(f"Error during training: {e}")
            return None, None

    def predict(self, blink_data):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler is not initialized. Train or load a model first.")
        try:
            features = self._prepare_features([blink_data])
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled, verbose=0)[0][0]
            return 'dash' if prediction > 0.5 else 'dot'
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def save_model(self, filepath):
        try:
            if self.model is not None and self.scaler is not None:
                model_path = f"{filepath}_model.keras"
                scaler_path = f"{filepath}_scaler.pkl"
                self.model.save(model_path)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                print(f"Model saved to {model_path}")
            else:
                print("No model or scaler to save.")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath):
        try:
            model_path = f"{filepath}_model.keras"
            scaler_path = f"{filepath}_scaler.pkl"
            self.model = load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

class UserManager:
    def __init__(self, users_dir="users", users_file="users.json"):
        self.users_dir = users_dir
        self.users_file = users_file
        self._ensure_directories()
        self.users = self._load_users()

    def _ensure_directories(self):
        if not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir)

    def _load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {self.users_file}. Starting fresh.")
                return {}
        return {}

    def _save_users(self):
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=4)

    def add_user(self, username):
        if username in self.users:
            print(f"User '{username}' already exists.")
            return False
        self.users[username] = {
            'created_date': datetime.now().isoformat(),
            'model_path': os.path.join(self.users_dir, f"{username}_blink_model"),
            'trained': False
        }
        self._save_users()
        print(f"User '{username}' created.")
        return True

    def get_user(self, username):
        return self.users.get(username)

    def mark_user_trained(self, username):
        if username in self.users:
            self.users[username]['trained'] = True
            self._save_users()

    def list_users(self):
        return list(self.users.keys())

class MorseCodeCommunicator:
    def __init__(self):
        self.blink_detector = BlinkDetector()
        self.morse_decoder = MorseCodeDecoder()
        self.user_manager = UserManager()
        self.classifier = BlinkClassifier()
        self.current_user = None
        self.current_morse_sequence = ""
        self.last_blink_time = 0
        self.message_file = "messages.txt"
        self.LETTER_PAUSE = 1.5
        self.WORD_PAUSE = 3.0

    def _provide_audio_feedback(self, blink_type):
        if not AUDIO_LIB:
            return
        def play_sound():
            try:
                if AUDIO_LIB == 'winsound':
                    freq = 1200 if blink_type == 'dot' else 600
                    duration = 100
                    winsound.Beep(freq, duration)
                elif AUDIO_LIB == 'beepy':
                    sound_id = 1 if blink_type == 'dot' else 4
                    beepy.beep(sound=sound_id)
            except Exception as e:
                pass
        threading.Thread(target=play_sound).start()

    def _calculate_timing(self, blink_data):
        if not blink_data:
            return
        durations = [blink['duration'] for blink in blink_data]
        avg_duration = np.mean(durations)
        self.LETTER_PAUSE = max(1.0, avg_duration * 8)
        self.WORD_PAUSE = self.LETTER_PAUSE * 2
        print(f"[Timing] Adjusted timing: Avg Blink={avg_duration:.2f}s, Letter Pause={self.LETTER_PAUSE:.2f}s, Word Pause={self.WORD_PAUSE:.2f}s")

    def _collect_training_data(self, cap, data_list, blink_type, target_count):
        collected_count = 0
        while collected_count < target_count:
            ret, frame = cap.read()
            if not ret:
                break
            status_text = f"Collecting {blink_type.upper()} blinks: {collected_count}/{target_count}"
            prompt_text = f"Perform a {blink_type.lower()} blink and press SPACE"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, prompt_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press ESC to cancel", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Training', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Canceled data collection.")
                return False
            if key == ord(' '):
                print(f"Capturing {blink_type} blink #{collected_count + 1}...")
                blink_info = None
                for _ in range(100):
                    ret, frame_capture = cap.read()
                    if not ret: continue
                    blink_info = self.blink_detector.detect_blink(frame_capture)
                    if blink_info and 0.05 < blink_info['duration'] < 2.5:
                        break
                    time.sleep(0.03)
                if blink_info:
                    data_list.append(blink_info)
                    collected_count += 1
                    self._provide_audio_feedback(blink_type)
                    print(f"-> Collected {blink_type} blink (duration: {blink_info['duration']:.3f}s)")
                else:
                    print("-> No valid blink detected. Please try again.")
        return True
    
    def train_user(self, username):
        print(f"\n=== Training Mode for {username} ===")
        print("Ensure good lighting and that your face is clearly visible.")
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Fatal: Could not open webcam.")
            self.blink_detector.calibrate_ear_threshold(cap)
            dot_blinks, dash_blinks = [], []
            print("\n--- Step 1: Collecting DOT blinks (short blinks) ---")
            if not self._collect_training_data(cap, dot_blinks, "dot", 15):
                raise Exception("Training canceled.")
            print("\n--- Step 2: Collecting DASH blinks (long blinks) ---")
            if not self._collect_training_data(cap, dash_blinks, "dash", 15):
                raise Exception("Training canceled.")
            cap.release()
            cv2.destroyAllWindows()
            if len(dot_blinks) < 5 or len(dash_blinks) < 5:
                print("\nInsufficient training data collected. Please try again.")
                return False
            self._calculate_timing(dot_blinks + dash_blinks)
            loss, accuracy = self.classifier.train(dot_blinks, dash_blinks)
            if loss is None:
                return False
            user_info = self.user_manager.get_user(username)
            self.classifier.save_model(user_info['model_path'])
            self.user_manager.mark_user_trained(username)
            print(f"\n[Success] Training complete! Final Accuracy: {accuracy:.2%}")
            return True
        except Exception as e:
            print(f"An error occurred during training: {e}")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            return False

    def load_user_model(self, username):
        user_info = self.user_manager.get_user(username)
        if user_info and user_info['trained']:
            if self.classifier.load_model(user_info['model_path']):
                self.current_user = username
                print(f"User '{username}' selected.")
                return True
        print(f"User '{username}' needs to be trained first.")
        return False

    def communicate(self):
        if not self.current_user:
            print("Error: No user selected. Please select a user from the main menu.")
            return
        print(f"\n=== Communication Mode | User: {self.current_user} ===")
        print("Look at the camera and start blinking in Morse code.")
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Fatal: Could not open webcam.")
            message = ""
            self.current_morse_sequence = ""
            self.last_blink_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]

                blink_info = self.blink_detector.detect_blink(frame)
                if blink_info and 0.05 < blink_info['duration'] < 2.5:
                    blink_type = self.classifier.predict(blink_info)
                    if blink_type:
                        self.current_morse_sequence += '.' if blink_type == 'dot' else '-'
                        self._provide_audio_feedback(blink_type)
                        self.last_blink_time = time.time()
                time_since_last_blink = time.time() - self.last_blink_time
                if self.current_morse_sequence:
                    if time_since_last_blink > self.WORD_PAUSE:
                        letter = self.morse_decoder.decode(self.current_morse_sequence)
                        message += letter + " "
                        print(f"Word Decoded: '{self.current_morse_sequence}' -> {letter} ")
                        self.current_morse_sequence = ""
                        self.last_blink_time = time.time()
                    elif time_since_last_blink > self.LETTER_PAUSE:
                        letter = self.morse_decoder.decode(self.current_morse_sequence)
                        message += letter
                        print(f"Letter Decoded: '{self.current_morse_sequence}' -> {letter}")
                        self.current_morse_sequence = ""
                        self.last_blink_time = time.time()
                cv2.putText(frame, f"MESSAGE: {message}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"CURRENT: {self.current_morse_sequence}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, "Press 'ESC' to quit, 'c' to clear", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow('Morse Code Communicator', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27: # ESC key
                    break
                if key == ord('c'):
                    message, self.current_morse_sequence = "", ""
                    print("Message cleared.")
            if message.strip():
                print(f"\nFinal Message: {message}")
        except Exception as e:
            print(f"An error occurred during communication: {e}")
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()

def main():
    communicator = MorseCodeCommunicator()
    print("\n" + "="*50)
    print("   Blink-Based Morse Code Communication System")
    print("="*50 + "\n")
    while True:
        print("\n--- Main Menu ---")
        if communicator.current_user:
            print(f"Current User: {communicator.current_user}")
        print("1. List & Select User")
        print("2. Create New User")
        print("3. Train Current User")
        print("4. Start Communication")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ").strip()
        if choice == '1':
            users = communicator.user_manager.list_users()
            if not users:
                print("No users found. Please create one first.")
                continue
            print("\nAvailable Users:")
            for i, user in enumerate(users):
                user_info = communicator.user_manager.get_user(user)
                status = "Trained" if user_info['trained'] else "Not Trained"
                print(f"  {i+1}. {user} ({status})")
            try:
                user_choice = int(input("\nEnter user number to select: ")) - 1
                if 0 <= user_choice < len(users):
                    username = users[user_choice]
                    if communicator.user_manager.get_user(username)['trained']:
                        communicator.load_user_model(username)
                    else:
                        print(f"User '{username}' selected, but needs training.")
                        communicator.current_user = username
                else:
                    print("Invalid number.")
            except ValueError:
                print("Invalid input.")
        elif choice == '2':
            username = input("Enter new username: ").strip()
            if username and communicator.user_manager.add_user(username):
                communicator.current_user = username
                if input("Train this user now? (y/n): ").lower() == 'y':
                    communicator.train_user(username)
        elif choice == '3':
            if communicator.current_user:
                communicator.train_user(communicator.current_user)
            else:
                print("Please select a user first.")
        elif choice == '4':
            if communicator.current_user and communicator.user_manager.get_user(communicator.current_user)['trained']:
                 communicator.load_user_model(communicator.current_user)
                 communicator.communicate()
            else:
                print("Please select a trained user first.")
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")

if __name__ == "__main__":
    main()
