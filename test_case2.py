import os
import warnings

# Suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage to avoid GPU warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import json
import time
from datetime import datetime
import mediapipe as mp
import dlib
from scipy.spatial import distance
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import threading
import queue
from collections import deque
import pickle

# Additional TensorFlow logging suppression
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Suppress MediaPipe warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

class BlinkDetector:
    def __init__(self):
        # Initialize dlib face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Try to load the shape predictor model
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print("Downloading dlib shape predictor model...")
            self._download_shape_predictor()
        
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Initialize MediaPipe as fallback
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Lower threshold for better detection
            min_tracking_confidence=0.3
        )
        
        # dlib eye landmark indices (68-point model)
        self.LEFT_EYE_POINTS = list(range(36, 42))   # Points 36-41
        self.RIGHT_EYE_POINTS = list(range(42, 48))  # Points 42-47
        
        # MediaPipe fallback indices
        self.LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]  
        self.RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
        
        # Adaptive thresholds
        self.base_ear_thresh = 0.21
        self.current_ear_thresh = self.base_ear_thresh
        self.ear_history = deque(maxlen=30)  # Track EAR history for adaptive threshold
        
        self.EYE_AR_CONSEC_FRAMES = 1
        self.counter = 0
        self.blink_detected = False
        self.blink_start_time = 0
        
        # Lighting adaptation
        self.brightness_history = deque(maxlen=10)
        self.use_enhancement = False
        
    def _download_shape_predictor(self):
        """Download the dlib shape predictor model if not available"""
        import urllib.request
        import bz2
        
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        print("Downloading shape predictor model (this may take a moment)...")
        
        try:
            urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
            
            # Extract the bz2 file
            with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2", 'rb') as f_in:
                with open("shape_predictor_68_face_landmarks.dat", 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Clean up
            os.remove("shape_predictor_68_face_landmarks.dat.bz2")
            print("Shape predictor model downloaded successfully!")
            
        except Exception as e:
            print(f"Failed to download shape predictor: {e}")
            print("Please download manually from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            raise
    
    def enhance_frame(self, frame):
        """Enhance frame for better detection in poor lighting"""
        # Calculate frame brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        self.brightness_history.append(brightness)
        
        # Determine if enhancement is needed
        avg_brightness = np.mean(self.brightness_history)
        self.use_enhancement = avg_brightness < 80  # Threshold for dark conditions
        
        if self.use_enhancement:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            
            # Enhance each channel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Additional brightness adjustment if very dark
            if avg_brightness < 50:
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=20)
            
            return enhanced
        
        return frame
    
    def eye_aspect_ratio_dlib(self, eye_landmarks):
        """Calculate EAR using dlib 6-point eye landmarks"""
        # Vertical distances
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        if C == 0:
            return 0
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def eye_aspect_ratio_mediapipe(self, eye_landmarks):
        """Calculate EAR for MediaPipe landmarks (fallback)"""
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])  
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])  
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])  
        
        if C == 0:
            return 0
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_eye_landmarks_mediapipe(self, landmarks, eye_indices, image_width, image_height):
        """Extract eye landmarks from MediaPipe and convert to pixel coordinates"""
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            eye_points.append([x, y])
        return np.array(eye_points)
    
    def adapt_threshold(self, current_ear):
        """Dynamically adapt EAR threshold based on recent history"""
        if current_ear is not None:
            self.ear_history.append(current_ear)
            
            if len(self.ear_history) >= 10:
                # Calculate adaptive threshold based on recent EAR values
                recent_ears = list(self.ear_history)[-10:]
                mean_ear = np.mean(recent_ears)
                std_ear = np.std(recent_ears)
                
                # Adaptive threshold: mean - 2*std, but within reasonable bounds
                adaptive_thresh = max(0.15, min(0.25, mean_ear - 2*std_ear))
                
                # Smooth threshold changes
                self.current_ear_thresh = 0.7 * self.current_ear_thresh + 0.3 * adaptive_thresh
    
    def detect_blink_dlib(self, frame):
        """Primary detection using dlib"""
        enhanced_frame = self.enhance_frame(frame)
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        
        if len(faces) > 0:
            face = faces[0]  # Use first detected face
            landmarks = self.predictor(gray, face)
            
            # Extract eye coordinates
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in self.LEFT_EYE_POINTS])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in self.RIGHT_EYE_POINTS])
            
            # Calculate EAR for both eyes
            left_ear = self.eye_aspect_ratio_dlib(left_eye)
            right_ear = self.eye_aspect_ratio_dlib(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            return ear, True
        
        return None, False
    
    def detect_blink_mediapipe(self, frame):
        """Fallback detection using MediaPipe"""
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
    
    def detect_blink(self, frame):
        """Main detection method with dlib primary and MediaPipe fallback"""
        blink_info = None
        current_ear = None
        
        # Try dlib first
        ear, dlib_success = self.detect_blink_dlib(frame)
        
        # Fallback to MediaPipe if dlib fails
        if not dlib_success:
            ear, mp_success = self.detect_blink_mediapipe(frame)
            if not mp_success:
                return None, None
        
        current_ear = ear
        
        # Adapt threshold based on current conditions
        self.adapt_threshold(current_ear)
        
        # Blink detection logic
        if ear is not None and ear < self.current_ear_thresh:
            self.counter += 1
            if not self.blink_detected:
                self.blink_start_time = time.time()
                self.blink_detected = True
        else:
            if self.counter >= self.EYE_AR_CONSEC_FRAMES and self.blink_detected:
                blink_duration = time.time() - self.blink_start_time
                # Only register reasonable duration blinks (increased max for dashes)
                if 0.05 < blink_duration < 3.0:
                    blink_info = {
                        'duration': blink_duration,
                        'intensity': max(0.01, self.current_ear_thresh - min(ear if ear else 0, self.current_ear_thresh)),
                        'timestamp': time.time(),
                        'min_ear': ear if ear else 0,
                        'enhanced': self.use_enhancement  # Track if enhancement was used
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

class BlinkClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.dot_threshold = 0.4  # Default threshold for dot/dash classification (increased for better distinction)
        
    def create_model(self):
        model = Sequential([
            Dense(32, activation='relu', input_shape=(4,)),  # Increased input features
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
                blink.get('min_ear', 0.2),  # Minimum EAR during blink
                1.0 / (blink['duration'] + 0.001)  # Inverse duration as feature
            ]
            features.append(feature)
        return np.array(features)
    
    def train(self, dot_blinks, dash_blinks):
        print(f"Training with {len(dot_blinks)} dots and {len(dash_blinks)} dashes")
        
        # Prepare training data
        dot_features = self.prepare_features(dot_blinks)
        dash_features = self.prepare_features(dash_blinks)
        
        X = np.vstack([dot_features, dash_features])
        y = np.hstack([np.zeros(len(dot_features)), np.ones(len(dash_features))])
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Check for NaN values
        if np.isnan(X).any():
            print("Warning: NaN values found in features, replacing with 0")
            X = np.nan_to_num(X)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate simple threshold as backup
        dot_durations = [b['duration'] for b in dot_blinks]
        dash_durations = [b['duration'] for b in dash_blinks]
        
        if dot_durations and dash_durations:
            self.dot_threshold = (max(dot_durations) + min(dash_durations)) / 2
            print(f"Backup threshold set to: {self.dot_threshold:.3f}s")
        
        # Create and train model
        self.model = self.create_model()
        
        try:
            history = self.model.fit(
                X_scaled, y, 
                epochs=50,  # Reduced epochs
                batch_size=min(8, len(X)), 
                verbose=1, 
                validation_split=0.2 if len(X) > 5 else 0
            )
            
            loss, accuracy = self.model.evaluate(X_scaled, y, verbose=0)
            print(f"Training completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            return loss, accuracy
            
        except Exception as e:
            print(f"Model training failed: {e}")
            print("Will use threshold-based classification")
            self.model = None
            return 0, 0.5
    
    def predict(self, blink_data):
        # Try ML model first
        if self.model is not None and self.scaler is not None:
            try:
                features = self.prepare_features([blink_data])
                features_scaled = self.scaler.transform(features)
                prediction = self.model.predict(features_scaled, verbose=0)[0][0]
                return 'dash' if prediction > 0.5 else 'dot'
            except Exception as e:
                print(f"Model prediction failed: {e}, using threshold")
        
        # Fallback to threshold-based classification
        return 'dash' if blink_data['duration'] > self.dot_threshold else 'dot'
    
    def save_model(self, filepath):
        try:
            if self.model is not None:
                self.model.save(f"{filepath}_model.h5")
            
            # Always save the scaler and threshold
            model_data = {
                'scaler': self.scaler,
                'dot_threshold': self.dot_threshold,
                'has_model': self.model is not None
            }
            with open(f"{filepath}_data.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model data saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        try:
            # Load model data
            with open(f"{filepath}_data.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data['scaler']
            self.dot_threshold = model_data['dot_threshold']
            
            # Try to load the neural network model
            if model_data['has_model']:
                try:
                    self.model = load_model(f"{filepath}_model.h5")
                    print("Neural network model loaded successfully")
                except:
                    print("Neural network model not found, using threshold method")
                    self.model = None
            else:
                self.model = None
                
            print(f"Classifier loaded with threshold: {self.dot_threshold:.3f}s")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

class UserManager:
    def __init__(self):
        self.users_dir = "users"
        self.users_file = "users.json"
        self.ensure_directories()
        self.users = self.load_users()
    
    def ensure_directories(self):
        if not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir)
    
    def load_users(self):
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_users(self):
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
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
    
    def list_users(self):
        return list(self.users.keys())

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
        self.last_letter_time = 0  # Track when last letter was completed
        self.LETTER_PAUSE = 5.0    # 5 seconds to complete a letter (2s more than cooldown)
        self.SPACE_PAUSE = 5.0     # 5 seconds after letter to add space
        
    def train_user(self, username):
        print(f"\n=== Training Model for {username} ===")
        print("Please position yourself in front of the camera.")
        print("Make sure your face is clearly visible and well-lit.")
        print("Look directly at the camera.")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
            
        # Test camera first
        print("\nTesting camera and face detection...")
        for i in range(30):  # Test for 1 second
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera")
                cap.release()
                return False
                
            blink_info, current_ear = self.blink_detector.detect_blink(frame)
            
            if current_ear is not None:
                cv2.putText(frame, f"EAR: {current_ear:.3f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Face detected! Press any key to continue", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected - adjust position", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) != -1:  # Any key pressed
                break
        else:
            if current_ear is None:
                print("Could not detect face. Please check lighting and camera position.")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        
        # Collect dot blinks
        print("\n--- Collecting DOT blinks ---")
        print("Perform SHORT, QUICK blinks (like normal blinking)")
        print("Each blink should be about 0.1-0.3 seconds")
        
        dot_blinks = []
        if not self.collect_training_data(cap, dot_blinks, "DOT", 15):
            cap.release()
            return False
        
        # Collect dash blinks
        print("\n--- Collecting DASH blinks ---")
        print("Perform LONG, DELIBERATE blinks (keep eyes closed longer)")
        print("Each blink should be about 0.5-1.5 seconds")
        
        dash_blinks = []
        if not self.collect_training_data(cap, dash_blinks, "DASH", 15):
            cap.release()
            return False
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(dot_blinks) >= 8 and len(dash_blinks) >= 8:
            print(f"\nCollected {len(dot_blinks)} dots and {len(dash_blinks)} dashes")
            print("Training model...")
            
            # Show collected data statistics
            dot_durations = [b['duration'] for b in dot_blinks]
            dash_durations = [b['duration'] for b in dash_blinks]
            
            print(f"Dot durations: {min(dot_durations):.3f}s - {max(dot_durations):.3f}s (avg: {np.mean(dot_durations):.3f}s)")
            print(f"Dash durations: {min(dash_durations):.3f}s - {max(dash_durations):.3f}s (avg: {np.mean(dash_durations):.3f}s)")
            
            loss, accuracy = self.classifier.train(dot_blinks, dash_blinks)
            
            # Save model
            user_info = self.user_manager.get_user(username)
            self.classifier.save_model(user_info['model_path'])
            self.user_manager.mark_user_trained(username)
            
            print(f"Training completed! Model saved.")
            return True
        else:
            print(f"Insufficient training data: {len(dot_blinks)} dots, {len(dash_blinks)} dashes")
            print("Need at least 8 of each type")
            return False
    
    def collect_training_data(self, cap, data_list, blink_type, target_count):
        collecting = False
        collected_count = 0
        instruction_shown = False
        cooldown_time = 0
        
        while collected_count < target_count:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera")
                return False
            
            current_time = time.time()
            
            # Display information
            cv2.putText(frame, f"Collecting {blink_type} blinks: {collected_count}/{target_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if not collecting and current_time > cooldown_time:
                cv2.putText(frame, "Press SPACE to start collecting next blink", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif collecting:
                cv2.putText(frame, f"Perform {blink_type} blink NOW!", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Press SPACE to cancel this attempt", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            elif current_time <= cooldown_time:
                remaining = int(cooldown_time - current_time) + 1
                cv2.putText(frame, f"Wait {remaining} seconds...", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(frame, "Press ESC to cancel training", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if collecting:
                blink_info, current_ear = self.blink_detector.detect_blink(frame)
                
                # Show current EAR
                if current_ear is not None:
                    cv2.putText(frame, f"EAR: {current_ear:.3f}", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if blink_info:
                    # Validate the blink duration with more forgiving ranges
                    duration = blink_info['duration']
                    if blink_type == "DOT" and 0.05 <= duration <= 0.4:
                        data_list.append(blink_info)
                        collected_count += 1
                        collecting = False
                        cooldown_time = current_time + 1.5  # 1.5 second cooldown
                        print(f"✓ Collected {blink_type} #{collected_count}: {duration:.3f}s")
                    elif blink_type == "DASH" and 0.3 <= duration <= 2.0:  # More forgiving range
                        data_list.append(blink_info)
                        collected_count += 1
                        collecting = False
                        cooldown_time = current_time + 1.5
                        print(f"✓ Collected {blink_type} #{collected_count}: {duration:.3f}s")
                    elif blink_type == "DOT" and duration > 0.4:
                        collecting = False
                        cooldown_time = current_time + 1
                        print(f"✗ Too long for DOT ({duration:.3f}s). Try shorter blink.")
                    elif blink_type == "DASH" and duration < 0.3:
                        collecting = False
                        cooldown_time = current_time + 1
                        print(f"✗ Too short for DASH ({duration:.3f}s). Try longer blink.")
                    elif blink_type == "DASH" and duration > 2.0:
                        collecting = False
                        cooldown_time = current_time + 1
                        print(f"✗ Too long for DASH ({duration:.3f}s). Try shorter blink.")
                    else:
                        collecting = False
                        cooldown_time = current_time + 1
                        print(f"✗ Invalid duration ({duration:.3f}s). Try again.")
            
            cv2.imshow('Training', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and current_time > cooldown_time:
                collecting = not collecting
                if collecting:
                    print(f"Ready to collect {blink_type} blink #{collected_count + 1}")
            elif key == 27:  # ESC
                print("Training cancelled by user")
                return False
        
        return True
    
    def load_user_model(self, username):
        user_info = self.user_manager.get_user(username)
        if user_info and user_info['trained']:
            if self.classifier.load_model(user_info['model_path']):
                self.current_user = username
                print(f"Model loaded for user: {username}")
                return True
            else:
                print(f"Failed to load model for user: {username}")
        return False
    
    def communicate(self):
        if not self.current_user:
            print("No user selected!")
            return
        
        print(f"\n=== Communication Mode - User: {self.current_user} ===")
        print("Start blinking to communicate!")
        print("Short blinks = dots, Long blinks = dashes")
        print("3-second delay after each blink to avoid accidental detection")
        print("Wait 5 seconds after last blink to complete a letter")
        print("Wait another 5 seconds after letter to add a space")
        print("Blink 'EXIT' to quit communication mode")
        print("Press 'q' to quit, 'c' to clear current message")
        
        cap = cv2.VideoCapture(0)
        message = ""
        blink_cooldown_end = 0  # Time when cooldown ends
        BLINK_COOLDOWN = 3.0    # 3 seconds cooldown after each blink
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            h, w = frame.shape[:2]
            
            # Display current message and morse sequence
            cv2.putText(frame, f"Message: {message}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Current: {self.current_morse_sequence}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 'c' to clear, Blink 'EXIT' to quit", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Check if message contains partial EXIT and highlight it
            message_upper = message.upper().replace(' ', '')  # Remove spaces for EXIT detection
            if message_upper.endswith('E'):
                cv2.putText(frame, "EXIT Progress: E", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            elif message_upper.endswith('EX'):
                cv2.putText(frame, "EXIT Progress: EX", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            elif message_upper.endswith('EXI'):
                cv2.putText(frame, "EXIT Progress: EXI", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            elif message_upper.endswith('EXIT'):
                cv2.putText(frame, "EXIT DETECTED! Exiting...", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show letter completion timer if there's a current sequence
            if self.current_morse_sequence and self.last_blink_time > 0:
                time_since_last_blink = current_time - self.last_blink_time
                time_until_letter_complete = self.LETTER_PAUSE - time_since_last_blink
                
                if time_until_letter_complete > 0:
                    cv2.putText(frame, f"Letter completes in: {time_until_letter_complete:.1f}s", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                else:
                    cv2.putText(frame, "Letter will complete soon...", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Show space counter if we completed a letter and waiting for space
            elif self.last_letter_time > 0 and not self.current_morse_sequence:
                time_since_last_letter = current_time - self.last_letter_time
                time_until_space = self.SPACE_PAUSE - time_since_last_letter
                
                if time_until_space > 0:
                    cv2.putText(frame, f"Space in: {time_until_space:.1f}s (keep eyes open)", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # Visual progress bar for space
                    progress = (self.SPACE_PAUSE - time_until_space) / self.SPACE_PAUSE
                    bar_width = 200
                    bar_height = 10
                    cv2.rectangle(frame, (10, 140), (10 + bar_width, 140 + bar_height), (50, 50, 50), -1)
                    cv2.rectangle(frame, (10, 140), (10 + int(bar_width * progress), 140 + bar_height), (0, 255, 255), -1)
                    cv2.putText(frame, f"Space Progress: {progress*100:.0f}%", 
                               (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Display cooldown timer on the right side
            if current_time < blink_cooldown_end:
                remaining_time = blink_cooldown_end - current_time
                timer_text = f"Cooldown: {remaining_time:.1f}s"
                timer_color = (0, 165, 255)  # Orange color for cooldown
                cv2.putText(frame, timer_text, 
                           (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)
                cv2.putText(frame, "Ignoring blinks...", 
                           (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, timer_color, 1)
                
                # Draw a countdown circle
                center = (w - 50, 100)
                radius = 25
                # Background circle
                cv2.circle(frame, center, radius, (50, 50, 50), -1)
                # Progress arc
                angle = int(360 * (1 - remaining_time / BLINK_COOLDOWN))
                cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, timer_color, 3)
                # Timer number in center
                cv2.putText(frame, f"{int(remaining_time) + 1}", 
                           (center[0] - 10, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Show ready status
                cv2.putText(frame, "Ready for blinks", 
                           (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Draw ready indicator
                center = (w - 50, 100)
                cv2.circle(frame, center, 25, (0, 255, 0), 3)
                cv2.putText(frame, "OK", 
                           (center[0] - 15, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Only detect blinks if cooldown period has passed
            if current_time >= blink_cooldown_end:
                blink_info, current_ear = self.blink_detector.detect_blink(frame)
                if blink_info:
                    # Classify blink
                    blink_type = self.classifier.predict(blink_info)
                    if blink_type == 'dot':
                        self.current_morse_sequence += '.'
                    elif blink_type == 'dash':
                        self.current_morse_sequence += '-'
                    
                    # Set cooldown period
                    blink_cooldown_end = current_time + BLINK_COOLDOWN
                    self.last_blink_time = current_time
                    self.last_letter_time = 0  # Reset space timer when new blink detected
                    print(f"Detected {blink_type} ({blink_info['duration']:.3f}s): {self.current_morse_sequence}")
                    print(f"Threshold: {self.classifier.dot_threshold:.3f}s | Duration: {blink_info['duration']:.3f}s")
                    print(f"Cooldown active for {BLINK_COOLDOWN} seconds...")
            
            # Check for letter completion based on timing
            # Only check if we have a sequence AND enough time has passed AND we're not in cooldown
            if (self.current_morse_sequence and 
                self.last_blink_time > 0 and 
                current_time >= blink_cooldown_end and 
                (current_time - self.last_blink_time) > self.LETTER_PAUSE):
                
                # Complete letter
                letter = self.morse_decoder.decode(self.current_morse_sequence)
                if letter != '?':
                    message += letter
                    print(f"Letter decoded: {self.current_morse_sequence} -> {letter}")
                    
                    # Check for EXIT command (ignore spaces)
                    message_clean = message.upper().replace(' ', '')
                    if message_clean.endswith('EXIT'):
                        print("\n=== EXIT command detected! ===")
                        print("Exiting communication mode...")
                        break
                        
                else:
                    print(f"Unknown morse code: {self.current_morse_sequence}")
                
                self.current_morse_sequence = ""
                self.last_letter_time = current_time  # Track when letter was completed
            
            # Check for space addition - only if no current sequence and enough time since last letter
            elif (not self.current_morse_sequence and 
                  self.last_letter_time > 0 and 
                  current_time >= blink_cooldown_end and 
                  (current_time - self.last_letter_time) > self.SPACE_PAUSE):
                
                # Add space and reset
                if message and not message.endswith(' '):  # Don't add multiple spaces
                    message += " "
                    print("Space added")
                self.last_letter_time = 0  # Reset space timer
            
            cv2.imshow('Communication', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                message = ""
                self.current_morse_sequence = ""
                self.last_blink_time = 0
                self.last_letter_time = 0
                blink_cooldown_end = 0  # Reset cooldown when clearing
                print("Message cleared")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if message.strip():
            print(f"\nFinal message: {message}")

def main():
    print("=== Blink-Based Morse Code Communication System ===")
    print("Using MediaPipe for face detection (no additional downloads required)")
    
    communicator = MorseCodeCommunicator()
    
    while True:
        print("\n=== Main Menu ===")
        print("1. List existing users")
        print("2. Select user")
        print("3. Create new user")
        print("4. Start communication")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
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
            if users:
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
                            print(f"User {username} needs training first!")
                    else:
                        print("Invalid user number.")
                except ValueError:
                    print("Please enter a valid number.")
            else:
                print("No users found. Create a new user first.")
        
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
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
