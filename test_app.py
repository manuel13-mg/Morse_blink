import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
import mediapipe as mp
from scipy.spatial import distance
import tensorflow as tf
from tf_keras.models import Sequential, load_model
from tf_keras.layers import Dense, Dropout
from tf_keras.optimizers import Adam
import threading
import queue
from collections import deque
import pickle

class BlinkDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # More specific indices for EAR calculation (left eye)
        self.LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]  
        # More specific indices for EAR calculation (right eye)
        self.RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
        
        self.EYE_AR_THRESH = 0.21  # Adjusted threshold
        self.EYE_AR_CONSEC_FRAMES = 1  # Reduced for better sensitivity
        self.counter = 0
        self.blink_detected = False
        self.blink_start_time = 0
        
    def eye_aspect_ratio(self, eye_landmarks):
        # Calculate EAR for MediaPipe landmarks
        # Vertical distances (top to bottom)
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])  
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])  
        
        # Horizontal distance (left to right)
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])  
        
        # EAR formula
        if C == 0:
            return 0
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_eye_landmarks(self, landmarks, eye_indices, image_width, image_height):
        """Extract eye landmarks and convert to pixel coordinates"""
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            eye_points.append([x, y])
        return np.array(eye_points)
    
    def detect_blink(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        blink_info = None
        current_ear = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                
                # Get eye landmarks
                left_eye = self.get_eye_landmarks(face_landmarks, self.LEFT_EYE_EAR_INDICES, w, h)
                right_eye = self.get_eye_landmarks(face_landmarks, self.RIGHT_EYE_EAR_INDICES, w, h)
                
                # Calculate EAR for both eyes
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                current_ear = ear
                
                # Blink detection logic
                if ear < self.EYE_AR_THRESH:
                    self.counter += 1
                    if not self.blink_detected:
                        self.blink_start_time = time.time()
                        self.blink_detected = True
                else:
                    if self.counter >= self.EYE_AR_CONSEC_FRAMES and self.blink_detected:
                        blink_duration = time.time() - self.blink_start_time
                        # Only register blinks that are reasonable duration
                        if 0.05 < blink_duration < 2.0:  # Between 50ms and 2 seconds
                            blink_info = {
                                'duration': blink_duration,
                                'intensity': max(0.01, self.EYE_AR_THRESH - min(ear, self.EYE_AR_THRESH)),
                                'timestamp': time.time(),
                                'min_ear': ear
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
            '.--.-.': '@', '...---...': 'SOS'
        }
    
    def decode(self, morse_sequence):
        return self.morse_code_dict.get(morse_sequence, '?')

class BlinkClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.dot_threshold = 0.3  # Default threshold for dot/dash classification
        
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
        self.LETTER_PAUSE = 2.0
        self.WORD_PAUSE = 4.0
        
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
        print("Each blink should be about 0.1-0.2 seconds")
        
        dot_blinks = []
        if not self.collect_training_data(cap, dot_blinks, "DOT", 15):
            cap.release()
            return False
        
        # Collect dash blinks
        print("\n--- Collecting DASH blinks ---")
        print("Perform LONG, DELIBERATE blinks (keep eyes closed longer)")
        print("Each blink should be about 0.4-0.8 seconds")
        
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
                    # Validate the blink duration
                    duration = blink_info['duration']
                    if blink_type == "DOT" and 0.05 <= duration <= 0.35:
                        data_list.append(blink_info)
                        collected_count += 1
                        collecting = False
                        cooldown_time = current_time + 1.5  # 1.5 second cooldown
                        print(f"✓ Collected {blink_type} #{collected_count}: {duration:.3f}s")
                    elif blink_type == "DASH" and 0.35 <= duration <= 1.2:
                        data_list.append(blink_info)
                        collected_count += 1
                        collecting = False
                        cooldown_time = current_time + 1.5
                        print(f"✓ Collected {blink_type} #{collected_count}: {duration:.3f}s")
                    elif blink_type == "DOT" and duration > 0.35:
                        collecting = False
                        cooldown_time = current_time + 1
                        print(f"✗ Too long for DOT ({duration:.3f}s). Try shorter blink.")
                    elif blink_type == "DASH" and duration < 0.35:
                        collecting = False
                        cooldown_time = current_time + 1
                        print(f"✗ Too short for DASH ({duration:.3f}s). Try longer blink.")
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
        print("Pause between letters, longer pause between words")
        print("Press 'q' to quit, 'c' to clear current message")
        
        cap = cv2.VideoCapture(0)
        message = ""
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display current message and morse sequence
            cv2.putText(frame, f"Message: {message}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Current: {self.current_morse_sequence}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 'c' to clear", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Detect blinks
            blink_info, current_ear = self.blink_detector.detect_blink(frame)
            if blink_info:
                # Classify blink
                blink_type = self.classifier.predict(blink_info)
                if blink_type == 'dot':
                    self.current_morse_sequence += '.'
                elif blink_type == 'dash':
                    self.current_morse_sequence += '-'
                
                self.last_blink_time = time.time()
                print(f"Detected {blink_type} ({blink_info['duration']:.3f}s): {self.current_morse_sequence}")
            
            # Check for letter/word completion based on timing
            current_time = time.time()
            if self.current_morse_sequence and (current_time - self.last_blink_time) > self.LETTER_PAUSE:
                # Complete letter
                letter = self.morse_decoder.decode(self.current_morse_sequence)
                if letter != '?':
                    message += letter
                    print(f"Letter decoded: {self.current_morse_sequence} -> {letter}")
                else:
                    print(f"Unknown morse code: {self.current_morse_sequence}")
                
                self.current_morse_sequence = ""
                
                # Check for word completion
                if (current_time - self.last_blink_time) > self.WORD_PAUSE:
                    message += " "
                    print("Word completed")
            
            cv2.imshow('Communication', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                message = ""
                self.current_morse_sequence = ""
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
