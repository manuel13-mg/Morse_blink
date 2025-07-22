from Cam_integration import BlinkDetector
from MorseDecoder import MorseCodeDecoder
from Train import BlinkClassifier
from UserManager import UserManager
import cv2
import time
import numpy as np

blink_detector = BlinkDetector()
decoder = MorseCodeDecoder()
classifier = BlinkClassifier()
user_manager = UserManager()

BLINK_THRESHOLD = 4.5

class MorseCodeCommunicator:
    def __init__(self):
        self.code = ''
        self.message = ''
        self.last_blink_time = 0

    def process_blink(self, blink_ratio):
        if blink_ratio > BLINK_THRESHOLD:
            blink_type = classifier.predict(blink_ratio)
            if blink_type == 0:
                self.code += '.'
            else:
                self.code += '-'
            self.last_blink_time = time.time()

    def check_and_decode(self):
        if time.time() - self.last_blink_time > 2 and self.code:
            decoded_char = decoder.decode(self.code)
            self.message += decoded_char
            print("Decoded:", decoded_char, "| Message:", self.message)
            self.code = ''

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = blink_detector.get_landmarks(frame)
            if landmarks:
                blink_ratio = blink_detector.calculate_blink_ratio(landmarks)
                self.process_blink(blink_ratio)
                self.check_and_decode()

            cv2.putText(frame, f"Message: {self.message}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Morse Code Communicator", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    print("\n--- Blink Morse Communicator ---")
    print("1. Train new user")
    print("2. Communicate")
    print("3. Delete user")
    print("4. List users")
    print("5. Exit")

    while True:
        choice = input("\nEnter your choice: ")
        if choice == '1':
            username = input("Enter new username: ")
            X, y = [], []
            print("\n[INFO] Recording blinks. Press 'q' to stop.")
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                landmarks = blink_detector.get_landmarks(frame)
                if landmarks:
                    ratio = blink_detector.calculate_blink_ratio(landmarks)
                    cv2.putText(frame, f"Ratio: {ratio:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Recording", frame)
                    label = input("Label for current blink (0 for dot, 1 for dash): ")
                    if label in ['0', '1']:
                        X.append(ratio)
                        y.append([1, 0] if label == '0' else [0, 1])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            classifier.train(X, y)
            user_manager.save_user(username, classifier)
            print(f"Model saved for user '{username}'.")

        elif choice == '2':
            username = input("Enter your username: ")
            loaded_model = user_manager.load_user(username)
            if loaded_model:
                global classifier
                classifier = loaded_model
                communicator = MorseCodeCommunicator()
                communicator.run()
            else:
                print("User model not found.")

        elif choice == '3':
            username = input("Enter username to delete: ")
            user_manager.delete_user(username)
            print(f"Deleted model for user '{username}'.")

        elif choice == '4':
            users = user_manager.list_users()
            print("\nRegistered users:", users)

        elif choice == '5':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == '__main__':
    main()
