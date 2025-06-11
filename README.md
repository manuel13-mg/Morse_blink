# 👁️ Blink-Based Morse Code Communication System

A revolutionary communication system that allows users to send morse code messages using only eye blinks! Perfect for accessibility, silent communication, or just having fun with morse code.

## 🌟 Features

### Core Functionality
- **Eye Blink Detection**: Advanced computer vision using MediaPipe and dlib
- **Morse Code Translation**: Real-time conversion of blinks to morse code letters
- **Smart Classification**: AI-powered distinction between dots (short blinks) and dashes (long blinks)
- **User Training**: Personalized models for each user's unique blinking patterns
- **Visual Feedback**: Real-time display with timers and progress indicators

### User Experience
- **Beginner-Friendly**: Clear timing system with visual countdown
- **Multi-User Support**: Individual profiles with trained models
- **Real-Time Communication**: Live morse code to text conversion
- **Error Prevention**: Smart cooldown system prevents accidental detection

## 🚀 Quick Start

### Prerequisites
```bash
pip install opencv-python mediapipe tensorflow scikit-learn numpy dlib
```

### Installation
1. Clone or download this repository
2. Install required dependencies
3. Run the application:
```bash
python test_app.py
```

## 📖 How to Use

### 1. First Time Setup
1. **Create a New User**
   - Select option 3 from main menu
   - Enter your username
   - Complete the training process

2. **Training Process**
   - **Dot Training**: Make 20 short, quick blinks (< 0.4 seconds)
   - **Dash Training**: Make 20 long blinks (> 0.4 seconds)
   - The system learns your unique blinking patterns

### 2. Communication
1. **Select Your User** (option 2)
2. **Start Communication** (option 4)
3. **Begin Blinking**:
   - **Short blinks** = dots (.)
   - **Long blinks** = dashes (-)

### 3. Timing System
- **3-second cooldown** after each blink (prevents accidental detection)
- **5-second pause** to complete a letter
- **5-second pause** after letter to add a space
- Visual countdown shows remaining time

## 🎯 Morse Code Reference

### Common Letters
| Letter | Code | Letter | Code | Letter | Code |
|--------|------|--------|------|--------|------|
| A | .-   | J | .--- | S | ... |
| B | -... | K | -.-  | T | -   |
| C | -.-. | L | .-.. | U | ..- |
| D | -..  | M | --   | V | ...- |
| E | .    | N | -.   | W | .-- |
| F | ..-. | O | ---  | X | -..- |
| G | --.  | P | .--. | Y | -.-- |
| H | .... | Q | --.- | Z | --.. |
| I | ..   | R | .-.  |   |      |

### Numbers
| Number | Code | Number | Code |
|--------|------|--------|------|
| 0 | ----- | 5 | ..... |
| 1 | .---- | 6 | -.... |
| 2 | ..--- | 7 | --... |
| 3 | ...-- | 8 | ---.. |
| 4 | ....- | 9 | ----. |

## 🔧 System Architecture

### Components
1. **BlinkDetector**: Computer vision for eye blink detection
2. **BlinkClassifier**: AI model for dot/dash classification
3. **MorseCodeDecoder**: Converts morse sequences to letters
4. **UserManager**: Handles user profiles and training data
5. **MorseCodeCommunicator**: Main communication interface

### Detection Methods
- **Primary**: dlib facial landmark detection
- **Fallback**: MediaPipe face mesh detection
- **Enhancement**: Frame preprocessing for better detection

## 📊 Technical Specifications

### Timing Configuration
- **Blink Duration Range**: 0.05 - 3.0 seconds
- **Default Dot/Dash Threshold**: 0.4 seconds (customizable per user)
- **Cooldown Period**: 3 seconds
- **Letter Completion**: 5 seconds
- **Space Addition**: 5 seconds

### AI Model
- **Architecture**: Neural Network (TensorFlow/Keras)
- **Features**: Duration, intensity, minimum EAR, inverse duration
- **Training**: Personalized for each user
- **Fallback**: Threshold-based classification

## 🎮 Controls

### During Communication
- **q**: Quit communication mode
- **c**: Clear current message
- **Blink "EXIT"**: Exit communication mode via morse code

### Visual Indicators
- **Green Circle**: Ready for blinks
- **Orange Timer**: Cooldown active
- **Cyan Timer**: Ready period (thinking time)
- **Progress Bars**: Letter and space completion progress

## 🛠️ Troubleshooting

### Common Issues

**Blinks Not Detected**
- Ensure good lighting
- Position face clearly in camera view
- Check if camera is working
- Try retraining your user model

**Wrong Dot/Dash Classification**
- Retrain your user model
- Make more distinct short vs long blinks
- Check lighting conditions

**Camera Issues**
- Ensure camera permissions are granted
- Close other applications using camera
- Try different camera index in code

### Performance Tips
- Use consistent lighting
- Maintain steady head position
- Practice distinct short/long blinks
- Allow system to complete cooldown periods

## 📁 File Structure

```
Morse_blink/
├── test_app.py              # Main application
├── README.md                # This file
├── users/                   # User profiles directory
│   ├── users.json          # User database
│   └── [username]/         # Individual user data
│       ├── model.h5        # Trained neural network
│       └── data.pkl        # Model parameters
└── shape_predictor_68_face_landmarks.dat  # dlib model (auto-downloaded)
```

## 🔮 Future Enhancements

- [ ] Mobile app version
- [ ] Bluetooth connectivity
- [ ] Voice feedback
- [ ] Multiple language support
- [ ] Advanced morse code features (punctuation, numbers)
- [ ] Group communication
- [ ] Message history and logging

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Better computer vision algorithms
- Enhanced AI models
- User interface improvements
- Performance optimizations
- Additional features

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **MediaPipe**: Google's machine learning framework
- **dlib**: Facial landmark detection
- **OpenCV**: Computer vision library
- **TensorFlow**: Machine learning platform

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review the timing system explanation
3. Ensure proper training data collection
4. Verify camera and lighting setup

---

**Happy Blinking! 👁️✨**

*Transform your blinks into words, your eyes into a keyboard!*