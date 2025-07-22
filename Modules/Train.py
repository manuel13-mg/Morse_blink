import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

class BlinkClassifier:
    def __init__(self):
        self.model_path = 'models/blink_classifier.h5'
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=1, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, X, y):
        self.model.fit(np.array(X), np.array(y), epochs=20, verbose=1)
        self.model.save(self.model_path)

    def predict(self, x):
        prediction = self.model.predict(np.array([x]))
        return np.argmax(prediction)
