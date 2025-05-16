import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from typing import Tuple, Optional

class EMG_CNN:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize EMG CNN classifier
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        
        if model_path:
            self.load_model(model_path)
        
    def load_data(self, filepath):
        """Load and preprocess EMG data"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"EMG data file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Normalize and reshape for CNN
        X = self.scaler.fit_transform(X)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Encode labels
        y = self.le.fit_transform(y)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def build_model(self, input_shape, num_classes):
        """Build 1D CNN architecture"""
        self.model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'), 
            MaxPooling1D(2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    def train(self, filepath, epochs=10):
        """Train CNN model"""
        X_train, X_test, y_train, y_test = self.load_data(filepath)
        self.build_model((X_train.shape[1], 1), len(np.unique(y_train)))
        
        history = self.model.fit(X_train, y_train,
                               epochs=epochs,
                               validation_data=(X_test, y_test))
        
        # Save model and preprocessing objects
        self.model.save('emg_cnn_model.h5')
        joblib.dump(self.scaler, 'cnn_scaler.pkl')
        joblib.dump(self.le, 'label_encoder.pkl')
        
        return history

    def predict(self, filepath):
        """Make predictions on new EMG data"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prediction file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        X = df.values
        X = self.scaler.transform(X)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        preds = self.model.predict(X)
        return self.le.inverse_transform(np.argmax(preds, axis=1))
