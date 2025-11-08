"""
Emotion Detection Model Training Script
Trains a CNN model on facial expressions to detect 7 emotions:
Happy, Sad, Angry, Disgusted, Fearful, Neutral, Surprised
"""

import numpy as np
import pandas as pd
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import cv2

# Define emotion classes
EMOTIONS = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

def build_emotion_model():
    """
    Build a Convolutional Neural Network for emotion detection.
    
    Model Architecture:
    - Input: 48x48 grayscale images
    - Conv layers with ReLU activation
    - MaxPooling for dimensionality reduction
    - Dense layers with dropout for regularization
    - Output: 7 emotion classes with softmax
    """
    model = Sequential()
    
    # First Convolutional Block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second Convolutional Block
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Flattening and Dense Layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(EMOTIONS), activation='softmax'))
    
    return model

def compile_model(model):
    """Compile the model with Adam optimizer."""
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    return model

def train_model_simple():
    """
    Simple training function using random data for demonstration.
    In production, use actual FER2013 dataset.
    """
    print("Building emotion detection model...")
    model = build_emotion_model()
    model = compile_model(model)
    
    print("Model Summary:")
    model.summary()
    
    # Generate dummy data for demonstration
    print("\nGenerating sample training data...")
    X_train = np.random.random((1000, 48, 48, 1)).astype('float32')
    y_train = np.random.randint(0, len(EMOTIONS), 1000)
    y_train = np.eye(len(EMOTIONS))[y_train]  # One-hot encoding
    
    X_val = np.random.random((200, 48, 48, 1)).astype('float32')
    y_val = np.random.randint(0, len(EMOTIONS), 200)
    y_val = np.eye(len(EMOTIONS))[y_val]
    
    # Normalize pixel values
    X_train /= 255.0
    X_val /= 255.0
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Save the model
    print("\nSaving model...")
    model.save('emotion_model.h5')
    
    # Save emotion labels
    with open('emotion_labels.pkl', 'wb') as f:
        pickle.dump(EMOTIONS, f)
    
    print("Model training completed and saved!")
    return model, history

def load_emotion_model():
    """Load pre-trained emotion model."""
    from tensorflow.keras.models import load_model
    try:
        model = load_model('emotion_model.h5')
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model...")
        model, _ = train_model_simple()
        return model

if __name__ == "__main__":
    # Train the model
    model, history = train_model_simple()
    print("\nModel training completed successfully!")
    print("Model saved as 'emotion_model.h5'")
    print("Emotion labels saved as 'emotion_labels.pkl'")