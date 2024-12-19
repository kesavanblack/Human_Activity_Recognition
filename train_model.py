# train_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

def load_data():
    # Load your dataset here (e.g., videos, labels)
    # This is a placeholder function. Implement your data loading logic.
    return np.random.rand(100, 10, 1), np.random.randint(0, 3, size=(100,))

def create_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(10, 1), return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(3, activation='softmax'))  # Assuming 3 classes
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10, batch_size=16)  # Adjust epochs and batch size as needed

def save_model(model, model_path):
    model.save(model_path)

if __name__ == '__main__':
    # Load your training data
    X_train, y_train = load_data()

    # Create your model
    model = create_model()

    # Train the model
    train_model(model, X_train, y_train)

    # Define the path to save the model
    model_path = 'models/activity_model.h5'

    # Save the trained model
    save_model(model, model_path)
    print(f'Model saved at: {model_path}')
