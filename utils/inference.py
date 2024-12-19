# utils/inference.py
import numpy as np
import cv2

def preprocess_frame(frame):
    # Implement your preprocessing here (e.g., resizing, normalization)
    frame_resized = cv2.resize(frame, (640, 480))
    # Further preprocessing if necessary (e.g., normalization)
    return frame_resized

def perform_inference(model, processed_frame):
    # This function should implement the logic for predicting the activity
    # For example, reshape the input and make predictions
    input_data = np.expand_dims(processed_frame, axis=0)  # Add batch dimension
    prediction = model.predict(input_data)
    return np.argmax(prediction, axis=-1)  # Assuming a classification model
