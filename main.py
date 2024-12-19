import cv2
import numpy as np
from collections import deque
from utils.model import load_model
from utils.inference import perform_inference

def preprocess_frame(frame):
    # Resize the frame to the input size required by the model
    frame_resized = cv2.resize(frame, (640, 480))  # Adjust to match your model input
    # Normalize the frame pixel values
    frame_normalized = frame_resized / 255.0  # Normalize pixel values (0-1 range)
    
    return frame_normalized

if __name__ == '__main__':
    # Load the trained model
    model = load_model('models/activity_model.h5')

    # Start video capture from the webcam
    cap = cv2.VideoCapture(0)

    # Initialize a deque to hold the last 10 frames
    frame_buffer = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Add the processed frame to the buffer
        frame_buffer.append(processed_frame)

        # If we have enough frames in the buffer, make a prediction
        if len(frame_buffer) == 10:
            # Convert deque to numpy array and reshape for the model
            input_data = np.array(frame_buffer)  # shape: (10, 640, 480, 3)
            input_data = input_data.reshape(1, 10, 640, 480, 3)  # shape: (1, 10, 640, 480, 3)

            # Perform inference to predict activity
            prediction = perform_inference(model, input_data)

            # Display the predicted activity on the frame
            cv2.putText(frame, f'Activity: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the current frame with the prediction
        cv2.imshow('Activity Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
