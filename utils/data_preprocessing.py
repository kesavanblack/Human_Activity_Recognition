import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Initialize Mediapipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_pose_keypoints(frame):
    """
    Extract pose keypoints from a frame using MediaPipe.
    Returns a flat array of (x, y, z) coordinates for each keypoint.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        return keypoints
    else:
        return None

def load_data_from_videos(video_paths, labels, sequence_length=30):
    """
    Process videos to extract pose keypoints, structure them as sequences, 
    and return X (sequences) and y (activity labels) for training.
    """
    X, y = [], []
    
    for i, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keypoints from the frame
            keypoints = extract_pose_keypoints(frame)
            if keypoints:
                frames.append(keypoints)
            
            # Once enough frames for a sequence are gathered, save them
            if len(frames) == sequence_length:
                X.append(frames)
                y.append(labels[i])
                frames.pop(0)  # Remove the oldest frame to slide the window
            
        cap.release()

    return np.array(X), np.array(y)

def prepare_data(data_dir, activity_labels, sequence_length=30):
    """
    Load video files from data directory, extract sequences and prepare labels.
    """
    video_paths, labels = [], []

    for label, activity in enumerate(activity_labels):
        activity_folder = os.path.join(data_dir, activity)
        for file_name in os.listdir(activity_folder):
            if file_name.endswith(('.mp4', '.avi')):
                video_paths.append(os.path.join(activity_folder, file_name))
                labels.append(label)

    # Load and process data from videos
    X, y = load_data_from_videos(video_paths, labels, sequence_length)

    # One-hot encode labels for training
    y = to_categorical(y, num_classes=len(activity_labels))
    return X, y

def split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)

