# utils/model.py
from tensorflow.keras.models import load_model as tf_load_model

def load_model(model_path):
    # Ensure that the model path is correct and that the file exists
    try:
        return tf_load_model(model_path)
    except OSError as e:
        print(f"Error loading model: {e}")
        return None
