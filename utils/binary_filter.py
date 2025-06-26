import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.abspath('..')))

# Load once
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "leaf_filter_model (1).keras")
leaf_filter = tf.keras.models.load_model(MODEL_PATH)

def binary_leaf_check(img_array: np.ndarray) -> bool:
    resized = tf.image.resize(img_array, (224, 224))
    resized = tf.expand_dims(resized, axis=0)
    prediction = leaf_filter.predict(resized)[0][0]
    return prediction >= 0.3  # You can adjust the threshold
