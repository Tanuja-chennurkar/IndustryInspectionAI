import os
import sys
import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.preprocessing.preprocessor import ImagePreprocessor

def test_preprocessing_consistency():
    img_path = os.path.abspath("test_images/capsules_normal.jpg")
    preprocessor = ImagePreprocessor(target_size=(256, 256))

    # 1. TF Path Preprocessing
    tf_tensor = preprocessor.preprocess_image_path(img_path).numpy()

    # 2. OpenCV Numpy Preprocessing
    img_bgr = cv2.imread(img_path)
    np_tensor = preprocessor.preprocess_numpy(img_bgr, is_bgr=True)

    # Calculate absolute difference
    diff = np.abs(tf_tensor - np_tensor)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max pixel diff between TF and OpenCV preprocessing: {max_diff:.6f}")
    print(f"Mean pixel diff between TF and OpenCV preprocessing: {mean_diff:.6f}")
    assert max_diff < 0.05, f"Preprocessing mismatch! Max diff: {max_diff}"
    print("[SUCCESS] Preprocessing pipelines are consistent!")

if __name__ == "__main__":
    test_preprocessing_consistency()
