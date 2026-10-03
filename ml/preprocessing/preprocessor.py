"""
Image Preprocessing Module for VisA Anomaly Detection.
Provides configurable image loading, resizing, normalization, and tf.data preprocessing.
"""

import os
import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple, Union, Optional

class ImagePreprocessor:
    """
    Configurable preprocessor for industrial inspection images.
    Supports both TensorFlow tensors and OpenCV/NumPy arrays.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        color_channels: int = 3,
        normalize_range: Tuple[float, float] = (0.0, 1.0)
    ):
        self.target_size = tuple(target_size)
        self.color_channels = color_channels
        self.normalize_range = normalize_range

    def preprocess_image_path(self, image_path: tf.Tensor) -> tf.Tensor:
        """
        Reads and preprocesses an image file path using TensorFlow operations.
        Returns a float32 tensor of shape (H, W, C) normalized to target range.
        """
        image_bytes = tf.io.read_file(image_path)
        if self.color_channels == 1:
            image = tf.image.decode_image(image_bytes, channels=1, expand_animations=False)
        else:
            image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)

        image = tf.image.resize(image, self.target_size, method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(image, tf.float32)

        # Scaling [0, 255] to [0, 1]
        image = image / 255.0
        if self.normalize_range == (-1.0, 1.0):
            image = (image * 2.0) - 1.0

        return image

    def preprocess_mask_path_py(self, mask_path_str: bytes) -> np.ndarray:
        """
        Python wrapper for reading mask files safely.
        """
        path_str = mask_path_str.decode('utf-8') if isinstance(mask_path_str, bytes) else str(mask_path_str)
        if not path_str or not os.path.exists(path_str):
            return np.zeros((*self.target_size, 1), dtype=np.float32)

        mask = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((*self.target_size, 1), dtype=np.float32)

        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)[:, :, np.newaxis]
        return mask

    def preprocess_mask_path(self, mask_path: tf.Tensor) -> tf.Tensor:
        """
        Reads and preprocesses a ground truth binary mask path tensor.
        Returns a float32 tensor of shape (H, W, 1) with values in {0.0, 1.0}.
        """
        mask_tensor = tf.py_function(
            func=self.preprocess_mask_path_py,
            inp=[mask_path],
            Tout=tf.float32
        )
        mask_tensor.set_shape((*self.target_size, 1))
        return mask_tensor

    def preprocess_numpy(self, img_np: np.ndarray, is_bgr: bool = True) -> np.ndarray:
        """
        Preprocesses a raw OpenCV BGR or RGB NumPy array.
        Returns a float32 array scaled and resized to target size.
        """
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
        elif img_np.shape[2] == 3 and is_bgr:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img_np, (self.target_size[1], self.target_size[0]))
        img_float = img_resized.astype(np.float32) / 255.0

        if self.normalize_range == (-1.0, 1.0):
            img_float = (img_float * 2.0) - 1.0

        return img_float


    def postprocess_to_uint8(self, img_tensor: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
        """
        Converts normalized float image tensor/array back to uint8 [0, 255] RGB.
        """
        if isinstance(img_tensor, tf.Tensor):
            img_np = img_tensor.numpy()
        else:
            img_np = img_tensor.copy()

        if self.normalize_range == (-1.0, 1.0):
            img_np = (img_np + 1.0) / 2.0

        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        return img_np
