"""
Custom Loss Functions for Structural Anomaly Detection.
Combines Perceptual Structural Similarity (SSIM) and L1/L2 Reconstruction Error.
"""

import tensorflow as tf
from typing import Tuple

class SSIML1Loss(tf.keras.losses.Loss):
    """
    Combined SSIM + L1 loss for self-supervised visual reconstruction.
    SSIM measures structural, luminance, and contrast changes (edges/scratches),
    while L1 loss penalizes absolute pixel intensity deviations.
    """
    def __init__(self, alpha: float = 0.8, name: str = "ssim_l1_loss"):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # SSIM calculation (expects values in range [0, 1])
        # max_val is 1.0 since images are normalized to [0.0, 1.0]
        ssim_val = tf.image.ssim(y_true, y_pred, max_val=1.0)
        ssim_loss = 1.0 - tf.reduce_mean(ssim_val)

        # L1 Loss
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

        # Combined weighted loss
        total_loss = (self.alpha * ssim_loss) + ((1.0 - self.alpha) * l1_loss)
        return total_loss

def get_anomaly_loss(alpha: float = 0.8):
    """
    Factory helper to return SSIML1Loss instance.
    """
    return SSIML1Loss(alpha=alpha)
