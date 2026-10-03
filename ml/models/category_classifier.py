"""
Automatic Category Classification Model for VisA Object Categories.
Predicts industrial product category from input RGB image using MobileNetV2 pretrained backbone.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, List, Dict, Optional

class CategoryClassifier:
    """
    MobileNetV2 Transfer Learning Network for 12-class industrial object classification.
    """
    def __init__(
        self,
        categories: List[str],
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        saved_models_dir: str = "ml/saved_models"
    ):
        self.categories = categories
        self.input_shape = input_shape
        self.num_classes = len(categories)
        self.saved_models_dir = os.path.abspath(saved_models_dir)
        self.model_path = os.path.join(self.saved_models_dir, "category_classifier.keras")
        self.model = self.build_model()

    def build_model(self) -> Model:
        """
        Builds MobileNetV2 pretrained backbone with frozen weights and 12-class classification head.
        """
        inputs = layers.Input(shape=self.input_shape, name="classifier_input")

        # Add Data Augmentation to prevent overfitting on tiny dataset
        x = layers.RandomFlip("horizontal_and_vertical")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)

        # MobileNetV2 expects input in [-1, 1] or [0, 1] with scaling. We pass [0, 1] scaled float32.
        # MobileNetV2 preprocess scaling converts [0, 255] to [-1, 1]
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        base_model.trainable = False  # Freeze MobileNetV2 backbone to prevent overfitting on tiny datasets

        features = base_model(x, training=False)
        features = layers.Dense(128, activation="relu", name="dense_features")(features)
        features = layers.Dropout(0.3, name="dropout_features")(features)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="category_probs")(features)

        model = Model(inputs, outputs, name="mobilenetv2_category_classifier")
        return model

    def load_weights_if_exists(self) -> bool:
        """
        Loads pre-trained weights if saved_models/category_classifier.keras exists.
        """
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                return True
            except Exception as e:
                print(f"[Warning] Could not load classifier model: {e}")
                return False
        return False

    def predict_category(self, img_tensor: tf.Tensor) -> Tuple[str, float]:
        """
        Predicts object category name and confidence score from image tensor.
        """
        if len(img_tensor.shape) == 3:
            img_tensor = tf.expand_dims(img_tensor, axis=0)

        probs = self.model(img_tensor, training=False).numpy()[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        predicted_category = self.categories[best_idx]

        return predicted_category, confidence
