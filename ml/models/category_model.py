"""
Category-Aware Normality Model Manager for VisA Industrial Categories.
Handles category-specific autoencoders, model saving/loading, versioning,
and anomaly threshold tracking per category.
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple, List, Union

from ml.models.autoencoder import build_conv_autoencoder, ConvAutoencoder
from ml.models.losses import SSIML1Loss

class CategoryAwareAnomalyDetector:
    """
    Manager for category-aware self-supervised anomaly detection models.
    Supports individual category autoencoders or shared feature representation with category heads.
    """
    def __init__(
        self,
        categories: List[str],
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        saved_models_dir: str = "ml/saved_models"
    ):
        self.categories = categories
        self.input_shape = input_shape
        self.saved_models_dir = os.path.abspath(saved_models_dir)
        os.makedirs(self.saved_models_dir, exist_ok=True)
        
        self.category_models: Dict[str, ConvAutoencoder] = {}
        self.category_thresholds: Dict[str, float] = {cat: 0.5 for cat in categories}
        self.model_version: str = "v1.0.0"

    def get_or_create_model(self, category: str) -> ConvAutoencoder:
        """
        Retrieves existing category model or instantiates a new ConvAutoencoder.
        """
        if category not in self.category_models:
            self.category_models[category] = build_conv_autoencoder(input_shape=self.input_shape)
        return self.category_models[category]

    def save_category_model(
        self,
        category: str,
        threshold: float = 0.5,
        version: str = "v1.0.0",
        training_config: Optional[Dict] = None
    ) -> str:
        """
        Saves category model weights, architecture, and threshold metadata.
        """
        cat_dir = os.path.join(self.saved_models_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        model = self.get_or_create_model(category)
        weights_path = os.path.join(cat_dir, "model.weights.h5")
        model.save_weights(weights_path)

        metadata = {
            "category": category,
            "version": version,
            "threshold": float(threshold),
            "input_shape": list(self.input_shape),
            "weights_file": "model.weights.h5",
            "training_config": training_config or {}
        }
        meta_path = os.path.join(cat_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.category_thresholds[category] = threshold
        print(f"[Model Saved] Category '{category}' -> {cat_dir}")
        return cat_dir

    def load_category_model(self, category: str) -> ConvAutoencoder:
        """
        Loads trained model weights and threshold metadata for a specific category.
        """
        cat_dir = os.path.join(self.saved_models_dir, category)
        weights_path = os.path.join(cat_dir, "model.weights.h5")
        meta_path = os.path.join(cat_dir, "metadata.json")

        model = self.get_or_create_model(category)

        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"[Model Loaded] Category '{category}' weights restored from {weights_path}")

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.category_thresholds[category] = float(meta.get("threshold", 0.5))

        return model
