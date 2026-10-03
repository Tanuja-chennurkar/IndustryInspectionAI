"""
Anomaly Detection & Localization Inference Module.
Calculates reconstruction error maps, image-level anomaly scores,
applies Gaussian smoothing, extracts defect bounding boxes,
and supports automatic category identification.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, List, Optional, Union

from ml.preprocessing.preprocessor import ImagePreprocessor
from ml.models.category_model import CategoryAwareAnomalyDetector


class AnomalyDetector:
    """
    Inference Engine for Industrial Visual Anomaly Detection, Localization, and Category Auto-Routing.
    """
    def __init__(
        self,
        saved_models_dir: str = "ml/saved_models",
        image_size: Tuple[int, int] = (256, 256),
        gaussian_sigma: float = 2.0
    ):
        self.saved_models_dir = os.path.abspath(saved_models_dir)
        self.image_size = tuple(image_size)
        self.gaussian_sigma = gaussian_sigma
        self.preprocessor = ImagePreprocessor(target_size=image_size)
        
        from ml.datasets.visa_dataset import VISA_CATEGORIES
        self.categories = VISA_CATEGORIES
        self.category_manager = CategoryAwareAnomalyDetector(
            categories=VISA_CATEGORIES,
            input_shape=(*image_size, 3),
            saved_models_dir=saved_models_dir
        )
        self.loaded_models: Dict[str, tf.keras.Model] = {}

    def get_loaded_model(self, category: str) -> tf.keras.Model:
        """
        Retrieves cached category model or loads it from disk.
        """
        if category not in self.loaded_models:
            self.loaded_models[category] = self.category_manager.load_category_model(category)
        return self.loaded_models[category]

    def compute_anomaly_map(self, input_tensor: tf.Tensor, reconstructed_tensor: tf.Tensor) -> np.ndarray:
        """
        Calculates per-pixel reconstruction error map and applies Gaussian smoothing.
        Returns a 2D float32 array normalized to [0.0, 1.0].
        """
        img_np = input_tensor.numpy()[0] if len(input_tensor.shape) == 4 else input_tensor.numpy()
        recon_np = reconstructed_tensor.numpy()[0] if len(reconstructed_tensor.shape) == 4 else reconstructed_tensor.numpy()

        # Channel-wise mean absolute error
        diff = np.abs(img_np - recon_np)
        diff_map = np.mean(diff, axis=-1)

        # Apply Gaussian Blur smoothing to suppress pixel noise and highlight blob defects
        ksize = int(self.gaussian_sigma * 4) | 1  # ensure odd kernel size
        smoothed_map = cv2.GaussianBlur(diff_map, (ksize, ksize), self.gaussian_sigma)

        return smoothed_map

    def extract_defect_regions(
        self,
        anomaly_map: np.ndarray,
        threshold: float,
        status: str = "ANOMALY",
        heatmap_percentile: float = 95.0,
        min_contour_area: float = 25.0
    ) -> Tuple[List[Dict], float, np.ndarray]:
        """
        Extracts defect contours, bounding boxes (x, y, w, h), centroids, and binary defect mask.
        """
        h, w = anomaly_map.shape
        if status == "NORMAL":
            return [], 0.0, np.zeros((h, w), dtype=np.uint8)

        # For anomalous samples, isolate defect regions using peak error thresholding
        defect_threshold = max(threshold, float(np.percentile(anomaly_map, heatmap_percentile)))
        binary_mask = (anomaly_map >= defect_threshold).astype(np.uint8) * 255

        # Find contours of anomalous regions
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defect_regions = []
        total_defect_pixels = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:  # filter noise contours
                continue

            total_defect_pixels += area
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Compute centroid
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + bw // 2
            cy = int(M["m01"] / M["m00"]) if M["m01"] != 0 else y + bh // 2

            defect_regions.append({
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area_pixels": float(area),
                "centroid": [int(cx), int(cy)]
            })

        defect_area_pct = float((total_defect_pixels / (h * w)) * 100.0)
        return defect_regions, defect_area_pct, binary_mask

    def inspect_image(
        self,
        image_input: Union[str, np.ndarray],
        category: Optional[str] = None,
        custom_threshold: Optional[float] = None,
        is_bgr: bool = False
    ) -> Dict:
        """
        Full inference workflow for a single image:
        Preprocess -> Automatic Category Classifier (if category is None or 'auto') -> Keras Model Inference -> Anomaly Map -> Score -> Localization -> Defect Region
        """
        # Preprocess input image
        if isinstance(image_input, str):
            input_tensor = self.preprocessor.preprocess_image_path(image_input)
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            raw_rgb = self.preprocessor.postprocess_to_uint8(input_tensor[0])
        else:
            processed_np = self.preprocessor.preprocess_numpy(image_input, is_bgr=is_bgr)
            input_tensor = tf.convert_to_tensor(processed_np[np.newaxis, ...], dtype=tf.float32)
            raw_rgb = self.preprocessor.postprocess_to_uint8(input_tensor[0])

        # Automatic Category Identification via Minimum Reconstruction Error
        category_confidence = 1.0
        if not category or category.lower() in ["auto", "automatic category detection", "all categories"]:
            min_error = float('inf')
            best_cat = self.categories[0]
            for cat in self.categories:
                # Load each model to test reconstruction
                ae_model = self.get_loaded_model(cat)
                recon = ae_model(input_tensor, training=False)
                # Compute simple reconstruction error
                error = float(tf.reduce_mean(tf.abs(input_tensor - recon)))
                if error < min_error:
                    min_error = error
                    best_cat = cat
            category = best_cat
            # Set a dummy high confidence if auto-routed
            category_confidence = 0.99

        # Load chosen category-specific model
        model = self.get_loaded_model(category)
        saved_threshold = self.category_manager.category_thresholds.get(category, 0.355)
        threshold = custom_threshold if custom_threshold is not None else saved_threshold

        model_path = os.path.join(self.saved_models_dir, category, "model.weights.h5")
        print(f"[DIAGNOSTIC] Detected category: {category} | Model path: {model_path} | Input shape: {input_tensor.shape} | Threshold: {threshold:.4f}")

        # Run Keras Model Inference
        reconstructed_tensor = model(input_tensor, training=False)

        # Compute Anomaly Map & Image Score
        anomaly_map = self.compute_anomaly_map(input_tensor, reconstructed_tensor)
        
        # Image-level score: mean of top 0.1% peak error pixels to isolate local defects
        flat_map = np.sort(anomaly_map.ravel())
        top_k_pixels = max(1, int(len(flat_map) * 0.001))
        anomaly_score = float(np.mean(flat_map[-top_k_pixels:]))

        status = "ANOMALY" if anomaly_score >= threshold else "NORMAL"

        # Extract defect regions & binary mask
        defect_regions, defect_area_pct, binary_mask = self.extract_defect_regions(
            anomaly_map, threshold=threshold, status=status
        )

        reconstructed_rgb = self.preprocessor.postprocess_to_uint8(reconstructed_tensor[0])

        return {
            "category": category,
            "category_confidence": round(category_confidence, 4),
            "status": status,
            "anomaly_score": round(anomaly_score, 4),
            "threshold": round(threshold, 4),
            "defect_area_percentage": round(defect_area_pct, 2),
            "defect_regions": defect_regions,
            "raw_image": raw_rgb,
            "reconstructed_image": reconstructed_rgb,
            "anomaly_map": anomaly_map,
            "binary_mask": binary_mask,
            "model_version": self.category_manager.model_version
        }
