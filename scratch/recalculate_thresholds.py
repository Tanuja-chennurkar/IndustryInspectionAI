import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset, VISA_CATEGORIES
from ml.inference.detector import AnomalyDetector

def recalculate_thresholds(dataset_root="ml/data/VisA", saved_models_dir="ml/saved_models"):
    dataset = VisADataset(dataset_root=dataset_root, batch_size=16)
    detector = AnomalyDetector(saved_models_dir=saved_models_dir)
    
    for cat in VISA_CATEGORIES:
        print(f"\nRecalculating threshold for category: {cat}")
        model = detector.get_loaded_model(cat)
        
        train_ds = dataset.create_tf_dataset(category=cat, split="train", is_training=False)
        
        normal_peak_scores = []
        for batch in train_ds:
            recon_batch = model(batch, training=False)
            
            for i in range(batch.shape[0]):
                input_tensor = tf.expand_dims(batch[i], axis=0)
                recon_tensor = tf.expand_dims(recon_batch[i], axis=0)
                
                anomaly_map = detector.compute_anomaly_map(input_tensor, recon_tensor)
                
                flat_map = np.sort(anomaly_map.ravel())
                top_k_pixels = max(1, int(len(flat_map) * 0.001))
                peak_score = float(np.mean(flat_map[-top_k_pixels:]))
                normal_peak_scores.append(peak_score)
                
            mean_score = float(np.mean(normal_peak_scores))
            max_score = float(np.max(normal_peak_scores))
            threshold = max_score + 0.001
        
            print(f"[{cat}] Mean Peak Score: {mean_score:.4f}, Max: {max_score:.4f}")
            print(f"[{cat}] Normal-calibrated threshold: {threshold:.4f}")
        
        # Update metadata.json
        meta_path = os.path.join(saved_models_dir, cat, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            
            metadata["previous_threshold"] = metadata.get("threshold")
            metadata["threshold"] = threshold
            metadata["threshold_calibration"] = {
                "score": "top_0.1_percent_peak_error",
                "normal_train_mean": mean_score,
                "normal_train_max": max_score,
                "margin": 0.001
            }
            
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[{cat}] Updated metadata.json")

if __name__ == "__main__":
    recalculate_thresholds()
