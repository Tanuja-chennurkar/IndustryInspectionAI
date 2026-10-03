"""
VisA Dataset Inspection and Verification Script.
Validates categories, image dimensions, ground-truth masks, split distributions,
and tests TensorFlow tf.data.Dataset integration.
"""

import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
from typing import Dict

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset, VISA_CATEGORIES
from ml.preprocessing.preprocessor import ImagePreprocessor

def run_dataset_inspection(dataset_root: str = "ml/data/VisA") -> Dict:
    """
    Performs full dataset inspection across all 12 categories.
    """
    print("=" * 80)
    print("      INDUSTRIAL VISUAL INSPECTION - VISA DATASET INSPECTION REPORT      ")
    print("=" * 80)

    dataset = VisADataset(dataset_root=dataset_root)
    df = dataset.discover_and_index_files()

    print(f"\n[1] VisA Dataset Location: {os.path.abspath(dataset_root)}")
    print(f"[2] Total Categories Found: {len(dataset.categories)}")
    print(f"[3] Total Indexed Records: {len(df)}")

    # Category breakdown table
    stats = dataset.get_summary_statistics()
    
    print("\n" + "-" * 80)
    print(f"{'Category':<15} | {'Normal':<8} | {'Anomaly':<8} | {'Train (Norm)':<12} | {'Test (N+A)':<10} | {'Total':<6}")
    print("-" * 80)

    for cat, cat_stat in stats["categories"].items():
        print(
            f"{cat:<15} | "
            f"{cat_stat['normal']:<8} | "
            f"{cat_stat['anomaly']:<8} | "
            f"{cat_stat['train_samples']:<12} | "
            f"{cat_stat['test_samples']:<10} | "
            f"{cat_stat['total']:<6}"
        )
    print("-" * 80)
    print(
        f"{'OVERALL TOTAL':<15} | "
        f"{stats['total_normal']:<8} | "
        f"{stats['total_anomaly']:<8} | "
        f"{sum(c['train_samples'] for c in stats['categories'].values()):<12} | "
        f"{sum(c['test_samples'] for c in stats['categories'].values()):<10} | "
        f"{stats['total_images']:<6}"
    )
    print("-" * 80)

    # Verification of images and ground truth masks
    print("\n[4] Verifying Image and Mask Integrity...")
    sample_row = df[df["label"] == 1].iloc[0] if not df[df["label"] == 1].empty else df.iloc[0]
    
    img_path = sample_row["image_path"]
    mask_path = sample_row["mask_path"]

    img_raw = cv2.imread(img_path)
    print(f"  - Sample Image: {os.path.basename(img_path)} ({sample_row['category']})")
    print(f"  - Raw Shape: {img_raw.shape if img_raw is not None else 'N/A'}")
    print(f"  - Raw Pixel Range: min={img_raw.min()}, max={img_raw.max()}")

    if mask_path and os.path.exists(mask_path):
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        print(f"  - Sample Mask: {os.path.basename(mask_path)}")
        print(f"  - Mask Shape: {mask_raw.shape if mask_raw is not None else 'N/A'}")
        print(f"  - Unique Mask Values: {np.unique(mask_raw)}")
    else:
        print("  - Sample Mask: Normal sample (No defect mask required)")

    # TensorFlow Pipeline Verification
    print("\n[5] Testing TensorFlow tf.data.Dataset Pipeline...")
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    
    # Test Train Pipeline (Normal images only)
    train_ds = dataset.create_tf_dataset(category="all", split="train", is_training=True)
    train_batch = next(iter(train_ds))
    print(f"  - Training Batch Tensor Shape: {train_batch.shape} (dtype: {train_batch.dtype})")
    print(f"  - Preprocessed Pixel Range: min={tf.reduce_min(train_batch):.3f}, max={tf.reduce_max(train_batch):.3f}")

    # Test Test Pipeline (Image, Mask, Label)
    test_ds = dataset.create_tf_dataset(category="all", split="test", is_training=False)
    test_imgs, test_masks, test_lbls = next(iter(test_ds))
    print(f"  - Test Batch Image Shape: {test_imgs.shape}")
    print(f"  - Test Batch Mask Shape:  {test_masks.shape} (Unique values: {np.unique(test_masks.numpy())})")
    print(f"  - Test Batch Labels Shape: {test_lbls.shape} (Values: {test_lbls.numpy()[:8]})")

    print("\n[SUCCESS] VisA Dataset inspection and TensorFlow pipeline verification completed successfully!")
    print("=" * 80)

    return stats

if __name__ == "__main__":
    from ml.datasets.prepare_dataset import ensure_visa_structure
    ensure_visa_structure("ml/data/VisA")
    run_dataset_inspection("ml/data/VisA")
