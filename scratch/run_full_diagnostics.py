import os
import sys
import glob
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset, VISA_CATEGORIES
from ml.preprocessing.preprocessor import ImagePreprocessor
from ml.models.category_classifier import CategoryClassifier
from ml.models.category_model import CategoryAwareAnomalyDetector
from ml.inference.detector import AnomalyDetector

def run_diagnostic_1_classifier_test():
    print("=" * 80)
    print("   DIAGNOSTIC 1: CATEGORY CLASSIFIER TEST   ")
    print("=" * 80)

    preprocessor = ImagePreprocessor(target_size=(256, 256))
    dataset = VisADataset()
    df = dataset.discover_and_index_files()

    classifier_path = os.path.abspath("ml/saved_models/category_classifier.keras")
    print(f"Loading classifier from: {classifier_path}")
    print(f"File exists: {os.path.exists(classifier_path)}")

    if not os.path.exists(classifier_path):
        print("[ERROR] Classifier model file does NOT exist!")
        return

    classifier_model = tf.keras.models.load_model(classifier_path)
    print(f"Loaded classifier input shape: {classifier_model.input_shape}")
    print(f"Loaded classifier output shape: {classifier_model.output_shape}")

    # Select representative sample batch from each category (5 images per category)
    samples_per_cat = 5
    records = []
    
    y_true = []
    y_pred = []
    confidences = []

    for cat in VISA_CATEGORIES:
        cat_df = df[df["category"] == cat].head(samples_per_cat)
        for _, row in cat_df.iterrows():
            img_path = row["image_path"]
            img_tensor = preprocessor.preprocess_image_path(img_path)
            img_tensor = tf.expand_dims(img_tensor, axis=0)

            probs = classifier_model(img_tensor, training=False).numpy()[0]
            top_1_idx = int(np.argmax(probs))
            top_1_conf = float(probs[top_1_idx])
            pred_cat = VISA_CATEGORIES[top_1_idx]

            # Top 3
            top_3_indices = np.argsort(probs)[::-1][:3]
            top_3_preds = [(VISA_CATEGORIES[idx], round(float(probs[idx]), 4)) for idx in top_3_indices]

            y_true.append(cat)
            y_pred.append(pred_cat)
            confidences.append(top_1_conf)

            records.append({
                "actual": cat,
                "predicted": pred_cat,
                "confidence": round(top_1_conf, 4),
                "top_3": top_3_preds
            })

    # Metrics calculation
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    overall_acc = (correct / total) * 100.0 if total > 0 else 0.0
    avg_conf = (sum(confidences) / total) * 100.0 if total > 0 else 0.0

    print(f"\nOverall Classifier Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    print(f"Average Top-1 Confidence: {avg_conf:.2f}%")

    # Prediction Distribution
    pred_counts = pd.Series(y_pred).value_counts().to_dict()
    print(f"Prediction Distribution across {total} samples:")
    for cat in VISA_CATEGORIES:
        print(f"  - {cat:<12}: {pred_counts.get(cat, 0)} predictions")

    # Per-Category Accuracy
    print("\nPer-Category Accuracy:")
    per_cat_acc = {}
    for cat in VISA_CATEGORIES:
        cat_total = sum(1 for t in y_true if t == cat)
        cat_correct = sum(1 for t, p in zip(y_true, y_pred) if t == cat and p == cat)
        acc = (cat_correct / cat_total) * 100.0 if cat_total > 0 else 0.0
        per_cat_acc[cat] = acc
        print(f"  - {cat:<12}: {acc:.2f}% ({cat_correct}/{cat_total})")

    # 12x12 Confusion Matrix
    print("\n12x12 Confusion Matrix (Rows: Actual, Cols: Predicted):")
    cm_df = pd.DataFrame(0, index=VISA_CATEGORIES, columns=VISA_CATEGORIES)
    for t, p in zip(y_true, y_pred):
        cm_df.loc[t, p] += 1
    print(cm_df.to_string())

def run_diagnostic_2_verify_loading():
    print("\n" + "=" * 80)
    print("   DIAGNOSTIC 2: VERIFY CLASSIFIER MODEL LOADING & METADATA   ")
    print("=" * 80)
    classifier_path = os.path.abspath("ml/saved_models/category_classifier.keras")
    print(f"Classifier model path: {classifier_path}")
    print(f"File exists: {os.path.exists(classifier_path)}")
    if os.path.exists(classifier_path):
        mtime = os.path.getmtime(classifier_path)
        import datetime
        dt_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        size_kb = os.path.getsize(classifier_path) / 1024.0
        print(f"File modified timestamp: {dt_str}")
        print(f"File size: {size_kb:.2f} KB")

        model = tf.keras.models.load_model(classifier_path)
        print(f"Model Input Shape: {model.input_shape}")
        print(f"Model Output Shape: {model.output_shape}")
        print(f"Number of Output Classes: {model.output_shape[-1]}")
        print(f"Class-to-Index Mapping: {dict(enumerate(VISA_CATEGORIES))}")

def run_diagnostic_3_manual_routing():
    print("\n" + "=" * 80)
    print("   DIAGNOSTIC 3: MANUAL ROUTING TEST   ")
    print("=" * 80)
    
    detector = AnomalyDetector(saved_models_dir="ml/saved_models")
    preprocessor = ImagePreprocessor(target_size=(256, 256))

    test_cases = [
        ("pcb1", "test_images/capsules_normal.jpg"),  # fallback if pcb1 image not in test_images
        ("cashew", "test_images/capsules_normal.jpg"),
        ("candle", "test_images/capsules_normal.jpg"),
    ]

    # Find actual dataset normal images for pcb1, cashew, candle
    dataset = VisADataset()
    df = dataset.discover_and_index_files()

    for cat in ["pcb1", "cashew", "candle"]:
        cat_normal = df[(df["category"] == cat) & (df["label"] == 0)]
        if not cat_normal.empty:
            img_path = cat_normal.iloc[0]["image_path"]
        else:
            img_path = os.path.abspath("test_images/capsules_normal.jpg")

        model_path = os.path.abspath(f"ml/saved_models/{cat}/model.weights.h5")
        input_tensor = preprocessor.preprocess_image_path(img_path)
        input_tensor_batched = tf.expand_dims(input_tensor, axis=0)

        model = detector.get_loaded_model(cat)
        recon_tensor = model(input_tensor_batched, training=False)

        input_np = input_tensor.numpy()
        recon_np = recon_tensor.numpy()[0]

        in_min, in_max, in_mean = float(np.min(input_np)), float(np.max(input_np)), float(np.mean(input_np))
        rec_min, rec_max, rec_mean = float(np.min(recon_np)), float(np.max(recon_np)), float(np.mean(recon_np))

        res = detector.inspect_image(image_input=img_path, category=cat)

        print(f"\nCategory: [{cat}]")
        print(f"  - Image Path: {img_path}")
        print(f"  - Model Path: {model_path}")
        print(f"  - Input Tensor Shape: {input_tensor_batched.shape}")
        print(f"  - Input Min/Max/Mean: [{in_min:.4f}, {in_max:.4f}, {in_mean:.4f}]")
        print(f"  - Reconstruction Min/Max/Mean: [{rec_min:.4f}, {rec_max:.4f}, {rec_mean:.4f}]")
        print(f"  - Anomaly Score: {res['anomaly_score']:.4f}")
        print(f"  - Threshold: {res['threshold']:.4f}")
        print(f"  - Final Status: {res['status']}")

def run_diagnostic_4_reconstruction_test():
    print("\n" + "=" * 80)
    print("   DIAGNOSTIC 4: RECONSTRUCTION TEST (TENSOR STATS & SAVED IMAGES)   ")
    print("=" * 80)
    
    detector = AnomalyDetector(saved_models_dir="ml/saved_models")
    preprocessor = ImagePreprocessor(target_size=(256, 256))

    img_path = os.path.abspath("test_images/capsules_normal.jpg")
    input_tensor = preprocessor.preprocess_image_path(img_path)
    input_tensor_batched = tf.expand_dims(input_tensor, axis=0)

    model = detector.get_loaded_model("capsules")
    recon_tensor = model(input_tensor_batched, training=False)

    img_np = input_tensor_batched.numpy()[0]
    recon_np = recon_tensor.numpy()[0]
    diff_np = np.abs(img_np - recon_np)

    in_min, in_max, in_mean = float(np.min(img_np)), float(np.max(img_np)), float(np.mean(img_np))
    rec_min, rec_max, rec_mean = float(np.min(recon_np)), float(np.max(recon_np)), float(np.mean(recon_np))
    err_min, err_max, err_mean = float(np.min(diff_np)), float(np.max(diff_np)), float(np.mean(diff_np))

    print(f"Input Tensor        : min={in_min:.4f}, max={in_max:.4f}, mean={in_mean:.4f}")
    print(f"Reconstruction Tensor: min={rec_min:.4f}, max={rec_max:.4f}, mean={rec_mean:.4f}")
    print(f"Error Map Tensor     : min={err_min:.4f}, max={err_max:.4f}, mean={err_mean:.4f}")

    os.makedirs("scratch/reconstruction_debug", exist_ok=True)
    input_uint8 = preprocessor.postprocess_to_uint8(img_np)
    recon_uint8 = preprocessor.postprocess_to_uint8(recon_np)
    err_uint8 = cv2.applyColorMap((np.clip(np.mean(diff_np, axis=-1) / max(1e-5, err_max), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    cv2.imwrite("scratch/reconstruction_debug/input.png", cv2.cvtColor(input_uint8, cv2.COLOR_RGB2BGR))
    cv2.imwrite("scratch/reconstruction_debug/reconstruction.png", cv2.cvtColor(recon_uint8, cv2.COLOR_RGB2BGR))
    cv2.imwrite("scratch/reconstruction_debug/error_map.png", err_uint8)

    print("\nSaved images:")
    print("  - scratch/reconstruction_debug/input.png")
    print("  - scratch/reconstruction_debug/reconstruction.png")
    print("  - scratch/reconstruction_debug/error_map.png")

    if rec_max - rec_min < 0.05:
        print("\n[DIAGNOSTIC CONCLUSION B]: TensorFlow model output is nearly constant/gray!")
    else:
        print("\n[DIAGNOSTIC CONCLUSION A]: TensorFlow model output is dynamic with rich spatial details! Any gray issue is frontend/visualization display.")

def run_diagnostic_7_threshold_test():
    print("\n" + "=" * 80)
    print("   DIAGNOSTIC 7: THRESHOLD TEST ACROSS ALL 12 CATEGORIES   ")
    print("=" * 80)

    category_manager = CategoryAwareAnomalyDetector(categories=VISA_CATEGORIES)
    print(f"{'Category':<15} | {'Mean Loss':<12} | {'Std Loss':<12} | {'Threshold':<15} | {'Val Samples'}")
    print("-" * 75)

    for cat in VISA_CATEGORIES:
        meta_path = os.path.abspath(f"ml/saved_models/{cat}/metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            t = meta.get("threshold", 0.0)
            tc = meta.get("training_config", {})
            mean_l = tc.get("normal_mean_loss", 0.0)
            std_l = tc.get("normal_std_loss", 0.0)
            val_cnt = tc.get("batch_size", 0)
            print(f"{cat:<15} | {mean_l:<12.5f} | {std_l:<12.5f} | {t:<15.5f} | {val_cnt}")
        else:
            print(f"{cat:<15} | METADATA NOT FOUND")

if __name__ == "__main__":
    run_diagnostic_1_classifier_test()
    run_diagnostic_2_verify_loading()
    run_diagnostic_3_manual_routing()
    run_diagnostic_4_reconstruction_test()
    run_diagnostic_7_threshold_test()
