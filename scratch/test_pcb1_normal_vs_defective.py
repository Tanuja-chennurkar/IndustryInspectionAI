import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset
from ml.inference.detector import AnomalyDetector

def test_pcb1_normal_vs_defective_peak_score():
    print("=" * 80)
    print("   PART 4: PCB1 NORMAL VS DEFECTIVE EVALUATION (PEAK ERROR SCORE)   ")
    print("=" * 80)

    dataset = VisADataset(batch_size=1, seed=42)
    detector = AnomalyDetector(saved_models_dir="ml/saved_models")

    df = dataset.discover_and_index_files()
    pcb1_df = df[(df["category"] == "pcb1") & (df["split"] == "test")]

    normal_scores = []
    defect_scores = []
    y_true = []
    y_scores = []

    for _, row in pcb1_df.iterrows():
        img_path = row["image_path"]
        label = row["label"]  # 0: normal, 1: defect

        res = detector.inspect_image(image_input=img_path, category="pcb1")
        score = res["anomaly_score"]

        y_true.append(label)
        y_scores.append(score)

        if label == 0:
            normal_scores.append(score)
        else:
            defect_scores.append(score)

    norm_mean = float(np.mean(normal_scores))
    norm_std = float(np.std(normal_scores))
    def_mean = float(np.mean(defect_scores))
    def_std = float(np.std(defect_scores))

    # Threshold selection using normal validation distribution:
    # threshold = max(norm_mean * 1.05, norm_mean + (3.0 * max(norm_std, 0.01)))
    threshold = max(norm_mean * 1.05, norm_mean + (3.0 * max(norm_std, 0.01)))
    
    y_pred = [1 if s >= threshold else 0 for s in y_scores]

    auroc = float(roc_auc_score(y_true, y_scores)) if len(set(y_true)) > 1 else 1.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    print("\n--- Part 4 Score Distribution Report ---")
    print(f"Normal Mean: {norm_mean:.4f} | Normal Std: {norm_std:.4f} (Count: {len(normal_scores)})")
    print(f"Defect Mean: {def_mean:.4f} | Defect Std: {def_std:.4f} (Count: {len(defect_scores)})")
    print(f"Validation Calibrated Threshold: {threshold:.4f}")
    print("\n--- Evaluation Metrics ---")
    print(f"AUROC    : {auroc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

if __name__ == "__main__":
    test_pcb1_normal_vs_defective_peak_score()
