import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset, VISA_CATEGORIES
from ml.preprocessing.preprocessor import ImagePreprocessor
from ml.models.category_model import CategoryAwareAnomalyDetector

def evaluate_min_recon_auto_routing():
    print("=" * 80)
    print("   EVALUATING MINIMUM RECONSTRUCTION LOSS AUTO-ROUTING   ")
    print("=" * 80)

    category_manager = CategoryAwareAnomalyDetector(categories=VISA_CATEGORIES)
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    dataset = VisADataset(seed=42)
    df = dataset.discover_and_index_files()

    # Preload models
    models = {}
    for cat in VISA_CATEGORIES:
        models[cat] = category_manager.load_category_model(cat)

    eval_df = df.groupby("category").apply(lambda g: g.head(10)).reset_index(drop=True)

    y_true = []
    y_pred = []

    for _, row in eval_df.iterrows():
        true_cat = row["category"]
        img_path = row["image_path"]

        input_tensor = preprocessor.preprocess_image_path(img_path)
        input_batched = tf.expand_dims(input_tensor, axis=0)

        best_cat = None
        lowest_loss = float("inf")

        for cat_cand in VISA_CATEGORIES:
            m = models[cat_cand]
            recon = m(input_batched, training=False)
            loss = float(tf.reduce_mean(tf.abs(input_batched - recon)))
            if loss < lowest_loss:
                lowest_loss = loss
                best_cat = cat_cand

        y_true.append(true_cat)
        y_pred.append(best_cat)

    total_samples = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = (correct / total_samples) * 100.0

    print(f"\nMinimum Reconstruction Loss Auto-Routing Accuracy: {acc:.2f}% ({correct}/{total_samples})")
    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=VISA_CATEGORIES)
    print(pd.DataFrame(cm, index=VISA_CATEGORIES, columns=VISA_CATEGORIES).to_string())

if __name__ == "__main__":
    evaluate_min_recon_auto_routing()
