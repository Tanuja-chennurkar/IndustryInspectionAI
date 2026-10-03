import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset, VISA_CATEGORIES
from ml.preprocessing.preprocessor import ImagePreprocessor
from ml.models.category_classifier import CategoryClassifier

def evaluate_classifier_independently():
    print("=" * 80)
    print("   PART 6: INDEPENDENT MOBILENETV2 CLASSIFIER EVALUATION   ")
    print("=" * 80)

    classifier = CategoryClassifier(categories=VISA_CATEGORIES)
    loaded = classifier.load_weights_if_exists()
    assert loaded, "Failed to load MobileNetV2 classifier weights!"

    preprocessor = ImagePreprocessor(target_size=(256, 256))
    dataset = VisADataset(seed=42)
    df = dataset.discover_and_index_files()

    # Evaluate on test/val set across all 12 categories (10 samples per category = 120 samples)
    eval_df = df.groupby("category").apply(lambda g: g.head(10)).reset_index(drop=True)

    y_true = []
    y_pred = []
    top1_confs = []
    top3_correct = 0

    category_to_idx = {cat: idx for idx, cat in enumerate(VISA_CATEGORIES)}

    for _, row in eval_df.iterrows():
        cat = row["category"]
        img_path = row["image_path"]

        img_tensor = preprocessor.preprocess_image_path(img_path)
        img_batched = tf.expand_dims(img_tensor, axis=0)

        probs = classifier.model(img_batched, training=False).numpy()[0]
        top1_idx = int(np.argmax(probs))
        top1_conf = float(probs[top1_idx])
        pred_cat = VISA_CATEGORIES[top1_idx]

        top3_indices = np.argsort(probs)[::-1][:3]
        true_idx = category_to_idx[cat]
        if true_idx in top3_indices:
            top3_correct += 1

        y_true.append(cat)
        y_pred.append(pred_cat)
        top1_confs.append(top1_conf)

    total_samples = len(y_true)
    correct_cnt = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    overall_acc = (correct_cnt / total_samples) * 100.0
    top3_acc = (top3_correct / total_samples) * 100.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    avg_conf = float(np.mean(top1_confs)) * 100.0

    print(f"\nOverall Top-1 Accuracy: {overall_acc:.2f}% ({correct_cnt}/{total_samples})")
    print(f"Top-3 Accuracy        : {top3_acc:.2f}% ({top3_correct}/{total_samples})")
    print(f"Macro F1-Score        : {macro_f1:.4f}")
    print(f"Average Top-1 Conf    : {avg_conf:.2f}%")

    # Prediction Distribution
    pred_counts = pd.Series(y_pred).value_counts().to_dict()
    print("\nPrediction Distribution across classes:")
    for cat in VISA_CATEGORIES:
        print(f"  - {cat:<12}: {pred_counts.get(cat, 0)} predictions")

    # Per-Category Accuracy
    print("\nPer-Category Accuracy:")
    for cat in VISA_CATEGORIES:
        cat_total = sum(1 for t in y_true if t == cat)
        cat_corr = sum(1 for t, p in zip(y_true, y_pred) if t == cat and p == cat)
        acc = (cat_corr / cat_total) * 100.0 if cat_total > 0 else 0.0
        print(f"  - {cat:<12}: {acc:.2f}% ({cat_corr}/{cat_total})")

    # 12x12 Confusion Matrix
    print("\n12x12 Confusion Matrix (Rows: Actual, Cols: Predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=VISA_CATEGORIES)
    cm_df = pd.DataFrame(cm, index=VISA_CATEGORIES, columns=VISA_CATEGORIES)
    print(cm_df.to_string())

if __name__ == "__main__":
    evaluate_classifier_independently()
