"""
Comprehensive Evaluation Metrics Module for Industrial Visual Inspection.
Computes Image-Level (AUROC, Precision, Recall, F1, Confusion Matrix)
and Pixel-Level Localization (Pixel AUROC, IoU, Dice/F1) metrics.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Union

class AnomalyEvaluator:
    """
    Evaluator class for image-level detection and pixel-level localization metrics.
    """

    @staticmethod
    def compute_image_metrics(y_true: List[int], y_scores: List[float], threshold: float = 0.5) -> Dict:
        """
        Computes Image-Level AUROC, Precision, Recall, F1, and Confusion Matrix.
        """
        y_true = np.array(y_true, dtype=int)
        y_scores = np.array(y_scores, dtype=float)
        y_pred = (y_scores >= threshold).astype(int)

        # Image AUROC
        try:
            if len(np.unique(y_true)) > 1:
                image_auroc = float(roc_auc_score(y_true, y_scores))
            else:
                image_auroc = 1.0 if np.all(y_true == y_pred) else 0.5
        except Exception:
            image_auroc = 0.5

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        # Confusion Matrix
        if len(np.unique(y_true)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        else:
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        return {
            "image_auroc": round(float(image_auroc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp)
            }
        }

    @staticmethod
    def compute_pixel_metrics(y_true_masks: List[np.ndarray], anomaly_maps: List[np.ndarray], threshold: float = 0.5) -> Dict:
        """
        Computes Pixel-Level AUROC, IoU (Intersection over Union), and Dice/F1 score across ground-truth binary masks.
        """
        if not y_true_masks or not anomaly_maps:
            return {"pixel_auroc": 0.0, "iou": 0.0, "dice_f1": 0.0}

        y_true_flat = []
        y_score_flat = []
        y_pred_flat = []

        for gt_mask, a_map in zip(y_true_masks, anomaly_maps):
            gt_bin = (gt_mask > 0).astype(int).ravel()
            a_map_norm = np.clip(a_map / max(1e-5, np.max(a_map)), 0, 1).ravel()
            pred_bin = (a_map_norm >= threshold).astype(int)

            y_true_flat.append(gt_bin)
            y_score_flat.append(a_map_norm)
            y_pred_flat.append(pred_bin)

        y_true_all = np.concatenate(y_true_flat)
        y_score_all = np.concatenate(y_score_flat)
        y_pred_all = np.concatenate(y_pred_flat)

        # Pixel AUROC
        try:
            if len(np.unique(y_true_all)) > 1:
                pixel_auroc = float(roc_auc_score(y_true_all, y_score_all))
            else:
                pixel_auroc = 1.0
        except Exception:
            pixel_auroc = 0.5

        # IoU and Dice Score
        intersection = np.sum((y_true_all == 1) & (y_pred_all == 1))
        union = np.sum((y_true_all == 1) | (y_pred_all == 1))
        iou = float(intersection / union) if union > 0 else (1.0 if np.all(y_true_all == y_pred_all) else 0.0)

        dice = float((2 * intersection) / (np.sum(y_true_all == 1) + np.sum(y_pred_all == 1))) if (np.sum(y_true_all == 1) + np.sum(y_pred_all == 1)) > 0 else 1.0

        return {
            "pixel_auroc": round(float(pixel_auroc), 4),
            "iou": round(float(iou), 4),
            "dice_f1": round(float(dice), 4)
        }
