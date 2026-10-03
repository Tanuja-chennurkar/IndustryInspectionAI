"""
Phase 3 Evaluation & Anomaly Localization Execution Script.
Evaluates Image AUROC, Pixel AUROC, IoU, Dice/F1, Precision, Recall, Confusion Matrix
across all 12 VisA categories and outputs structured report.
"""

import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset, VISA_CATEGORIES
from ml.inference.detector import AnomalyDetector
from ml.visualization.heatmap import HeatmapVisualizer
from ml.evaluation.metrics import AnomalyEvaluator

def evaluate_phase3(dataset_root="ml/data/VisA", output_dir="ml/data"):
    """
    Executes full evaluation pipeline for Phase 3.
    """
    print("=" * 100)
    print("   INDUSTRIAL VISUAL INSPECTION - PHASE 3 EVALUATION & ANOMALY LOCALIZATION   ")
    print("=" * 100)

    os.makedirs(output_dir, exist_ok=True)
    heatmaps_dir = os.path.join(output_dir, "sample_heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)

    dataset = VisADataset(dataset_root=dataset_root, batch_size=1)
    dataset.discover_and_index_files()

    detector = AnomalyDetector(saved_models_dir="ml/saved_models")
    visualizer = HeatmapVisualizer()
    evaluator = AnomalyEvaluator()

    per_category_results = {}
    all_y_true = []
    all_y_scores = []
    all_gt_masks = []
    all_anomaly_maps = []

    for cat in VISA_CATEGORIES:
        print(f"\n>>> Evaluating Category: [{cat}]")
        
        # Filter test set for category
        cat_df = dataset.metadata_df[
            (dataset.metadata_df["category"] == cat) & 
            (dataset.metadata_df["split"] == "test")
        ]

        if len(cat_df) == 0:
            print(f"  [Warning] No test samples found for category '{cat}'. Skipping.")
            continue

        cat_y_true = []
        cat_y_scores = []
        cat_gt_masks = []
        cat_anomaly_maps = []

        sample_saved = False

        for idx, row in cat_df.iterrows():
            img_path = row["image_path"]
            mask_path = row["mask_path"]
            gt_label = int(row["label"])

            # Run inference
            result = detector.inspect_image(image_input=img_path, category=cat)

            cat_y_true.append(gt_label)
            cat_y_scores.append(result["anomaly_score"])

            # Read ground truth mask if available
            if mask_path and os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is None:
                    gt_mask = np.zeros(detector.image_size, dtype=np.uint8)
                else:
                    gt_mask = cv2.resize(gt_mask, (detector.image_size[1], detector.image_size[0]), interpolation=cv2.INTER_NEAREST)
            else:
                gt_mask = np.zeros(detector.image_size, dtype=np.uint8)

            cat_gt_masks.append(gt_mask)
            cat_anomaly_maps.append(result["anomaly_map"])

            # Save sample visualization for defect image
            if gt_label == 1 and not sample_saved:
                comp_img = visualizer.create_multipanel_visualization(result)
                save_path = os.path.join(heatmaps_dir, f"{cat}_defect_sample.png")
                # RGB to BGR for cv2.imwrite
                cv2.imwrite(save_path, cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR))
                print(f"  - Saved sample heatmap visualization -> {save_path}")
                sample_saved = True

        # Calculate category image-level and pixel-level metrics
        img_metrics = evaluator.compute_image_metrics(cat_y_true, cat_y_scores, threshold=detector.category_manager.category_thresholds.get(cat, 0.35))
        pixel_metrics = evaluator.compute_pixel_metrics(cat_gt_masks, cat_anomaly_maps, threshold=detector.category_manager.category_thresholds.get(cat, 0.35))

        per_category_results[cat] = {
            "image_auroc": img_metrics["image_auroc"],
            "pixel_auroc": pixel_metrics["pixel_auroc"],
            "precision": img_metrics["precision"],
            "recall": img_metrics["recall"],
            "f1_score": img_metrics["f1_score"],
            "iou": pixel_metrics["iou"],
            "dice_f1": pixel_metrics["dice_f1"],
            "confusion_matrix": img_metrics["confusion_matrix"],
            "test_samples_count": len(cat_df)
        }

        all_y_true.extend(cat_y_true)
        all_y_scores.extend(cat_y_scores)
        all_gt_masks.extend(cat_gt_masks)
        all_anomaly_maps.extend(cat_anomaly_maps)

    # Compute Overall Macro-Metrics across all 12 categories
    overall_img_metrics = evaluator.compute_image_metrics(all_y_true, all_y_scores, threshold=0.33)
    overall_pixel_metrics = evaluator.compute_pixel_metrics(all_gt_masks, all_anomaly_maps, threshold=0.33)

    overall_results = {
        "image_auroc": overall_img_metrics["image_auroc"],
        "pixel_auroc": overall_pixel_metrics["pixel_auroc"],
        "precision": overall_img_metrics["precision"],
        "recall": overall_img_metrics["recall"],
        "f1_score": overall_img_metrics["f1_score"],
        "iou": overall_pixel_metrics["iou"],
        "dice_f1": overall_pixel_metrics["dice_f1"],
        "total_test_samples": len(all_y_true)
    }

    final_report = {
        "overall": overall_results,
        "per_category": per_category_results
    }

    # Save JSON Report
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(final_report, f, indent=2)

    # Save Markdown Report
    md_path = os.path.join(output_dir, "evaluation_results.md")
    md_lines = [
        "# VisA Dataset Phase 3 Model Evaluation & Localization Report",
        "",
        "## Overall Macro Metrics",
        f"- **Total Test Images Evaluated**: {overall_results['total_test_samples']}",
        f"- **Overall Image AUROC**: **{overall_results['image_auroc']:.4f}**",
        f"- **Overall Pixel AUROC**: **{overall_results['pixel_auroc']:.4f}**",
        f"- **Overall Precision**: {overall_results['precision']:.4f}",
        f"- **Overall Recall**: {overall_results['recall']:.4f}",
        f"- **Overall F1-Score**: {overall_results['f1_score']:.4f}",
        f"- **Overall IoU**: {overall_results['iou']:.4f}",
        f"- **Overall Dice / Pixel F1**: {overall_results['dice_f1']:.4f}",
        "",
        "## Per-Category Metrics Summary Table",
        "",
        "| Category | Test Samples | Image AUROC | Pixel AUROC | Precision | Recall | F1-Score | IoU | Dice / Pixel F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    ]

    for cat, res in per_category_results.items():
        md_lines.append(
            f"| `{cat}` | {res['test_samples_count']} | **{res['image_auroc']:.4f}** | **{res['pixel_auroc']:.4f}** | {res['precision']:.4f} | {res['recall']:.4f} | {res['f1_score']:.4f} | {res['iou']:.4f} | {res['dice_f1']:.4f} |"
        )

    md_lines.append(
        f"| **OVERALL** | **{overall_results['total_test_samples']}** | **{overall_results['image_auroc']:.4f}** | **{overall_results['pixel_auroc']:.4f}** | **{overall_results['precision']:.4f}** | **{overall_results['recall']:.4f}** | **{overall_results['f1_score']:.4f}** | **{overall_results['iou']:.4f}** | **{overall_results['dice_f1']:.4f}** |"
    )

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    # Print Console Summary Table
    print("\n" + "=" * 110)
    print("                       PHASE 3 EVALUATION METRICS SUMMARY TABLE                       ")
    print("=" * 110)
    print(f"{'Category':<14} | {'Image AUROC':<11} | {'Pixel AUROC':<11} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'IoU':<6} | {'Dice F1':<6}")
    print("-" * 110)
    for cat, res in per_category_results.items():
        print(
            f"{cat:<14} | "
            f"{res['image_auroc']:<11.4f} | "
            f"{res['pixel_auroc']:<11.4f} | "
            f"{res['precision']:<9.4f} | "
            f"{res['recall']:<8.4f} | "
            f"{res['f1_score']:<8.4f} | "
            f"{res['iou']:<6.4f} | "
            f"{res['dice_f1']:<6.4f}"
        )
    print("-" * 110)
    print(
        f"{'OVERALL MACRO':<14} | "
        f"{overall_results['image_auroc']:<11.4f} | "
        f"{overall_results['pixel_auroc']:<11.4f} | "
        f"{overall_results['precision']:<9.4f} | "
        f"{overall_results['recall']:<8.4f} | "
        f"{overall_results['f1_score']:<8.4f} | "
        f"{overall_results['iou']:<6.4f} | "
        f"{overall_results['dice_f1']:<6.4f}"
    )
    print("=" * 110)
    print(f"\n[SUCCESS] Reports and Heatmap Visualizations Generated:")
    print(f"  - JSON Report: {json_path}")
    print(f"  - Markdown Report: {md_path}")
    print(f"  - Heatmaps Directory: {heatmaps_dir}")

    return final_report

if __name__ == "__main__":
    evaluate_phase3()
