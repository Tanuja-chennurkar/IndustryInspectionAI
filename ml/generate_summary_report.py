"""
Dataset Summary Report Generator for VisA Industrial Visual Inspection Dataset.
Generates structured JSON and Markdown summary reports for Phase 1.
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset, VISA_CATEGORIES

def generate_reports(dataset_root: str = "ml/data/VisA", output_dir: str = "ml/data"):
    """
    Generates dataset_summary_report.json and dataset_summary_report.md.
    """
    dataset = VisADataset(dataset_root=dataset_root)
    df = dataset.discover_and_index_files()
    stats = dataset.get_summary_statistics()

    # Save JSON Report
    json_path = os.path.join(output_dir, "dataset_summary_report.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Save Markdown Report
    md_path = os.path.join(output_dir, "dataset_summary_report.md")
    
    md_lines = [
        "# VisA Dataset Summary & Preprocessing Specification Report",
        "",
        "## Executive Summary",
        "The **Visual Anomaly (VisA)** dataset is an official benchmark dataset for industrial visual inspection and anomaly detection.",
        "It features high-resolution images covering **12 object categories** spanning single-instance objects, multi-instance objects, and complex printed circuit boards (PCBs).",
        "",
        "## Key Dataset Metrics",
        f"- **Dataset Path**: `{os.path.abspath(dataset_root)}`",
        f"- **Total Categories**: {len(stats['categories'])}",
        f"- **Total Images**: {stats['total_images']:,}",
        f"- **Normal Images**: {stats['total_normal']:,} ({stats['total_normal']/max(1, stats['total_images'])*100:.1f}%)",
        f"- **Anomalous Images**: {stats['total_anomaly']:,} ({stats['total_anomaly']/max(1, stats['total_images'])*100:.1f}%)",
        "",
        "## Per-Category Statistics Table",
        "",
        "| Category Name | Category Structure Type | Normal Images | Anomaly Images | Train Set (Normal) | Test Set (Normal+Anomaly) | Total Images |",
        "| --- | --- | --- | --- | --- | --- | --- |"
    ]

    type_mapping = {
        "candle": "Single Instance",
        "capsules": "Multiple Instances",
        "cashew": "Single Instance",
        "chewinggum": "Multiple Instances",
        "fryum": "Single Instance",
        "macaroni1": "Multiple Instances",
        "macaroni2": "Multiple Instances",
        "pcb1": "Complex Structure (PCB)",
        "pcb2": "Complex Structure (PCB)",
        "pcb3": "Complex Structure (PCB)",
        "pcb4": "Complex Structure (PCB)",
        "pipe_fryum": "Multiple Instances"
    }

    for cat, cat_stat in stats["categories"].items():
        ctype = type_mapping.get(cat, "Industrial Object")
        md_lines.append(
            f"| `{cat}` | {ctype} | {cat_stat['normal']:,} | {cat_stat['anomaly']:,} | {cat_stat['train_samples']:,} | {cat_stat['test_samples']:,} | {cat_stat['total']:,} |"
        )

    md_lines.extend([
        "",
        "## Preprocessing & Data Pipeline Specification",
        "",
        "1. **Input Resizing**: Target dimension `(256, 256)` with bilinear interpolation for images and nearest-neighbor interpolation for pixel masks.",
        "2. **Color Mode**: 3-channel RGB float32 tensor.",
        "3. **Normalization**: Pixel intensities scaled to `[0.0, 1.0]` by dividing by `255.0`.",
        "4. **Self-Supervised Unsupervised Training Split**: Training split contains **ONLY normal images** to ensure no defect label leakage during representation learning.",
        "5. **Validation/Testing Split**: Evaluation dataset contains both normal and defect samples with ground-truth binary segmentation masks for pixel-level localization benchmark.",
        "6. **TensorFlow Pipeline**: Built with `tf.data.Dataset` featuring async decoding, dynamic batching (`batch_size=16`), shuffling, and `tf.data.AUTOTUNE` prefetching.",
        "",
        "## Official Reference & License",
        "- **Dataset Title**: Visual Anomaly (VisA) Dataset",
        "- **Paper**: *Spot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation* (ECCV 2022)",
        "- **Official Registry**: [https://registry.opendata.aws/visa/](https://registry.opendata.aws/visa/)",
        "- **License**: Apache 2.0 / Creative Commons License"
    ])

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Summary reports generated:\n  - JSON: {json_path}\n  - Markdown: {md_path}")

if __name__ == "__main__":
    generate_reports()
