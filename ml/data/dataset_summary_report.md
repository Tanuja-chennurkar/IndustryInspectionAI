# VisA Dataset Summary & Preprocessing Specification Report

## Executive Summary
The **Visual Anomaly (VisA)** dataset is an official benchmark dataset for industrial visual inspection and anomaly detection.
It features high-resolution images covering **12 object categories** spanning single-instance objects, multi-instance objects, and complex printed circuit boards (PCBs).

## Key Dataset Metrics
- **Dataset Path**: `D:\BTECH\project\IndustryInspectionAI\ml\data\VisA`
- **Total Categories**: 12
- **Total Images**: 240
- **Normal Images**: 180 (75.0%)
- **Anomalous Images**: 60 (25.0%)

## Per-Category Statistics Table

| Category Name | Category Structure Type | Normal Images | Anomaly Images | Train Set (Normal) | Test Set (Normal+Anomaly) | Total Images |
| --- | --- | --- | --- | --- | --- | --- |
| `candle` | Single Instance | 15 | 5 | 9 | 11 | 20 |
| `capsules` | Multiple Instances | 15 | 5 | 9 | 11 | 20 |
| `cashew` | Single Instance | 15 | 5 | 9 | 11 | 20 |
| `chewinggum` | Multiple Instances | 15 | 5 | 9 | 11 | 20 |
| `fryum` | Single Instance | 15 | 5 | 9 | 11 | 20 |
| `macaroni1` | Multiple Instances | 15 | 5 | 9 | 11 | 20 |
| `macaroni2` | Multiple Instances | 15 | 5 | 9 | 11 | 20 |
| `pcb1` | Complex Structure (PCB) | 15 | 5 | 9 | 11 | 20 |
| `pcb2` | Complex Structure (PCB) | 15 | 5 | 9 | 11 | 20 |
| `pcb3` | Complex Structure (PCB) | 15 | 5 | 9 | 11 | 20 |
| `pcb4` | Complex Structure (PCB) | 15 | 5 | 9 | 11 | 20 |
| `pipe_fryum` | Multiple Instances | 15 | 5 | 9 | 11 | 20 |

## Preprocessing & Data Pipeline Specification

1. **Input Resizing**: Target dimension `(256, 256)` with bilinear interpolation for images and nearest-neighbor interpolation for pixel masks.
2. **Color Mode**: 3-channel RGB float32 tensor.
3. **Normalization**: Pixel intensities scaled to `[0.0, 1.0]` by dividing by `255.0`.
4. **Self-Supervised Unsupervised Training Split**: Training split contains **ONLY normal images** to ensure no defect label leakage during representation learning.
5. **Validation/Testing Split**: Evaluation dataset contains both normal and defect samples with ground-truth binary segmentation masks for pixel-level localization benchmark.
6. **TensorFlow Pipeline**: Built with `tf.data.Dataset` featuring async decoding, dynamic batching (`batch_size=16`), shuffling, and `tf.data.AUTOTUNE` prefetching.

## Official Reference & License
- **Dataset Title**: Visual Anomaly (VisA) Dataset
- **Paper**: *Spot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation* (ECCV 2022)
- **Official Registry**: [https://registry.opendata.aws/visa/](https://registry.opendata.aws/visa/)
- **License**: Apache 2.0 / Creative Commons License