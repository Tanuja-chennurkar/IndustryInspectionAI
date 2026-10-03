"""
VisA Dataset Loader and tf.data Pipeline Generator.
Reads original VisA directory structure without modifying raw data.
Supports all 12 VisA categories and multi-category operation.
"""

import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Union
from ml.preprocessing.preprocessor import ImagePreprocessor

VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum"
]

class VisADataset:
    """
    Data manager and tf.data loader for the Visual Anomaly (VisA) dataset.
    """
    def __init__(
        self,
        dataset_root: str = "ml/data/VisA",
        categories: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (256, 256),
        batch_size: int = 16,
        seed: int = 42
    ):
        self.dataset_root = os.path.abspath(dataset_root)
        self.categories = categories or VISA_CATEGORIES
        self.image_size = tuple(image_size)
        self.batch_size = batch_size
        self.seed = seed
        self.preprocessor = ImagePreprocessor(target_size=image_size)
        self.metadata_df = pd.DataFrame()

    def discover_and_index_files(self) -> pd.DataFrame:
        """
        Scans VisA dataset root directory for all 12 categories.
        Looks for split_csv/full.csv first, or performs full filesystem indexing.
        """
        records = []
        full_csv = os.path.join(self.dataset_root, "split_csv", "full.csv")
        
        if os.path.exists(full_csv):
            df = pd.read_csv(full_csv)
            # Ensure proper path resolution
            for idx, row in df.iterrows():
                rel_img = str(row.get('image_path', '')).strip().replace('\\', os.sep)
                rel_mask = str(row.get('mask_path', '')).strip().replace('\\', os.sep)
                cat = str(row.get('category', row.get('object', ''))).strip()
                label_str = str(row.get('label', '')).strip().lower()
                split = str(row.get('split', 'train')).strip().lower()
                
                abs_img = os.path.join(self.dataset_root, rel_img)
                abs_mask = os.path.join(self.dataset_root, rel_mask) if rel_mask and rel_mask != 'nan' else ""
                
                records.append({
                    "image_path": abs_img,
                    "mask_path": abs_mask,
                    "category": cat,
                    "label_str": label_str,
                    "label": 0 if label_str == "normal" else 1,
                    "split": split
                })
        else:
            # Fallback: scan category directories
            for cat in self.categories:
                cat_dir = os.path.join(self.dataset_root, cat)
                if not os.path.exists(cat_dir):
                    continue

                # 1. Normal images
                normal_pattern = os.path.join(cat_dir, "Data", "Images", "Normal", "*.*")
                normal_files = sorted([f for f in glob.glob(normal_pattern) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                # Split normal images: 60% train, 40% test
                num_normal = len(normal_files)
                num_train = int(num_normal * 0.6)
                
                for idx, fpath in enumerate(normal_files):
                    split = "train" if idx < num_train else "test"
                    records.append({
                        "image_path": fpath,
                        "mask_path": "",
                        "category": cat,
                        "label_str": "normal",
                        "label": 0,
                        "split": split
                    })

                # 2. Anomaly images & masks
                anomaly_pattern = os.path.join(cat_dir, "Data", "Images", "Anomaly", "*.*")
                anomaly_files = sorted([f for f in glob.glob(anomaly_pattern) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                mask_dir = os.path.join(cat_dir, "Data", "Masks", "Anomaly")
                for fpath in anomaly_files:
                    fname = os.path.basename(fpath)
                    basename, _ = os.path.splitext(fname)
                    mask_path = os.path.join(mask_dir, f"{basename}.png")
                    if not os.path.exists(mask_path):
                        # try matching extensions
                        mask_candidates = glob.glob(os.path.join(mask_dir, f"{basename}.*"))
                        mask_path = mask_candidates[0] if mask_candidates else ""
                        
                    records.append({
                        "image_path": fpath,
                        "mask_path": mask_path,
                        "category": cat,
                        "label_str": "anomaly",
                        "label": 1,
                        "split": "test"
                    })

        self.metadata_df = pd.DataFrame(records)
        return self.metadata_df

    def get_summary_statistics(self) -> Dict:
        """
        Calculates detailed per-category and total dataset statistics.
        """
        if self.metadata_df.empty:
            self.discover_and_index_files()

        stats = {
            "total_images": len(self.metadata_df),
            "total_normal": len(self.metadata_df[self.metadata_df["label"] == 0]) if not self.metadata_df.empty else 0,
            "total_anomaly": len(self.metadata_df[self.metadata_df["label"] == 1]) if not self.metadata_df.empty else 0,
            "categories": {}
        }

        for cat in self.categories:
            cat_df = self.metadata_df[self.metadata_df["category"] == cat] if not self.metadata_df.empty else pd.DataFrame()
            normal_cnt = len(cat_df[cat_df["label"] == 0]) if not cat_df.empty else 0
            anomaly_cnt = len(cat_df[cat_df["label"] == 1]) if not cat_df.empty else 0
            train_cnt = len(cat_df[cat_df["split"] == "train"]) if not cat_df.empty else 0
            test_cnt = len(cat_df[cat_df["split"] == "test"]) if not cat_df.empty else 0

            stats["categories"][cat] = {
                "total": len(cat_df),
                "normal": normal_cnt,
                "anomaly": anomaly_cnt,
                "train_samples": train_cnt,
                "test_samples": test_cnt
            }

        return stats

    def create_tf_dataset(
        self,
        category: Optional[str] = None,
        split: str = "train",
        is_training: bool = True
    ) -> tf.data.Dataset:
        """
        Creates an optimized tf.data.Dataset pipeline.
        
        In train split (unsupervised/self-supervised mode):
            - Loads ONLY normal images.
            - Output element: image tensor of shape (256, 256, 3)
            
        In test/val split (evaluation mode):
            - Loads normal & anomaly images + ground truth masks + labels.
            - Output element: (image_tensor, mask_tensor, label_tensor)
        """
        if self.metadata_df.empty:
            self.discover_and_index_files()

        df = self.metadata_df.copy()
        if category and category != "all":
            df = df[df["category"] == category]

        if split:
            df = df[df["split"] == split]

        # For self-supervised training, filter normal samples only
        if split == "train":
            df = df[df["label"] == 0]

        if len(df) == 0:
            raise ValueError(f"No samples found for category='{category}', split='{split}'")

        image_paths = df["image_path"].tolist()
        mask_paths = df["mask_path"].tolist()
        labels = df["label"].tolist()

        if split == "train":
            dataset = tf.data.Dataset.from_tensor_slices(image_paths)
            if is_training:
                dataset = dataset.shuffle(buffer_size=min(len(image_paths), 1000), seed=self.seed)

            dataset = dataset.map(
                lambda p: self.preprocessor.preprocess_image_path(p),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths, labels))
            dataset = dataset.map(
                lambda img_p, mask_p, lbl: (
                    self.preprocessor.preprocess_image_path(img_p),
                    self.preprocessor.preprocess_mask_path(mask_p),
                    tf.cast(lbl, tf.int32)
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
