"""
Phase 2 Baseline Model Training Script.
Trains Category-Aware Self-Supervised Convolutional Autoencoders on VisA Normal Samples.
"""

import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset, VISA_CATEGORIES
from ml.datasets.prepare_dataset import ensure_visa_structure
from ml.models.category_model import CategoryAwareAnomalyDetector
from ml.training.trainer import SelfSupervisedTrainer

def train_phase2(categories=None, epochs=50, batch_size=16, learning_rate=5e-4, seed=42, dataset_root="ml/data/VisA"):
    """
    Main training execution function for Phase 2 baseline models.
    """
    print("=" * 80)
    print("   INDUSTRIAL VISUAL INSPECTION - PHASE 2 BASELINE MODEL TRAINING   ")
    print("=" * 80)

    tf.keras.utils.set_random_seed(seed)
    ensure_visa_structure(dataset_root)
    dataset = VisADataset(dataset_root=dataset_root, batch_size=batch_size, seed=seed)
    detector = CategoryAwareAnomalyDetector(categories=VISA_CATEGORIES)

    target_cats = categories if categories and categories != ["all"] else VISA_CATEGORIES

    training_summary = {}

    for cat in target_cats:
        print(f"\n>>> Training Self-Supervised Normality Model for Category: [{cat}]")
        
        # Train dataset loads ONLY normal images (unsupervised self-supervised requirement)
        train_ds = dataset.create_tf_dataset(category=cat, split="train", is_training=True)
        val_ds = dataset.create_tf_dataset(category=cat, split="test", is_training=False)

        model = detector.get_or_create_model(cat)
        trainer = SelfSupervisedTrainer(model=model, learning_rate=learning_rate, alpha_ssim=0.8)

        start_time = time.time()
        history = trainer.fit(
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=epochs,
            early_stopping_patience=8,
            reduce_lr_patience=3,
            reduce_lr_factor=0.5,
            min_lr=1e-6,
            verbose=True
        )
        train_time = time.time() - start_time

        # Calculate baseline anomaly threshold from validation normal samples
        normal_losses = []
        for batch in train_ds:
            recon = model(batch, training=False)
            batch_loss = tf.reduce_mean(tf.abs(batch - recon), axis=[1, 2, 3]).numpy()
            normal_losses.extend(batch_loss)

        # Set threshold as max(Mean * 1.05, Mean + 3 * StdDev) to prevent normal sample false positives
        mean_loss = float(np.mean(normal_losses))
        std_loss = float(np.std(normal_losses))
        threshold = max(mean_loss * 1.05, mean_loss + (3.0 * max(std_loss, 0.01)))

        # Save model and metadata
        save_dir = detector.save_category_model(
            category=cat,
            threshold=threshold,
            version="v1.0.0",
            training_config={
                "epochs": epochs,
                "batch_size": batch_size,
                "final_loss": history["loss"][-1],
                "normal_mean_loss": mean_loss,
                "normal_std_loss": std_loss,
                "training_time_seconds": train_time
            }
        )

        training_summary[cat] = {
            "final_loss": float(history["loss"][-1]),
            "threshold": float(threshold),
            "save_dir": save_dir,
            "training_time": train_time
        }

    print("\n" + "=" * 80)
    print("                      PHASE 2 TRAINING SUMMARY REPORT                       ")
    print("=" * 80)
    print(f"{'Category':<15} | {'Final Loss':<12} | {'Normality Threshold':<20} | {'Status'}")
    print("-" * 80)
    for cat, res in training_summary.items():
        print(f"{cat:<15} | {res['final_loss']:<12.5f} | {res['threshold']:<20.5f} | TRAINED & SAVED")
    print("=" * 80)
    return training_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Baseline Model Trainer")
    parser.add_argument("--categories", nargs="+", default=["all"], help="Category or list of categories to train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per category")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_phase2(categories=args.categories, epochs=args.epochs, batch_size=args.batch_size)
