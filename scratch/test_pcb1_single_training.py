import os
import sys
import time
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset
from ml.models.autoencoder import ConvAutoencoder
from ml.models.losses import SSIML1Loss
from ml.training.trainer import SelfSupervisedTrainer
from ml.preprocessing.preprocessor import ImagePreprocessor
from ml.inference.detector import AnomalyDetector

def test_loss_scale_and_train_pcb1():
    print("=" * 80)
    print("   PART 2 & 3: PCB1 SINGLE CATEGORY TRAINING & VALIDATION   ")
    print("=" * 80)

    seed = 42
    tf.keras.utils.set_random_seed(seed)

    dataset = VisADataset(batch_size=16, seed=seed)
    train_ds = dataset.create_tf_dataset(category="pcb1", split="train", is_training=True)
    val_ds = dataset.create_tf_dataset(category="pcb1", split="test", is_training=False)

    model = ConvAutoencoder(input_shape=(256, 256, 3), latent_dim=256)
    # Build model explicitly
    model.build((None, 256, 256, 3))
    print(f"Compact Autoencoder Total Parameters: {model.count_params():,}")

    # Part 2: Verify numerical scale of SSIM and MAE before training
    print("\n--- Part 2: Verifying Numerical Loss Components ---")
    loss_fn = SSIML1Loss(alpha=0.8)
    
    for b_idx, batch in enumerate(train_ds.take(3)):
        recon_dummy = model(batch, training=False)
        ssim_comp = float(tf.reduce_mean(1.0 - tf.image.ssim(batch, recon_dummy, max_val=1.0)))
        mae_comp = float(tf.reduce_mean(tf.abs(batch - recon_dummy)))
        total_l = float(loss_fn(batch, recon_dummy))

        print(f"Batch {b_idx+1}:")
        print(f"  - SSIM component (1 - SSIM): {ssim_comp:.4f} (Weighted 0.8: {0.8*ssim_comp:.4f})")
        print(f"  - MAE component (MAE)      : {mae_comp:.4f} (Weighted 0.2: {0.2*mae_comp:.4f})")
        print(f"  - Combined Total Loss      : {total_l:.4f}")

    # Part 2 Training
    print("\n--- Part 2: Training PCB1 ConvAutoencoder (Max 50 Epochs) ---")
    trainer = SelfSupervisedTrainer(model=model, learning_rate=5e-4, alpha_ssim=0.8)
    start_time = time.time()
    history = trainer.fit(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=50,
        early_stopping_patience=8,
        reduce_lr_patience=3,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
        verbose=True
    )
    elapsed = time.time() - start_time
    print(f"\n[PCB1 Training Completed in {elapsed:.2f}s]")

    # Save PCB1 model weights
    save_dir = os.path.abspath("ml/saved_models/pcb1")
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "model.weights.h5")
    model.save_weights(weights_path)
    print(f"[PCB1 Model Saved] -> {weights_path}")

    # Part 3: Critical Autoencoder Validation on several known normal PCB1 images
    print("\n" + "=" * 80)
    print("   PART 3: CRITICAL PCB1 RECONSTRUCTION VALIDATION   ")
    print("=" * 80)

    preprocessor = ImagePreprocessor(target_size=(256, 256))
    df = dataset.discover_and_index_files()
    pcb1_normal_rows = df[(df["category"] == "pcb1") & (df["label"] == 0)].head(4)

    for idx, row in pcb1_normal_rows.reset_index().iterrows():
        img_path = row["image_path"]
        input_tensor = preprocessor.preprocess_image_path(img_path)
        input_batched = tf.expand_dims(input_tensor, axis=0)

        recon_tensor = model(input_batched, training=False)

        img_np = input_batched.numpy()[0]
        recon_np = recon_tensor.numpy()[0]

        in_min, in_max, in_mean = float(np.min(img_np)), float(np.max(img_np)), float(np.mean(img_np))
        rec_min, rec_max, rec_mean = float(np.min(recon_np)), float(np.max(recon_np)), float(np.mean(recon_np))

        mae_val = float(np.mean(np.abs(img_np - recon_np)))
        ssim_val = float(tf.reduce_mean(tf.image.ssim(input_batched, recon_tensor, max_val=1.0)).numpy())
        score_val = 0.8 * (1.0 - ssim_val) + 0.2 * mae_val

        print(f"\nTest Image #{idx+1}: {os.path.basename(img_path)}")
        print(f"  - Input min/max/mean        : [{in_min:.4f}, {in_max:.4f}, {in_mean:.4f}]")
        print(f"  - Reconstruction min/max/mean: [{rec_min:.4f}, {rec_max:.4f}, {rec_mean:.4f}]")
        print(f"  - MAE: {mae_val:.4f} | SSIM: {ssim_val:.4f} | Anomaly Score: {score_val:.4f}")

        # Check for constant gray collapse
        spatial_range = rec_max - rec_min
        print(f"  - Reconstruction Spatial Dynamic Range (Max - Min): {spatial_range:.4f}")
        if spatial_range > 0.30:
            print("  - [CHECK PASSED] Reconstruction shows rich spatial variation!")
        else:
            print("  - [WARNING] Low spatial dynamic range.")

if __name__ == "__main__":
    test_loss_scale_and_train_pcb1()
