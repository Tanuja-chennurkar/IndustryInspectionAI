"""
Self-Supervised Anomaly Model Trainer for VisA Categories.
Trains Convolutional Autoencoder on Normal images using SSIM + L1 loss.
"""

import os
import time
import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple

from ml.models.autoencoder import ConvAutoencoder
from ml.models.losses import SSIML1Loss

class SelfSupervisedTrainer:
    """
    Training pipeline for self-supervised visual reconstruction.
    """
    def __init__(
        self,
        model: ConvAutoencoder,
        learning_rate: float = 5e-4,
        alpha_ssim: float = 0.8
    ):
        self.model = model
        self.loss_fn = SSIML1Loss(alpha=alpha_ssim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(self, x_batch: tf.Tensor) -> tf.Tensor:
        """
        Executes a single optimization step.
        """
        with tf.GradientTape() as tape:
            x_reconstructed = self.model(x_batch, training=True)
            loss = self.loss_fn(x_batch, x_reconstructed)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    @tf.function
    def val_step(self, x_batch: tf.Tensor) -> tf.Tensor:
        """
        Executes validation step without gradients.
        """
        x_reconstructed = self.model(x_batch, training=False)
        loss = self.loss_fn(x_batch, x_reconstructed)
        return loss

    def fit(
        self,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
        epochs: int = 50,
        early_stopping_patience: int = 8,
        reduce_lr_patience: int = 3,
        reduce_lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        verbose: bool = True
    ) -> Dict:
        """
        Runs complete training loop with Early Stopping and ReduceLROnPlateau.
        """
        history = {"loss": [], "val_loss": [], "learning_rate": [], "epoch_times": []}
        
        best_loss = float("inf")
        best_weights = None
        no_improvement_count = 0
        lr_no_improvement_count = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0.0
            steps = 0

            for batch in train_ds:
                loss = self.train_step(batch)
                total_loss += float(loss)
                steps += 1

            avg_train_loss = total_loss / max(1, steps)
            history["loss"].append(avg_train_loss)

            avg_val_loss = 0.0
            if val_ds is not None:
                val_total = 0.0
                val_steps = 0
                for v_batch in val_ds:
                    if isinstance(v_batch, tuple):
                        v_img = v_batch[0]
                    else:
                        v_img = v_batch
                    v_loss = self.val_step(v_img)
                    val_total += float(v_loss)
                    val_steps += 1
                avg_val_loss = val_total / max(1, val_steps)
                history["val_loss"].append(avg_val_loss)
                monitored_loss = avg_val_loss
            else:
                monitored_loss = avg_train_loss

            current_lr = float(self.optimizer.learning_rate.numpy())
            history["learning_rate"].append(current_lr)

            elapsed = time.time() - start_time
            history["epoch_times"].append(elapsed)

            if verbose:
                val_str = f" - val_loss: {avg_val_loss:.5f}" if val_ds is not None else ""
                print(f"Epoch {epoch:02d}/{epochs:02d} - loss: {avg_train_loss:.5f}{val_str} - lr: {current_lr:.6f} - time: {elapsed:.2f}s")

            # Check for best loss & weight restoration
            if monitored_loss < best_loss - 1e-4:
                best_loss = monitored_loss
                best_weights = self.model.get_weights()
                no_improvement_count = 0
                lr_no_improvement_count = 0
            else:
                no_improvement_count += 1
                lr_no_improvement_count += 1

            # Reduce LR on plateau
            if lr_no_improvement_count >= reduce_lr_patience and current_lr > min_lr:
                new_lr = max(current_lr * reduce_lr_factor, min_lr)
                self.optimizer.learning_rate.assign(new_lr)
                if verbose:
                    print(f"   [ReduceLROnPlateau] Reducing learning rate to {new_lr:.6f}")
                lr_no_improvement_count = 0

            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                if verbose:
                    print(f"   [EarlyStopping] Early stopping triggered at epoch {epoch}. Restoring best weights (best loss: {best_loss:.5f})")
                if best_weights is not None:
                    self.model.set_weights(best_weights)
                break

        if best_weights is not None and no_improvement_count < early_stopping_patience:
            self.model.set_weights(best_weights)

        return history
