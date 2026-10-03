import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset
from ml.models.losses import SSIML1Loss
from ml.preprocessing.preprocessor import ImagePreprocessor

def build_no_bn_autoencoder():
    # Encoder
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation=layers.LeakyReLU(0.2))(inputs)
    x = layers.Conv2D(64, (3, 3), strides=2, padding="same", activation=layers.LeakyReLU(0.2))(x)
    x = layers.Conv2D(128, (3, 3), strides=2, padding="same", activation=layers.LeakyReLU(0.2))(x)
    latent = layers.Conv2D(256, (3, 3), strides=2, padding="same", activation=layers.LeakyReLU(0.2))(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding="same", activation=layers.LeakyReLU(0.2))(latent)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation=layers.LeakyReLU(0.2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation=layers.LeakyReLU(0.2))(x)
    outputs = layers.Conv2DTranspose(3, (3, 3), strides=2, padding="same", activation="sigmoid")(x)

    return Model(inputs, outputs, name="no_bn_autoencoder")

def test_no_bn_training():
    print("=" * 80)
    print("   TESTING CONVAUTOENCODER WITHOUT BATCH NORMALIZATION   ")
    print("=" * 80)

    dataset = VisADataset(batch_size=16, seed=42)
    train_ds = dataset.create_tf_dataset(category="pcb1", split="train", is_training=True)

    model = build_no_bn_autoencoder()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = SSIML1Loss(alpha=0.8)

    for epoch in range(1, 41):
        total_loss = 0.0
        steps = 0
        for batch in train_ds:
            with tf.GradientTape() as tape:
                recon = model(batch, training=True)
                loss = loss_fn(batch, recon)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += float(loss)
            steps += 1
        avg_loss = total_loss / max(1, steps)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} - loss: {avg_loss:.5f}")

    # Inspect test image reconstruction
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    df = dataset.discover_and_index_files()
    img_path = df[(df["category"] == "pcb1") & (df["label"] == 0)].iloc[0]["image_path"]

    input_tensor = preprocessor.preprocess_image_path(img_path)
    input_batched = tf.expand_dims(input_tensor, axis=0)

    recon_tensor = model(input_batched, training=False)

    img_np = input_batched.numpy()[0]
    recon_np = recon_tensor.numpy()[0]

    in_min, in_max, in_mean = float(np.min(img_np)), float(np.max(img_np)), float(np.mean(img_np))
    rec_min, rec_max, rec_mean = float(np.min(recon_np)), float(np.max(recon_np)), float(np.mean(recon_np))

    print(f"\nNo-BN Model Reconstruction Stats on {os.path.basename(img_path)}:")
    print(f"  - Input min/max/mean        : [{in_min:.4f}, {in_max:.4f}, {in_mean:.4f}]")
    print(f"  - Reconstruction min/max/mean: [{rec_min:.4f}, {rec_max:.4f}, {rec_mean:.4f}]")
    print(f"  - Spatial Dynamic Range     : {rec_max - rec_min:.4f}")

if __name__ == "__main__":
    test_no_bn_training()
