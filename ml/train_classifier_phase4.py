"""
Phase 4 Category Classifier Training Script.
Trains 12-class industrial product category classifier for automatic image routing.
"""

import os
import sys
import argparse
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.datasets.visa_dataset import VisADataset, VISA_CATEGORIES
from ml.datasets.prepare_dataset import ensure_visa_structure
from ml.models.category_classifier import CategoryClassifier
from ml.preprocessing.preprocessor import ImagePreprocessor

def train_category_classifier(epochs=40, batch_size=16, learning_rate=1e-3, seed=42, dataset_root="ml/data/VisA"):
    """
    Trains MobileNetV2 CategoryClassifier on all 12 VisA categories.
    """
    print("=" * 80)
    print("   INDUSTRIAL VISUAL INSPECTION - PHASE 4 CATEGORY CLASSIFIER TRAINING   ")
    print("=" * 80)

    tf.keras.utils.set_random_seed(seed)
    ensure_visa_structure(dataset_root)
    dataset = VisADataset(dataset_root=dataset_root, seed=seed)
    df = dataset.discover_and_index_files()

    preprocessor = ImagePreprocessor(target_size=(256, 256))
    category_to_idx = {cat: idx for idx, cat in enumerate(VISA_CATEGORIES)}

    # Shuffle dataframe
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = int(len(df_shuffled) * 0.2)
    train_df = df_shuffled.iloc[val_size:]
    val_df = df_shuffled.iloc[:val_size]

    def create_dataset_from_df(sub_df, is_train=True):
        img_paths = sub_df["image_path"].tolist()
        labels = [category_to_idx[cat] for cat in sub_df["category"].tolist()]
        ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        if is_train:
            ds = ds.shuffle(buffer_size=len(img_paths), seed=seed)
        ds = ds.map(
            lambda p, l: (preprocessor.preprocess_image_path(p), l),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = create_dataset_from_df(train_df, is_train=True)
    val_ds = create_dataset_from_df(val_df, is_train=False)

    classifier = CategoryClassifier(categories=VISA_CATEGORIES)
    classifier.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    ]

    print(f"\n>>> Training MobileNetV2 Category Classifier across {len(VISA_CATEGORIES)} categories...")
    print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")
    classifier.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    os.makedirs("ml/saved_models", exist_ok=True)
    save_path = os.path.join("ml/saved_models", "category_classifier.keras")
    classifier.model.save(save_path)

    print(f"\n[SUCCESS] MobileNetV2 Category Classifier trained and saved -> {save_path}")
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Category Classifier Trainer")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_category_classifier(epochs=args.epochs, batch_size=args.batch_size)
