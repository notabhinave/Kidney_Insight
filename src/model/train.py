import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.model.build_model import build_model
from src.model.generator import KidneyDataGenerator
from src.utils.config import EPOCHS, BATCH_SIZE


# Directory containing preprocessed .npy files
DATA_DIR = "data/processed_clahe"


def collect_files():
    """
    Collect all file paths and labels.
    Tumour -> 1
    Non-tumour -> 0
    """
    file_paths = []
    labels = []

    tumour_dir = os.path.join(DATA_DIR, "tumour")
    non_tumour_dir = os.path.join(DATA_DIR, "non_tumour")

    for fname in os.listdir(tumour_dir):
        file_paths.append(os.path.join(tumour_dir, fname))
        labels.append(1)

    for fname in os.listdir(non_tumour_dir):
        file_paths.append(os.path.join(non_tumour_dir, fname))
        labels.append(0)

    return np.array(file_paths), np.array(labels)


def train():
    print("🔹 Collecting dataset files...")
    files, labels = collect_files()

    print(f"Total samples: {len(files)}")

    # optional bounding boxes file produced by pseudo‑label script
    bbox_dict = {}
    boxes_path = os.path.join(DATA_DIR, 'bbox_dict.npy')
    if os.path.exists(boxes_path):
        print("Loading bounding‑box dictionary for localisation")
        bbox_dict = np.load(boxes_path, allow_pickle=True).item()
        # keys should match absolute or relative paths used in collect_files()

    # Train–validation split
    X_train, X_val, y_train, y_val = train_test_split(
        files,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    # Data generators (batch-wise loading) – pass bbox dict to each
    train_gen = KidneyDataGenerator(
        X_train, y_train, bbox_dict=bbox_dict, batch_size=BATCH_SIZE, shuffle=True
    )

    val_gen = KidneyDataGenerator(
        X_val, y_val, bbox_dict=bbox_dict, batch_size=BATCH_SIZE, shuffle=False
    )

    # Build model
    model = build_model()
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="best_model.h5",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    print("🚀 Starting training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )

    print("✅ Training completed. Best model saved as best_model.h5")


if __name__ == "__main__":
    train()
