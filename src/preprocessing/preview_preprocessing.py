import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.preprocessing.crop_kidney_region import orientation_aware_crop
from src.utils.config import IMG_SIZE


def preview_npy(npy_path):
    # Load original .npy
    original = np.load(npy_path)

    # Apply orientation-aware cropping
    cropped = orientation_aware_crop(original, crop_ratio=0.7)

    # Resize (as model sees it)
    resized = cv2.resize(cropped, IMG_SIZE)

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title(f"Original\n{original.shape}")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"After Cropping\n{cropped.shape}")
    plt.imshow(cropped, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Final Input\n{resized.shape}")
    plt.imshow(resized, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # CHANGE THIS PATH to a CORONAL image .npy
    preview_npy("data/processed/tumour/Tumor- (223).npy")
