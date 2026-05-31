import os
import cv2
import numpy as np
from src.preprocessing.crop_kidney_region import orientation_aware_crop
from src.utils.config import IMG_SIZE, CLIP_LIMIT, TILE_GRID_SIZE

PROCESSED_DATA_DIR = "E:/kidneyinsight/data/processed"
OUTPUT_DIR = "E:/kidneyinsight/data/processed_clahe"

os.makedirs(os.path.join(OUTPUT_DIR, "tumour"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "non_tumour"), exist_ok=True)


def hu_window(image, min_val=0, max_val=255):
    """
    Safe intensity windowing for non-DICOM CT images.
    """
    image = image.astype("float32")
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val)
    image = (image * 255).astype("uint8")
    return image


def apply_clahe(image):
    """
    Normalize intensity and apply CLAHE
    """
    # Normalize intensity to 0–255
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = image.astype("uint8")

    # Apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT,
        tileGridSize=TILE_GRID_SIZE
    )
    return clahe.apply(image)


def process_folder(class_name):
    input_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
    output_dir = os.path.join(OUTPUT_DIR, class_name)

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)

        image = np.load(file_path)

        # Orientation-aware cropping
        image = orientation_aware_crop(image, crop_ratio=0.7)

        # HU windowing (kidney-specific)
        image = hu_window(image, min_val=0, max_val=255)

        # CLAHE
        enhanced = apply_clahe(image)


        # Resize
        enhanced = cv2.resize(enhanced, IMG_SIZE)

        # Save
        save_path = os.path.join(output_dir, file_name)
        np.save(save_path, enhanced)


# Process both classes
process_folder("tumour")
process_folder("non_tumour")

print("✅ CLAHE preprocessing completed.")
