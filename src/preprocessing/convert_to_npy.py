import os
import cv2
import numpy as np

# Paths
RAW_DATA_DIR = "E:/kidneyinsight/data/raw"
PROCESSED_DATA_DIR = "E:/kidneyinsight/data/processed"

TUMOUR_CLASSES = ["tumour"]
NON_TUMOUR_CLASSES = ["normal", "cyst", "stone"]

# Create output folders if they don't exist
os.makedirs(os.path.join(PROCESSED_DATA_DIR, "tumour"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, "non_tumour"), exist_ok=True)


def convert_and_save(class_list, output_folder):
    for class_name in class_list:
        input_dir = os.path.join(RAW_DATA_DIR, class_name)

        for file_name in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file_name)

            # Read image in grayscale
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Skipping invalid image: {file_path}")
                continue

            # Convert to numpy array and save
            save_name = file_name.split('.')[0] + ".npy"
            save_path = os.path.join(PROCESSED_DATA_DIR, output_folder, save_name)

            np.save(save_path, image)


# Convert Tumour images
convert_and_save(TUMOUR_CLASSES, "tumour")

# Convert Non-Tumour images
convert_and_save(NON_TUMOUR_CLASSES, "non_tumour")

print("✅ Conversion to .npy completed successfully.")
