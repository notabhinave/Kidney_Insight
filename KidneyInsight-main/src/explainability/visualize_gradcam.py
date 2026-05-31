import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from src.explainability.grad_cam import make_gradcam_heatmap, overlay_heatmap
from src.utils.config import IMG_SIZE


MODEL_PATH = "best_model.h5"


def load_image(image_path):
    # Check if it is an NPY file
    if image_path.endswith('.npy'):
        image = np.load(image_path)
        
        # If image is grayscale (2D), make it 3D (RGB)
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
            
        # Ensure it is the right size (224x224)
        if image.shape[:2] != IMG_SIZE:
            image = cv2.resize(image, IMG_SIZE)
            
        # Create a "visual" version (0-255) and a "model" version (0-1)
        if image.max() <= 1.0:
            original = (image * 255).astype('uint8')
            processed = image # Already 0-1
        else:
            original = image.astype('uint8')
            processed = image.astype('float32') / 255.0

    # If it is a normal JPG/PNG image
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        image = cv2.resize(image, IMG_SIZE)
        original = image
        processed = image.astype('float32') / 255.0

    # Expand dims for the model (1, 224, 224, 3)
    processed = np.expand_dims(processed, axis=0)
    
    return original, processed


def visualize(image_path):
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load image
    original, processed = load_image(image_path)

    # Predict
    prob = model.predict(processed)[0][0]

    print(f"Tumour Probability: {prob:.4f}")

    if prob < 0.3:
        print("Prediction: NON-TUMOUR → Grad-CAM skipped")
        return

    print("Prediction: TUMOUR → Applying Grad-CAM")

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(processed, model)

    # Overlay heatmap
    overlay = overlay_heatmap(
        cv2.cvtColor(original, cv2.COLOR_GRAY2BGR),
        heatmap
    )

    # Display
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original CT")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    # Point to YOUR tumour folder
    # Note: Using raw string (r"...") to handle Windows backslashes
    folder_path = r"C:\Users\Asus\Desktop\CODE\KidneyInsight-main\data\processed_clahe\tumour"
    
    # Automatically pick the first file in the folder
    if os.path.exists(folder_path) and len(os.listdir(folder_path)) > 0:
        file_name = os.listdir(folder_path)[0]
        full_path = os.path.join(folder_path, file_name)
        
        print(f"Visualizing file: {full_path}")
        visualize(full_path)
    else:
        print("Error: No .npy files found in the tumour folder!")
