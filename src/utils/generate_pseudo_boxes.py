"""Generate bounding-box pseudo labels using existing classifier + Grad-CAM.

Saves a dict mapping image path -> (xmin,ymin,xmax,ymax) normalized to [0,1].
Only processes files in the 'tumour' folder; you can extend to others.
"""
import os
import glob
import numpy as np
import cv2
import tensorflow as tf
import time
import sys
import logging

from src.explainability.grad_cam import make_gradcam_heatmap
from src.utils.config import IMG_SIZE

MODEL_PATH = "best_model.h5"
# workspace root contains 'processed_clahe' rather than 'data/processed_clahe'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT, "processed_clahe")
OUTPUT_PATH = os.path.join(DATA_DIR, "bbox_dict.npy")
LOG_PATH = os.path.join(DATA_DIR, "bbox_generation.log")


def main():
    # Prepare logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="a"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Loading model %s", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    tumour_dir = os.path.join(DATA_DIR, "tumour")
    if not os.path.isdir(tumour_dir):
        raise RuntimeError(f"Expected tumour directory at {tumour_dir}")

    files = sorted(glob.glob(os.path.join(tumour_dir, "*.npy")))
    total = len(files)
    logging.info("Found %d tumour files to process", total)

    bbox_dict = {}
    start_time = time.time()

    try:
        for idx, path in enumerate(files, start=1):
            logging.info("Processing (%d/%d): %s", idx, total, path)
            try:
                img = np.load(path)

                # make sure image is 2D
                if img.ndim == 3:
                    img = img[..., 0]
                img_resized = cv2.resize(img, IMG_SIZE)

                proc = img_resized.astype("float32")
                if proc.max() > 1.0:
                    proc = proc / 255.0
                proc = np.stack([proc, proc, proc], axis=-1)
                proc = np.expand_dims(proc, 0)

                heat = make_gradcam_heatmap(proc, model)
                heat = cv2.resize(heat, IMG_SIZE)

                gray = img_resized.astype("uint8")
                mask = (gray > 10).astype("uint8")
                heat = heat * mask

                _, th = cv2.threshold((heat * 255).astype("uint8"), 20, 255, cv2.THRESH_BINARY)
                # morphological closing to join nearby regions
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    # normalize
                    h_img, w_img = IMG_SIZE
                    bbox_dict[path] = (x / w_img, y / h_img, (x + w) / w_img, (y + h) / h_img)
                    logging.info("-> Saved bbox for %s -> %s", os.path.basename(path), bbox_dict[path])
                else:
                    logging.info("-> No contour found for %s", os.path.basename(path))

            except Exception:
                logging.exception("Error processing %s", path)

            # periodic save + ETA (save more frequently to avoid losing work)
            if idx % 10 == 0 or idx == total:
                try:
                    np.save(OUTPUT_PATH, bbox_dict)
                    logging.info("Saved %d pseudo boxes to %s (progress %d/%d)", len(bbox_dict), OUTPUT_PATH, idx, total)
                except Exception:
                    logging.exception("Failed saving bbox_dict at progress %d", idx)

            elapsed = time.time() - start_time
            avg = elapsed / idx
            remaining = avg * (total - idx)
            logging.info("Elapsed: %.1fs, ETA: %.1fs", elapsed, remaining)

    except KeyboardInterrupt:
        logging.warning("Interrupted by user. Saving progress to %s", OUTPUT_PATH)
        np.save(OUTPUT_PATH, bbox_dict)
        logging.info("Saved %d pseudo boxes before interruption", len(bbox_dict))
        return

    # final save (again)
    np.save(OUTPUT_PATH, bbox_dict)
    logging.info("Finished. Saved pseudo boxes for %d images to %s", len(bbox_dict), OUTPUT_PATH)


if __name__ == "__main__":
    main()
