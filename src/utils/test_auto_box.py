"""Test auto-boxing on 10 sample tumor images."""
import os
import glob
import numpy as np
import cv2
import random

DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TUMOUR_DIR = os.path.join(DATA_DIR, "processed_clahe", "tumour")

def auto_box_image(img_array):
    """Generate bounding box from kidney region (auto-detect)."""
    # img_array is 2D grayscale CT scan
    
    # Use Otsu threshold to find tissue region
    cv2.normalize(img_array, img_array, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(img_array.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    h_img, w_img = img_array.shape
    # sort by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    selected = None
    # pick first contour that doesn't touch the image border (margin 5 px)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if x > 5 and y > 5 and (x + w) < w_img - 5 and (y + h) < h_img - 5:
            selected = (x, y, w, h)
            break
    # if none found, fall back to largest
    if selected is None:
        x, y, w, h = cv2.boundingRect(contours[0])
    else:
        x, y, w, h = selected
    
    # Normalize to [0, 1]
    bbox = (x / w_img, y / h_img, (x + w) / w_img, (y + h) / h_img)
    return bbox, (x, y, w, h)


def main():
    # Get 10 random tumor images
    tumor_files = sorted(glob.glob(os.path.join(TUMOUR_DIR, "*.npy")))
    sample_files = random.sample(tumor_files, min(10, len(tumor_files)))
    
    print(f"Testing auto-boxing on {len(sample_files)} samples...\n")
    
    results = []
    
    for idx, path in enumerate(sample_files):
        img = np.load(path)
        
        # Ensure 2D
        if img.ndim == 3:
            img = img[..., 0]
        
        # Resize to standard size
        img_resized = cv2.resize(img, (224, 224))
        
        # Auto-generate box
        result = auto_box_image(img_resized.copy())
        
        if result:
            bbox_norm, (x, y, w, h) = result
            results.append((os.path.basename(path), bbox_norm))
            status = "✓"
        else:
            results.append((os.path.basename(path), None))
            status = "✗"
        
        # draw and save visualization (normalize so structures are visible)
        vis = img_resized.copy()
        cv2.normalize(vis, vis, 0, 255, cv2.NORM_MINMAX)
        vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_GRAY2BGR)
        if result:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
        outname = os.path.join(DATA_DIR, f"sample_{idx+1}.png")
        cv2.imwrite(outname, vis)

        print(f"[{idx+1}/10] {status} {os.path.basename(path):<40}", end="")
        if result:
            bbox_norm = result[0]
            print(f" Box: ({bbox_norm[0]:.2f}, {bbox_norm[1]:.2f}, {bbox_norm[2]:.2f}, {bbox_norm[3]:.2f})")
        else:
            print(" No box detected")
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY:")
    print("="*80)
    
    for filename, bbox in results:
        if bbox:
            print(f"✓ {filename:<50} Box: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})")
        else:
            print(f"✗ {filename:<50} No box detected")
    
    success_count = sum(1 for _, bbox in results if bbox is not None)
    print("="*80)
    print(f"\nSuccess rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    
    if success_count >= 7:
        print("✓ AUTO-BOXING LOOKS GOOD! Ready to apply to all 2,283 images.")
        return True
    else:
        print("✗ Success rate too low. May need adjustment.")
        return False


if __name__ == "__main__":
    main()
