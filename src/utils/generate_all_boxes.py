"""
Auto-generate bounding boxes for all tumor images using Otsu thresholding.
Output: bbox_dict.npy mapping filename -> normalized bbox (xmin, ymin, xmax, ymax)
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path

def auto_box_image(img_array):
    """
    Generate normalized bbox from kidney organ region using Otsu threshold.
    
    Args:
        img_array: 2D numpy array (224x224 grayscale)
    
    Returns:
        tuple: (bbox_norm, pixel_coords) or (None, None) if no contour found
               bbox_norm: (xmin, ymin, xmax, ymax) normalized to [0,1]
               pixel_coords: (x, y, w, h) in pixels
    """
    try:
        # Normalize to 0-255 range
        cv2.normalize(img_array, img_array, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_array.astype('uint8')
        
        # Otsu threshold
        _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        
        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Normalize to [0, 1]
        h_img, w_img = img_array.shape
        bbox_norm = (x / w_img, y / h_img, (x + w) / w_img, (y + h) / h_img)
        
        return bbox_norm, (x, y, w, h)
    except Exception as e:
        print(f"  Error processing image: {e}")
        return None, None


def main():
    tumor_dir = Path("processed_clahe/tumour")
    bbox_dict = {}
    
    # Get all tumor files
    tumor_files = sorted(glob.glob(str(tumor_dir / "*.npy")))
    total_files = len(tumor_files)
    
    print(f"Processing {total_files} tumor images...")
    print("=" * 80)
    
    successful = 0
    failed = 0
    
    for idx, fpath in enumerate(tumor_files, 1):
        fname = Path(fpath).name
        try:
            img_array = np.load(fpath)
            bbox_norm, pixel_coords = auto_box_image(img_array.copy())
            
            if bbox_norm is not None:
                bbox_dict[fname] = bbox_norm
                successful += 1
                if idx % 50 == 0 or idx == total_files:
                    print(f"[{idx:4d}/{total_files}] ✓ {fname:40s} Box: {bbox_norm}")
            else:
                failed += 1
                if idx % 100 == 0:
                    print(f"[{idx:4d}/{total_files}] ✗ {fname:40s} No contour found")
        except Exception as e:
            failed += 1
            if idx % 100 == 0:
                print(f"[{idx:4d}/{total_files}] ✗ {fname:40s} Error: {e}")
    
    # Save bbox dictionary
    output_path = Path("processed_clahe/bbox_dict.npy")
    np.save(output_path, bbox_dict)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print("=" * 80)
    print(f"Total files processed:   {total_files}")
    print(f"Successful boxes:        {successful} ({100*successful/total_files:.1f}%)")
    print(f"Failed (no contour):     {failed} ({100*failed/total_files:.1f}%)")
    print(f"Saved to:                {output_path}")
    print(f"Dict size:               {len(bbox_dict)} entries")
    print("=" * 80)


if __name__ == "__main__":
    main()
