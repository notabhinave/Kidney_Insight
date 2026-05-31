"""
Dump all tumour .npy files to a PNG directory without any overlays.
This lets the user open them in an image editor and draw bounding boxes manually.
"""
import os
import glob
import numpy as np
import cv2
from pathlib import Path

def main():
    data_root = Path(__file__).parents[2]
    tumour_dir = data_root / "processed_clahe" / "tumour"
    out_dir = data_root / "exported_images"
    out_dir.mkdir(exist_ok=True)

    files = sorted(glob.glob(str(tumour_dir / "*.npy")))
    total = len(files)
    print(f"Exporting {total} tumour images to {out_dir}")
    for idx, f in enumerate(files, 1):
        arr = np.load(f)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr_resized = cv2.resize(arr, (224, 224))
        cv2.normalize(arr_resized, arr_resized, 0, 255, cv2.NORM_MINMAX)
        outname = out_dir / (Path(f).stem + ".png")
        cv2.imwrite(str(outname), arr_resized.astype('uint8'))
        if idx % 200 == 0 or idx == total:
            print(f"[{idx}/{total}] saved {outname.name}")

    print("Done.")

if __name__ == '__main__':
    main()
