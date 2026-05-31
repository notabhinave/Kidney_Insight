import cv2
import numpy as np
import glob

def find_colored_boxes(img_path):
    img = cv2.imread(img_path)
    if img is None: return []
    
    # Try to find red or green or blue colored regions (lines)
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Let's just find the bounding box of non-grayscale pixels
    # Calculate difference between channels to find color
    b, g, r = cv2.split(img)
    diff1 = cv2.absdiff(r, g)
    diff2 = cv2.absdiff(g, b)
    diff3 = cv2.absdiff(b, r)
    color_mask = cv2.bitwise_or(cv2.bitwise_or(diff1, diff2), diff3)
    
    _, thresh = cv2.threshold(color_mask, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:
            boxes.append((x, y, w, h))
    return boxes

for img_path in glob.glob("Training_CT/*.png"):
    boxes = find_colored_boxes(img_path)
    print(f"{img_path}: {boxes}")
