import cv2
import numpy as np
import sys
import base64

if len(sys.argv) < 2:
    print('Usage: python analyze_image.py <path> [--b64]')
    sys.exit(1)

path = sys.argv[1]
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
print('shape', img.shape, 'min', img.min(), 'max', img.max())

if '--b64' in sys.argv:
    encoded = base64.b64encode(open(path, 'rb').read()).decode('ascii')
    print('\nBASE64:\n' + encoded)
