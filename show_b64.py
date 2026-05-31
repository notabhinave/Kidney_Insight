import cv2, base64, sys

if len(sys.argv) < 2:
    print('usage: python show_b64.py <image>')
    sys.exit(1)

path = sys.argv[1]
img = cv2.imread(path)
if img is None:
    print('failed to read', path)
    sys.exit(1)

# resize for smaller output
img2 = cv2.resize(img, (112, 112))
_, buf = cv2.imencode('.jpg', img2, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
print(base64.b64encode(buf).decode())
