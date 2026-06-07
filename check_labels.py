import cv2
import os

img_path = "detection/data/raw/IMG_0445.JPG"
txt_path = "detection/data/annotations/IMG_0445.txt"

if not os.path.exists(img_path) or not os.path.exists(txt_path):
    print("Files not found")
    exit()

img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f"OpenCV loaded shape: {w}x{h}")

with open(txt_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 5:
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            print(f"BBox: cx={cx:.3f}, cy={cy:.3f} -> Pixel: {x1},{y1} to {x2},{y2}")

cv2.imwrite("check_IMG_0445.jpg", img)
print("Saved to check_IMG_0445.jpg")
