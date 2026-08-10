"""
ROTATION = spinning an image around a center point by some angle.
Real use: fixing a sideways photo, straightening scans, data augmentation.
"""
import cv2
from utils import get_sample_image, save_and_show

img = get_sample_image()
h, w = img.shape[:2]
center = (w // 2, h // 2)  # rotate around the middle of the image

for angle in [45, 90, 180]:  # try a few different angles
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 1.0 means keep the same size (no zoom)
    rotated = cv2.warpAffine(img, M, (w, h))          # apply the rotation
    save_and_show(f"rotation_{angle}.jpg", rotated)
