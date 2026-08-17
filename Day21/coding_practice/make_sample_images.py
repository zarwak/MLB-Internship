"""
Generates synthetic sample images to test the app with, so you don't need a
real photo on hand. Run once: python make_sample_images.py
"""

import os

import cv2
import numpy as np

# Anchored to this file's own folder, not the terminal's current directory -
# otherwise running this script while cd'd into a sibling project folder
# would write these images there instead of here.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "sample_images")


def make_shapes_image() -> np.ndarray:
    """A white canvas with a triangle, square, rectangle, and circle - good
    for testing Contour Detection and Shape Detection."""
    img = np.full((400, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (150, 150), (200, 100, 50), -1)
    cv2.rectangle(img, (200, 40), (400, 120), (50, 150, 200), -1)
    cv2.circle(img, (100, 280), 70, (50, 200, 100), -1)
    triangle = np.array([[300, 350], [250, 220], [400, 220]], dtype=np.int32)
    cv2.fillPoly(img, [triangle], (150, 50, 200))
    return img


def make_photo_like_image() -> np.ndarray:
    """A noisy gradient with a few soft blobs - good for testing Blur, Edge
    Detection, Rotation, and Enhancement on something less clean than flat shapes."""
    h, w = 400, 500
    x = np.linspace(0, 255, w, dtype=np.uint8)
    gradient = np.tile(x, (h, 1))
    img = cv2.merge([gradient, gradient // 2 + 40, 255 - gradient])

    for center, radius, color in [((120, 100), 60, (255, 255, 255)),
                                   ((380, 300), 90, (30, 30, 30))]:
        overlay = img.copy()
        cv2.circle(overlay, center, radius, color, -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    noise = np.random.default_rng(42).normal(0, 12, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "shapes.png"), make_shapes_image())
    cv2.imwrite(os.path.join(OUTPUT_DIR, "photo_like.png"), make_photo_like_image())
    print(f"Wrote shapes.png and photo_like.png to {OUTPUT_DIR}")
