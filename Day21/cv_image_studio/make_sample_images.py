"""Generates sample_input_images/ for the Challenge Task app. Run once:
python make_sample_images.py"""

import os

import cv2
import numpy as np

# Anchored to this file's own folder, not the terminal's current directory -
# otherwise running this script while cd'd into a sibling project folder
# (e.g. cv_image_studio, which uses the same folder name) would write
# these images there instead of here.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "sample_input_images")


def make_shapes_image() -> np.ndarray:
    img = np.full((400, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (150, 150), (200, 100, 50), -1)
    cv2.rectangle(img, (200, 40), (400, 120), (50, 150, 200), -1)
    cv2.circle(img, (100, 280), 70, (50, 200, 100), -1)
    triangle = np.array([[300, 350], [250, 220], [400, 220]], dtype=np.int32)
    cv2.fillPoly(img, [triangle], (150, 50, 200))
    return img


def make_gradient_photo() -> np.ndarray:
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
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_dark_photo() -> np.ndarray:
    """A deliberately dark, low-contrast image - good for demonstrating
    why Brightness & Contrast is a genuinely useful addition."""
    rng = np.random.default_rng(7)
    base = rng.integers(20, 70, (400, 500, 3), dtype=np.uint8)
    cv2.circle(base, (250, 200), 100, (90, 60, 40), -1)
    return base


if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(INPUT_DIR, "shapes.png"), make_shapes_image())
    cv2.imwrite(os.path.join(INPUT_DIR, "gradient_photo.png"), make_gradient_photo())
    cv2.imwrite(os.path.join(INPUT_DIR, "dark_photo.png"), make_dark_photo())
    print(f"Wrote 3 sample images to {INPUT_DIR}")
