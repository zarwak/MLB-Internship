"""
Image Processing Toolkit
========================
A pure OpenCV image processing toolkit with helper functions.
This module no longer contains Streamlit UI code.
"""

import cv2
import numpy as np
import os

SAMPLE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "sample_images"))
OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "output_images"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sample_images():
    """Return a sorted list of sample image filenames."""
    if not os.path.exists(SAMPLE_DIR):
        return []
    return sorted([
        f for f in os.listdir(SAMPLE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])


def get_sample_image_path(filename):
    """Return the full path for a sample image filename."""
    return os.path.join(SAMPLE_DIR, filename)


def read_image(path):
    """Read an image from a file path and return it as a NumPy array (BGR)."""
    return cv2.imread(path)


def image_to_rgb(img):
    """Convert a BGR image to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def hex_to_bgr(hex_color):
    """Convert a hex color string (e.g. '#FF0000') to a BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def save_image(img, filename):
    """Save an image to the output directory."""
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, img)
    return save_path


def op_grayscale(img):
    """Convert an image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def op_resize(img, width, height):
    """Resize an image to the given dimensions."""
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def op_rotate(img, angle):
    """Rotate an image by a given angle in degrees."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))


def op_flip(img, direction):
    """Flip an image horizontally, vertically, or both."""
    flip_map = {"Horizontal": 1, "Vertical": 0, "Both": -1}
    return cv2.flip(img, flip_map[direction])


def op_crop(img, x, y, width, height):
    """Crop a region from the image."""
    h, w = img.shape[:2]
    x2 = min(x + width, w)
    y2 = min(y + height, h)
    return img[y:y2, x:x2]


def op_brightness_contrast(img, brightness, contrast):
    """Adjust brightness and contrast of an image."""
    result = img.copy().astype(np.float32)
    result = result * (contrast / 100.0 + 1.0)
    result = result + brightness
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def op_bgr_to_rgb(img):
    """Convert a BGR image to RGB format."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def op_draw_shapes(img, shapes):
    """Draw shapes on the image.

    shapes: list of dicts with 'type' and parameters.
    """
    canvas = img.copy()
    for shape in shapes:
        if shape["type"] == "Rectangle":
            cv2.rectangle(canvas, shape["pt1"], shape["pt2"], shape["color"], shape["thickness"])
        elif shape["type"] == "Circle":
            cv2.circle(canvas, shape["center"], shape["radius"], shape["color"], shape["thickness"])
        elif shape["type"] == "Line":
            cv2.line(canvas, shape["pt1"], shape["pt2"], shape["color"], shape["thickness"])
        elif shape["type"] == "Polygon":
            pts = np.array(shape["points"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], shape.get("is_closed", True), shape["color"], shape["thickness"])
    return canvas


def op_add_text(img, text, position, font_scale, color, thickness):
    """Add custom text to the image."""
    canvas = img.copy()
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return canvas


if __name__ == "__main__":
    print("toolkit.image_toolkit is now a pure OpenCV helper module.")
