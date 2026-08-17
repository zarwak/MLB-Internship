"""
Extended operations for the Challenge Task: the 7 base operations plus 3
custom ones not covered in class (Brightness & Contrast, Flip,
Thresholding), and a PARAM_SPECS dict that now supports two kinds of
controls - "slider" (a numeric range) and "choice" (a dropdown of named
options) - so app.py can build the right widget for either kind without
hardcoding per operation.
"""

import cv2
import numpy as np


# ---- Base 7 (same as coding_practice / cv_image_studio) -------------------

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(image: np.ndarray, ksize: int = 15) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def apply_edge_detection(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low, high)


def apply_rotation(image: np.ndarray, angle: float = 45.0) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, rotation_matrix, (new_w, new_h))


def apply_enhancement(image: np.ndarray, amount: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def apply_contour_detection(image: np.ndarray, min_area: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output


def _classify_shape(contour: np.ndarray) -> str:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    corners = len(approx)
    if corners == 3:
        return "Triangle"
    if corners == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.90 <= aspect_ratio <= 1.10 else "Rectangle"
    if corners > 4:
        area = cv2.contourArea(contour)
        radius = perimeter / (2 * np.pi)
        circularity = area / (np.pi * radius * radius) if radius > 0 else 0
        return "Circle" if 0.80 <= circularity <= 1.20 else "Polygon"
    return "Unknown"


def apply_shape_detection(image: np.ndarray, min_area: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        shape_name = _classify_shape(contour)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
        cv2.putText(output, shape_name, (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return output


# ---- Custom additions (not covered in class) -------------------------------

def apply_brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
    """new_pixel = contrast * pixel + brightness, clipped to 0-255.
    contrast > 1 stretches the range apart (more contrast); brightness shifts
    every pixel up or down. cv2.convertScaleAbs does the multiply-add-clip in one call."""
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def apply_flip(image: np.ndarray, mode: str = "Horizontal") -> np.ndarray:
    """Pure index reversal - no math, cv2.flip just reads the array backwards
    along an axis. 1 = left-right, 0 = top-bottom, -1 = both."""
    flip_code = {"Horizontal": 1, "Vertical": 0, "Both": -1}[mode]
    return cv2.flip(image, flip_code)


def apply_threshold(image: np.ndarray, thresh_value: int = 127, method: str = "Binary") -> np.ndarray:
    """Turn grayscale into pure black/white - the simplest possible segmentation.
    'Otsu (auto)' picks the cutoff from the image's own brightness histogram
    instead of a hardcoded value (same idea as Day19's shape detector)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    if method == "Binary":
        _, out = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    elif method == "Binary Inverted":
        _, out = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
    elif method == "Otsu (auto)":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    return out


OPERATIONS = {
    "Grayscale": apply_grayscale,
    "Blur": apply_blur,
    "Edge Detection": apply_edge_detection,
    "Rotation": apply_rotation,
    "Image Enhancement (Sharpen)": apply_enhancement,
    "Contour Detection": apply_contour_detection,
    "Shape Detection": apply_shape_detection,
    "Brightness & Contrast": apply_brightness_contrast,
    "Flip": apply_flip,
    "Threshold": apply_threshold,
}

# type "slider" (default) -> a numeric range widget.
# type "choice" -> a dropdown of named string options.
PARAM_SPECS = {
    "Grayscale": {},
    "Blur": {
        "ksize": {"label": "Kernel size", "min": 1, "max": 51, "default": 15, "step": 2},
    },
    "Edge Detection": {
        "low": {"label": "Lower threshold", "min": 0, "max": 255, "default": 50, "step": 1},
        "high": {"label": "Upper threshold", "min": 0, "max": 255, "default": 150, "step": 1},
    },
    "Rotation": {
        "angle": {"label": "Angle (degrees)", "min": -180, "max": 180, "default": 45, "step": 1},
    },
    "Image Enhancement (Sharpen)": {
        "amount": {"label": "Sharpen amount", "min": 0.0, "max": 3.0, "default": 1.5, "step": 0.1},
    },
    "Contour Detection": {
        "min_area": {"label": "Minimum contour area", "min": 0, "max": 5000, "default": 200, "step": 50},
    },
    "Shape Detection": {
        "min_area": {"label": "Minimum shape area", "min": 0, "max": 5000, "default": 200, "step": 50},
    },
    "Brightness & Contrast": {
        "brightness": {"label": "Brightness", "min": -100, "max": 100, "default": 0, "step": 1},
        "contrast": {"label": "Contrast", "min": 0.0, "max": 3.0, "default": 1.0, "step": 0.1},
    },
    "Flip": {
        "mode": {"type": "choice", "label": "Direction", "options": ["Horizontal", "Vertical", "Both"], "default": "Horizontal"},
    },
    "Threshold": {
        "method": {"type": "choice", "label": "Method", "options": ["Binary", "Binary Inverted", "Otsu (auto)"], "default": "Binary"},
        "thresh_value": {"label": "Threshold value (ignored for Otsu)", "min": 0, "max": 255, "default": 127, "step": 1},
    },
}
