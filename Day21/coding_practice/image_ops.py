"""
Image processing operations for the Day 21 coding practice app.

Every function follows the same shape on purpose: (image_in, **params) -> image_out.
- image_in is always a BGR numpy array (what cv2.imread / an uploaded photo gives us).
- image_out is either a BGR array (3 channels) or a single-channel grayscale array.
Keeping that contract identical for every function is what lets app.py pick any
operation out of a dictionary and call it the same way, instead of a long
if/elif chain per operation.
"""

import cv2
import numpy as np


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse 3 color channels into 1 brightness value per pixel."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(image: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Average each pixel with its neighbors using a Gaussian-weighted kernel."""
    if ksize % 2 == 0:
        ksize += 1  # GaussianBlur requires an odd kernel size
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def apply_edge_detection(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Find pixels where brightness changes sharply (Canny edge detector)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # denoise first, or Canny finds fake edges
    return cv2.Canny(blurred, low, high)


def apply_rotation(image: np.ndarray, angle: float = 45.0) -> np.ndarray:
    """Rotate around the image center, expanding the canvas so corners aren't cropped."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Recompute the bounding box so a rotated image doesn't get clipped at the edges.
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, rotation_matrix, (new_w, new_h))


def apply_enhancement(image: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Sharpen via unsharp masking: original + amount * (original - blurred)."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def apply_contour_detection(image: np.ndarray, min_area: int = 200) -> np.ndarray:
    """Threshold the image, trace blob outlines, and draw every contour found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output


def _classify_shape(contour: np.ndarray) -> str:
    """Approximate a contour to a simpler polygon, then classify it by corner count."""
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
    """Find shapes via contours, then label each one with its detected name."""
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


# One dictionary any UI can loop over: dropdown labels -> the function to call.
# This is what replaces a long if/elif chain in app.py.
OPERATIONS = {
    "Grayscale": apply_grayscale,
    "Blur": apply_blur,
    "Edge Detection": apply_edge_detection,
    "Rotation": apply_rotation,
    "Image Enhancement (Sharpen)": apply_enhancement,
    "Contour Detection": apply_contour_detection,
    "Shape Detection": apply_shape_detection,
}
