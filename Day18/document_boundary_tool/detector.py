"""
Document Boundary Detection Tool - the core pipeline.
Order: load -> resize for speed -> grayscale -> blur -> auto-thresholded
       Canny edges -> morphological closing (bridges gaps from shadows) ->
       find the largest 4-corner contour -> draw it on the original image.
Each step is its own small function, so you can test/understand one at a time.
"""
import cv2
import numpy as np

MAX_DIM = 1000  # working resolution - big phone photos get resized down to this for speed


def load_image(path):
    return cv2.imread(path)


def resize_for_processing(img, max_dim=MAX_DIM):
    """Shrinks big photos so edge detection runs fast and consistently.
    Returns (resized_img, scale) - scale is used later to map detected
    points back to the ORIGINAL full-resolution image."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale == 1.0:
        return img, 1.0
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess(img):
    """Grayscale + blur - see Day18/coding_practice/01_grayscale_and_blur.py for why."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred


def auto_canny(gray, sigma=0.33):
    """Canny thresholds derived from the image's own median brightness,
    instead of one hardcoded pair - adapts to each photo's lighting."""
    median = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def clean_edges(edges):
    """Morphological closing (dilate then erode) bridges small gaps in the
    edge outline caused by shadows or uneven lighting, so the page outline
    forms one connected shape instead of several broken pieces."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)


def order_points(pts):
    """Sorts 4 points into a fixed order: top-left, top-right, bottom-right,
    bottom-left, so downstream code always knows which corner is which."""
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)  # x+y is smallest at top-left, biggest at bottom-right
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)  # x-y is smallest at top-right, biggest at bottom-left
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def find_document_contour(cleaned_edges, min_area_ratio=0.2):
    """Looks through the biggest shapes in the edge map for one that
    simplifies to a clean 4-corner shape (a page). Returns (points,
    found_four_corners). If no clean 4-corner shape is big enough, falls
    back to a rotated bounding box of the single largest shape instead."""
    contours, _ = cv2.findContours(cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, False

    image_area = cleaned_edges.shape[0] * cleaned_edges.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # check the 5 biggest shapes

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * image_area:
            continue
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_points(approx.reshape(4, 2).astype("float32")), True

    # fallback: no clean 4-corner shape found - use the largest contour's
    # rotated bounding box as a best-guess outline instead of failing
    largest = contours[0]
    if cv2.contourArea(largest) < min_area_ratio * image_area:
        return None, False
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    return order_points(box.astype("float32")), False


def draw_boundary(img, points, found_four_corners):
    """Draws the detected boundary on a copy of the original image.
    Green = a clean 4-corner page outline was found.
    Orange = no clean outline found, this is an approximate best guess."""
    annotated = img.copy()
    if points is None:
        return annotated
    color = (0, 200, 0) if found_four_corners else (0, 140, 255)
    pts = points.astype(int)
    cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=4)
    for (x, y) in pts:
        cv2.circle(annotated, (x, y), 8, color, -1)
    return annotated


def detect_boundary(img):
    """Runs the full pipeline on one already-loaded (BGR) image.
    Returns a dict with every intermediate step plus the final result, so
    callers (process.py, app.py, the Challenge Task) can show or save any
    stage without re-running the pipeline."""
    resized, scale = resize_for_processing(img)
    gray, blurred = preprocess(resized)
    edges = auto_canny(blurred)
    cleaned = clean_edges(edges)
    points_resized, found_four_corners = find_document_contour(cleaned)

    points_original = None
    if points_resized is not None:
        points_original = points_resized / scale  # map back to full-resolution coordinates

    annotated = draw_boundary(img, points_original, found_four_corners)

    return {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "cleaned_edges": cleaned,
        "contour_points": points_original,
        "found_four_corners": found_four_corners,
        "annotated": annotated,
    }
