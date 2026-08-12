"""
Shape Detection System - the core pipeline.
Order: load -> grayscale -> blur -> auto-polarity Otsu threshold -> find
       contours -> for each contour big enough to matter: classify its
       shape, measure area/perimeter, get its bounding rect -> draw + label
       everything on the original image.
Each step is its own small function, so you can test/understand one at a time.
"""
import cv2
import numpy as np

MIN_AREA_RATIO = 0.0015  # contours smaller than this fraction of the image are ignored as noise


def load_image(path):
    return cv2.imread(path)


def preprocess(img):
    """Grayscale + blur - see Day19/coding_practice/01_contours_basics.py for why."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred


def auto_threshold(blurred):
    """Otsu's method picks the black/white cutoff from the image's own
    brightness histogram instead of a hardcoded number, so it adapts across
    differently-lit photos. THRESH_BINARY_INV assumes shapes are darker than
    their background (the common case: ink/print on paper, shapes on a
    whiteboard). If that guess is wrong - light shapes on a dark background -
    the white "foreground" ends up being most of the frame (the background)
    instead of a few small blobs (the shapes), so we detect that and flip it."""
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    white_ratio = np.count_nonzero(thresh) / thresh.size
    if white_ratio > 0.5:
        thresh = cv2.bitwise_not(thresh)
    return thresh


def find_shape_contours(thresh, image_area):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_RATIO * image_area
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def classify_shape(c, perimeter, area):
    """Corner-counting via approxPolyDP tells triangles/squares/rectangles
    apart cleanly. Circles approximate to many small corners instead of a
    fixed number, so they're caught separately with CIRCULARITY
    (4*pi*area/perimeter^2, which is 1.0 for a perfect circle and drops for
    anything elongated or angular)."""
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    corners = len(approx)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0

    if corners == 3:
        return "Triangle"
    if corners == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.90 <= aspect_ratio <= 1.10 else "Rectangle"
    if circularity > 0.80:
        return "Circle"
    return f"Polygon ({corners} sides)"


def analyze_shapes(img, contours):
    """Measures + classifies every contour. Returns a list of dicts (one per
    shape) so callers can draw, tabulate, or export the results however they
    like without re-running the pipeline."""
    shapes = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        label = classify_shape(c, perimeter, area)
        x, y, w, h = cv2.boundingRect(c)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        shapes.append({
            "contour": c,
            "label": label,
            "area": area,
            "perimeter": perimeter,
            "bbox": (x, y, w, h),
            "min_enclosing_circle": ((cx, cy), radius),
        })
    return shapes


def draw_all_contours(img, contours):
    out = img.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def draw_labeled_shapes(img, shapes):
    """Final annotated image: contour outline + bounding rectangle + a
    label with the shape name, area, and perimeter for every detected shape."""
    out = img.copy()
    for shape in shapes:
        x, y, w, h = shape["bbox"]
        cv2.drawContours(out, [shape["contour"]], -1, (0, 255, 0), 2)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 140, 0), 1)

        text = f"{shape['label']} | A={shape['area']:.0f} P={shape['perimeter']:.0f}"
        text_y = max(15, y - 8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, text_y - th - 4), (x + tw + 4, text_y + 4), (255, 255, 255), -1)
        cv2.putText(out, text, (x + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def detect_shapes(img):
    """Runs the full pipeline on one already-loaded (BGR) image. Returns a
    dict with every intermediate step plus the shape list, so callers
    (process.py, app.py, the Challenge Task) can show or save any stage
    without re-running the pipeline."""
    gray, blurred = preprocess(img)
    thresh = auto_threshold(blurred)

    image_area = img.shape[0] * img.shape[1]
    contours = find_shape_contours(thresh, image_area)
    shapes = analyze_shapes(img, contours)

    return {
        "gray": gray,
        "blurred": blurred,
        "thresh": thresh,
        "contours": contours,
        "contours_drawn": draw_all_contours(img, contours),
        "shapes": shapes,
        "annotated": draw_labeled_shapes(img, shapes),
    }
