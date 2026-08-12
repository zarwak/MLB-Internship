"""
SHAPE CLASSIFICATION - telling a triangle from a square from a circle, using
only the contour, boils down to counting corners:

  cv2.approxPolyDP(contour, epsilon, True)

This simplifies a contour down to its key corner points, dropping points
that don't meaningfully change its shape. `epsilon` controls how aggressive
the simplification is - we use 2% of the contour's own perimeter, so it
scales with each shape's size instead of using one fixed pixel value.

Then it's just counting how many corners came out:
  3 corners            -> Triangle
  4 corners             -> Square or Rectangle (tell them apart by aspect ratio)
  5+ corners, NOT round -> Polygon (labeled with its corner count)
  otherwise, round      -> Circle (circles approximate to many small corners,
                            so they're identified by CIRCULARITY instead:
                            4*pi*area / perimeter^2, which is 1.0 for a
                            perfect circle and lower for anything elongated
                            or angular)
"""
import cv2
from utils import get_sample_image, save_and_show

MIN_AREA = 150


def classify_shape(c):
    perimeter = cv2.arcLength(c, True)
    area = cv2.contourArea(c)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    corners = len(approx)

    circularity = 4 * 3.14159 * area / (perimeter * perimeter) if perimeter else 0

    if corners == 3:
        return "Triangle"
    if corners == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.90 <= aspect_ratio <= 1.10 else "Rectangle"
    if circularity > 0.80:
        return "Circle"
    return f"Polygon ({corners} sides)"


img = get_sample_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

annotated = img.copy()
for c in contours:
    if cv2.contourArea(c) < MIN_AREA:
        continue
    shape_name = classify_shape(c)

    cv2.drawContours(annotated, [c], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(c)
    cv2.putText(annotated, shape_name, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    print(f"{shape_name} at ({x},{y})")

save_and_show("04_shape_classification.jpg", annotated)
