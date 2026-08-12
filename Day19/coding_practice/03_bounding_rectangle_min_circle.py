"""
BOUNDING RECTANGLE & MINIMUM ENCLOSING CIRCLE - two ways to summarize a
contour's rough size/position with a simple shape:

  - cv2.boundingRect(c)         -> the smallest UPRIGHT rectangle (x, y, w, h)
                                    that fully contains the contour. Cheap,
                                    great for cropping and rough positioning.
  - cv2.minEnclosingCircle(c)   -> the smallest circle (center, radius) that
                                    fully contains the contour. Useful when a
                                    shape's orientation doesn't matter, only
                                    its extent from a center point.

Both only APPROXIMATE the contour's real shape - a diagonal rectangle still
gets an upright bounding box bigger than itself, and a square still gets a
circle bigger than itself. They're for quick size/position estimates, not
identifying what the shape actually is (that's the next script).
"""
import cv2
from utils import get_sample_image, save_and_show

MIN_AREA = 150

img = get_sample_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes_img = img.copy()
circles_img = img.copy()
for c in contours:
    if cv2.contourArea(c) < MIN_AREA:
        continue

    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(boxes_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    (cx, cy), radius = cv2.minEnclosingCircle(c)
    cv2.circle(circles_img, (int(cx), int(cy)), int(radius), (0, 140, 255), 2)
    cv2.circle(circles_img, (int(cx), int(cy)), 3, (0, 140, 255), -1)  # mark the center

save_and_show("03_bounding_rectangles.jpg", boxes_img)
save_and_show("03_min_enclosing_circles.jpg", circles_img)
