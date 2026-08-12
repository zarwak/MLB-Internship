"""
CONTOUR AREA & PERIMETER - once you have a contour, two of the most useful
numbers you can pull from it are:
  - cv2.contourArea(c)         -> the area enclosed by the contour, in pixels
  - cv2.arcLength(c, True)     -> the perimeter (True = the contour is closed)

These feed almost everything else: filtering out tiny noise contours,
telling shapes apart (a circle and a square of similar area have different
area-to-perimeter ratios), and reporting size to a user.
"""
import cv2
from utils import get_sample_image, save_and_show

MIN_AREA = 150  # ignore tiny specks/noise that aren't real shapes

img = get_sample_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

annotated = img.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area < MIN_AREA:
        continue
    perimeter = cv2.arcLength(c, True)

    cv2.drawContours(annotated, [c], -1, (0, 255, 0), 2)
    # label near the top of the shape so text doesn't sit on top of it
    x, y, w, h = cv2.boundingRect(c)
    label = f"A={area:.0f} P={perimeter:.0f}"
    cv2.putText(annotated, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    print(f"Contour at ({x},{y}): area={area:.1f}, perimeter={perimeter:.1f}")

save_and_show("02_area_perimeter.jpg", annotated)
