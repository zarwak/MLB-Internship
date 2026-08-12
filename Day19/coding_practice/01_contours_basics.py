"""
CONTOURS - the basics. A contour is just a curve joining all the continuous
points along a boundary that share the same color/intensity - in practice,
the outline of a blob of white pixels in a binary (black/white) image.

FINDING contours needs a binary image first, so the pipeline is:
  grayscale -> threshold (turn it black/white) -> cv2.findContours

cv2.findContours returns a LIST of contours, where each contour is just an
array of (x, y) points tracing that shape's outline. cv2.drawContours then
draws any of them back onto an image - here we draw ALL of them at once by
passing -1 as the contour index.
"""
import cv2
from utils import get_sample_image, save_and_show, build_collage

img = get_sample_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold: turn the image into pure black/white so findContours has clean
# blobs to trace. THRESH_BINARY_INV so shapes (usually darker than a light
# background) become the WHITE foreground that findContours looks for.
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# RETR_EXTERNAL = only outer boundaries (ignore holes inside shapes)
# CHAIN_APPROX_SIMPLE = compress straight-line segments down to their endpoints
#                        (a rectangle's outline becomes 4 points, not hundreds)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} contour(s)")

all_contours = img.copy()
cv2.drawContours(all_contours, contours, -1, (0, 255, 0), 2)  # -1 = draw every contour

save_and_show("01_threshold.jpg", thresh)
save_and_show("01_all_contours.jpg", all_contours)
save_and_show(
    "01_compare.jpg",
    build_collage([img, thresh, all_contours], ["original", "threshold", "contours drawn"]),
)
