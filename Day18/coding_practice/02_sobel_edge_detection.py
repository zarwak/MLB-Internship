"""
SOBEL OPERATOR - finds edges by measuring how fast brightness changes.
It looks in two directions separately:
  - Sobel X: picks up VERTICAL edges (brightness changing left-to-right)
  - Sobel Y: picks up HORIZONTAL edges (brightness changing top-to-bottom)
Combining both (magnitude) gives edges in every direction.
Real use: cheap and fast, but sensitive to noise - usually needs a blur first.
"""
import cv2
import numpy as np
from utils import get_sample_image, save_and_show, build_collage

img = get_sample_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ksize=3 is the size of the small grid Sobel slides across the image.
# CV_64F (instead of 8-bit) so negative brightness changes aren't clipped to 0.
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # combine both directions

# convert back to a normal displayable 0-255 image
sobel_x_disp = cv2.convertScaleAbs(sobel_x)
sobel_y_disp = cv2.convertScaleAbs(sobel_y)
sobel_combined_disp = cv2.convertScaleAbs(sobel_combined)

save_and_show("02_sobel_x.jpg", sobel_x_disp)
save_and_show("02_sobel_y.jpg", sobel_y_disp)
save_and_show("02_sobel_combined.jpg", sobel_combined_disp)
save_and_show(
    "02_compare.jpg",
    build_collage(
        [blurred, sobel_x_disp, sobel_y_disp, sobel_combined_disp],
        ["blurred", "sobel x", "sobel y", "sobel combined"],
    ),
)
