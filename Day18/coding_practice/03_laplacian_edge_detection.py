"""
LAPLACIAN OPERATOR - unlike Sobel (which checks ONE direction at a time),
Laplacian looks at how brightness changes in ALL directions at once, using
the rate of change of the rate of change (the "2nd derivative").
Result: it finds edges regardless of direction in a single pass, but it's
more sensitive to noise than Sobel, so blurring first matters even more.
"""
import cv2
from utils import get_sample_image, save_and_show, build_collage

img = get_sample_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)  # CV_64F keeps negative changes
laplacian_disp = cv2.convertScaleAbs(laplacian)  # back to a normal 0-255 image

save_and_show("03_laplacian.jpg", laplacian_disp)
save_and_show("03_compare.jpg", build_collage([blurred, laplacian_disp], ["blurred", "laplacian"]))
