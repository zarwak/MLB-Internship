"""
COMPARE ALL THREE - Sobel vs Laplacian vs Canny on the same image, so the
difference is visible side by side instead of guessed from memory.
"""
import cv2
from utils import get_sample_image, save_and_show, build_collage

img = get_sample_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))

laplacian = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F, ksize=3))

canny = cv2.Canny(blurred, 60, 160)

save_and_show(
    "05_compare_all.jpg",
    build_collage([blurred, sobel, laplacian, canny], ["blurred", "sobel", "laplacian", "canny"]),
)

# WHAT TO NOTICE:
# - Sobel: edges look "thick" and a bit soft/gray - it's a raw gradient, not thinned.
# - Laplacian: picks up edges in every direction at once, but is the noisiest
#   of the three - fine details and slight texture get exaggerated.
# - Canny: cleanest result - edges are thin, connected, single-pixel lines.
#   This is why Canny is the one used in the document boundary tool below.
