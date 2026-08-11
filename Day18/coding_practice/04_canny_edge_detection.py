"""
CANNY EDGE DETECTION - the industry-standard edge detector. It's really
Sobel plus extra smarts: it thins edges down to a single clean line, then
uses TWO thresholds to decide what counts as a real edge:
  - Below the low threshold: definitely not an edge, thrown away.
  - Above the high threshold: definitely an edge, kept.
  - In between: kept ONLY if it's connected to a definite edge.
This means Canny keeps faint-but-real edges (like a shadow's boundary) as
long as they connect to a strong edge, instead of losing them completely.

CHOOSING THRESHOLDS: pick them too low and you get noise everywhere; too
high and you lose real edges. Rather than guessing fixed numbers by hand,
"auto canny" picks thresholds based on the image's own median brightness -
that way it adapts to each photo instead of using one hardcoded pair.
"""
import cv2
import numpy as np
from utils import get_sample_image, save_and_show, build_collage


def auto_canny(gray, sigma=0.33):
    median = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


img = get_sample_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

canny_tight = cv2.Canny(blurred, 150, 200)   # both thresholds high -> only strong edges survive
canny_loose = cv2.Canny(blurred, 10, 50)     # both thresholds low -> lots of edges, some noise
canny_auto = auto_canny(blurred)             # thresholds derived from the image itself

save_and_show("04_canny_tight.jpg", canny_tight)
save_and_show("04_canny_loose.jpg", canny_loose)
save_and_show("04_canny_auto.jpg", canny_auto)
save_and_show(
    "04_compare.jpg",
    build_collage(
        [blurred, canny_tight, canny_loose, canny_auto],
        ["blurred", "tight (150,200)", "loose (10,50)", "auto (median-based)"],
    ),
)
