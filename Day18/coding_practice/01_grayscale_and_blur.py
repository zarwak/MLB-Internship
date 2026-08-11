"""
STEP 0 - grayscale + blur, the setup every edge detector below needs.
- Grayscale: edge detectors look at brightness changes, not color, so color
  is just extra data that slows things down and isn't needed.
- Gaussian Blur: real photos have tiny random noise (grain, JPEG artifacts).
  Without blurring first, an edge detector treats that noise as thousands of
  fake little edges. Blurring smooths it out so only real edges remain.
"""
import cv2
from utils import get_sample_image, save_and_show, build_collage

img = get_sample_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

save_and_show("01_original.jpg", img)
save_and_show("01_gray.jpg", gray)
save_and_show("01_blurred.jpg", blurred)
save_and_show("01_compare.jpg", build_collage([img, gray, blurred], ["original", "gray", "blurred"]))
