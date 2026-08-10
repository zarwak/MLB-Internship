"""
BLUR / NOISE-REMOVAL FILTERS - each one smooths an image, but in a different way.
- Gaussian Blur: general smoothing, softens the whole image evenly.
- Median Blur: best at removing random black/white dot noise ("salt & pepper").
- Bilateral Filter: smooths noise but tries to keep edges sharp (slower to run).
Real use: cleaning up a noisy scanned/photographed document before OCR.
"""
import cv2
from utils import get_sample_image, add_noise, save_and_show

img = get_sample_image()
noisy = add_noise(img)  # add fake noise so the cleanup effect is visible

gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)      # average of nearby pixels, closer ones count more
median = cv2.medianBlur(noisy, 5)                  # replaces each pixel with the middle value nearby
bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)  # smooths flat areas but keeps edges (like text) crisp

save_and_show("blur_0_noisy.jpg", noisy)
save_and_show("blur_1_gaussian.jpg", gaussian)
save_and_show("blur_2_median.jpg", median)
save_and_show("blur_3_bilateral.jpg", bilateral)
