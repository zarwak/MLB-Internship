"""
SHARPENING = makes edges and text look crisper and clearer.
Real use: making a slightly blurry document photo more readable, better OCR.
"""
import cv2
import numpy as np
from utils import get_sample_image, save_and_show

img = get_sample_image()
blurred = cv2.GaussianBlur(img, (5, 5), 0)  # blur it first, so we can clearly see sharpening fix it

# sharpening kernel: boosts the center pixel, subtracts its neighbors -> pops out edges
kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
sharpened = cv2.filter2D(blurred, -1, kernel)  # slide this kernel over every pixel

save_and_show("sharpen_before.jpg", blurred)
save_and_show("sharpen_after.jpg", sharpened)
