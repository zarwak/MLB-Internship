"""
MORPHOLOGICAL OPERATIONS - shape-based cleanup for BINARY (black/white)
images. They all slide a small "kernel" (a tiny window, e.g. 3x3) across the
image and grow or shrink the white regions based on what's under it.
Real use: edge/threshold maps from real photos are messy (broken lines from
shadows, tiny noise specks) - these operations clean that up before we try
to find shapes (like a document's boundary) in the image.

  - Erosion:    shrinks white regions. Removes small white noise specks, but
                also shrinks/breaks up real shapes if overused.
  - Dilation:   grows white regions. Fills small black gaps/holes, useful for
                reconnecting a broken edge line.
  - Opening:    erosion THEN dilation. Removes small white noise specks
                without shrinking the main shape (the shape "recovers" in
                the dilation step, the noise doesn't).
  - Closing:    dilation THEN erosion. Fills small black holes/gaps inside
                or between white regions, without growing the main shape.
  - Gradient:   dilation minus erosion. Leaves just the OUTLINE of each
                white region - a cheap way to get an edge map from a shape.
  - Top Hat:    original minus opening. Highlights small BRIGHT details that
                are smaller than the kernel (and get erased by opening).
  - Black Hat:  closing minus original. Highlights small DARK gaps/details
                that are smaller than the kernel (and get filled by closing).
"""
import cv2
import numpy as np
from utils import get_sample_image, add_noise, save_and_show, build_collage

img = get_sample_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# make a binary (black/white) image: text/lines/border become WHITE on a
# BLACK background - the standard input shape morphology expects
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
noisy_binary = add_noise(binary)  # sprinkle in some fake noise specks to clean up

kernel = np.ones((3, 3), np.uint8)
big_kernel = np.ones((9, 9), np.uint8)  # top hat/black hat need a bigger kernel to show anything

erosion = cv2.erode(noisy_binary, kernel, iterations=1)
dilation = cv2.dilate(noisy_binary, kernel, iterations=1)
opening = cv2.morphologyEx(noisy_binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(noisy_binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(noisy_binary, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(noisy_binary, cv2.MORPH_TOPHAT, big_kernel)
blackhat = cv2.morphologyEx(noisy_binary, cv2.MORPH_BLACKHAT, big_kernel)

save_and_show("06_0_noisy_binary.jpg", noisy_binary)
save_and_show("06_1_erosion.jpg", erosion)
save_and_show("06_2_dilation.jpg", dilation)
save_and_show("06_3_opening.jpg", opening)
save_and_show("06_4_closing.jpg", closing)
save_and_show("06_5_gradient.jpg", gradient)
save_and_show("06_6_tophat.jpg", tophat)
save_and_show("06_7_blackhat.jpg", blackhat)

# before vs after, for the two most commonly used for noise cleanup
save_and_show(
    "06_before_after_opening.jpg",
    build_collage([noisy_binary, opening], ["before (noisy)", "after opening"]),
)
save_and_show(
    "06_before_after_closing.jpg",
    build_collage([noisy_binary, closing], ["before (noisy)", "after closing"]),
)
save_and_show(
    "06_compare_all.jpg",
    build_collage(
        [noisy_binary, erosion, dilation, opening, closing, gradient, tophat, blackhat],
        ["noisy", "erosion", "dilation", "opening", "closing", "gradient", "tophat", "blackhat"],
    ),
)
