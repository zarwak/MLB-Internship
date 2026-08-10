"""
TRANSLATION = sliding an image left/right/up/down without rotating or resizing it.
Real use: shifting/aligning images, data augmentation for training AI models.
"""
import cv2
import numpy as np
from utils import get_sample_image, save_and_show

img = get_sample_image()
h, w = img.shape[:2]  # image height and width

tx, ty = 80, 40  # move 80 px right, 40 px down (use negative numbers to go left/up)

# translation matrix: row1 moves x by tx, row2 moves y by ty
M = np.float32([[1, 0, tx],
                 [0, 1, ty]])

shifted = cv2.warpAffine(img, M, (w, h))  # apply the shift

save_and_show("translation_original.jpg", img)
save_and_show("translation_shifted.jpg", shifted)
