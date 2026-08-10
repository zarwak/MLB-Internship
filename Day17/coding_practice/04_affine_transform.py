"""
AFFINE TRANSFORM = move/rotate/scale/skew an image, but straight lines stay
straight and parallel lines stay parallel.
Real use: fixing small skew, warping shapes, image augmentation.
It needs exactly 3 points (before -> after) to build the transform.
"""
import cv2
import numpy as np
from utils import get_sample_image, save_and_show

img = get_sample_image()
h, w = img.shape[:2]

pts1 = np.float32([[50, 50], [w - 50, 50], [50, h - 50]])       # 3 points on the original
pts2 = np.float32([[10, 100], [w - 100, 50], [100, h - 100]])   # where those 3 points should move to

M = cv2.getAffineTransform(pts1, pts2)   # build the transform matrix from the 3 point pairs
affine = cv2.warpAffine(img, M, (w, h))  # apply it to the whole image

save_and_show("affine_original.jpg", img)
save_and_show("affine_result.jpg", affine)
