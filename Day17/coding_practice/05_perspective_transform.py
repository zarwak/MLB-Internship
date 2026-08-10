"""
PERSPECTIVE TRANSFORM = straightens an image taken at an angle
(like a phone photo of a tilted paper on a table).
Real use: scanner apps, document straightening, prep before OCR.
It needs exactly 4 corner points (before -> after) to build the transform.
"""
import cv2
import numpy as np
from utils import get_tilted_sample, save_and_show, TILTED_POINTS, DOC_W, DOC_H

tilted_img = get_tilted_sample()  # our fake "photo of a tilted document on a table"

# the 4 corners of the document AS SEEN in the tilted photo (TL, TR, BR, BL)
# (normally you'd find these by clicking the corners or using edge detection)
src_points = TILTED_POINTS

# we want those same 4 corners to become a perfect flat rectangle
dst_points = np.float32([[0, 0], [DOC_W, 0], [DOC_W, DOC_H], [0, DOC_H]])

M = cv2.getPerspectiveTransform(src_points, dst_points)      # build the "straighten" matrix
straightened = cv2.warpPerspective(tilted_img, M, (DOC_W, DOC_H))  # apply it

save_and_show("perspective_tilted.jpg", tilted_img)
save_and_show("perspective_straightened.jpg", straightened)
