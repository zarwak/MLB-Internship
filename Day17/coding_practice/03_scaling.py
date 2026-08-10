"""
SCALING = making an image bigger or smaller.
Real use: resizing images to fit a model's input size, thumbnails, zoom.
"""
import cv2
from utils import get_sample_image, save_and_show

img = get_sample_image()

# fx, fy = scale factor for width, height. 1.5 = 150% size, 0.5 = 50% size
bigger = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)  # smooth, good for enlarging
smaller = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)   # best quality for shrinking

save_and_show("scaling_bigger.jpg", bigger)
save_and_show("scaling_smaller.jpg", smaller)
