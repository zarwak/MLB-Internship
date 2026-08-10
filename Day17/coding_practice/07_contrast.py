"""
CONTRAST = making the gap between light and dark parts stronger or weaker.
Real use: making faded/washed-out documents easier to read, better OCR accuracy.
"""
import cv2
from utils import get_sample_image, save_and_show

img = get_sample_image()

# convertScaleAbs does: new_pixel = old_pixel * alpha + beta
more_contrast = cv2.convertScaleAbs(img, alpha=1.8, beta=0)  # alpha > 1 = stronger contrast
less_contrast = cv2.convertScaleAbs(img, alpha=0.6, beta=0)  # alpha < 1 = weaker (flatter) contrast

save_and_show("contrast_more.jpg", more_contrast)
save_and_show("contrast_less.jpg", less_contrast)
