"""
BRIGHTNESS = making the whole image lighter or darker.
Real use: fixing dark/underexposed photos, matching lighting between images.
"""
import cv2
from utils import get_sample_image, save_and_show

img = get_sample_image()

# convertScaleAbs does: new_pixel = old_pixel * alpha + beta
brighter = cv2.convertScaleAbs(img, alpha=1.0, beta=60)   # beta > 0 adds light
darker = cv2.convertScaleAbs(img, alpha=1.0, beta=-60)    # beta < 0 removes light

save_and_show("brightness_up.jpg", brighter)
save_and_show("brightness_down.jpg", darker)
