import cv2
import numpy as np


def segment_image(image, method, threshold=127, block_size=11, c_value=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "Binary":
        _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    elif method == "Adaptive":
        out = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value,
        )
    elif method == "Otsu":
        _, out = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    return out
