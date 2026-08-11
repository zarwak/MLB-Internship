"""
Helper functions shared by all Day 18 practice scripts.
We don't have real document photos yet (those are for the mini project), so
we generate one fake sample document here, similar to Day17.
"""

import os
import cv2
import numpy as np

# folder this file lives in, so images/outputs always save HERE
# no matter which folder you were in when you ran "python 01_grayscale_and_blur.py"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

DOC_W, DOC_H = 700, 900  # size of our fake sample document


def _ensure_folders():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def create_sample_document():
    # a plain white "page" with a border, some text lines, and a couple of
    # shapes - gives edge detection and morphology something varied to work on
    doc = np.full((DOC_H, DOC_W, 3), 255, dtype=np.uint8)  # white background
    cv2.rectangle(doc, (15, 15), (DOC_W - 15, DOC_H - 15), (0, 0, 0), 3)  # page border

    cv2.putText(doc, "SAMPLE DOCUMENT", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    for i, y in enumerate(range(130, 500, 45)):
        text = f"This is line {i + 1} of the sample document body text."
        cv2.putText(doc, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # a couple of shapes so edge detection has more than just straight text lines
    cv2.rectangle(doc, (60, 560), (300, 700), (0, 0, 0), 2)
    cv2.circle(doc, (500, 630), 70, (0, 0, 0), 2)
    cv2.line(doc, (60, 760), (DOC_W - 60, 760), (0, 0, 0), 2)

    return doc


VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def get_sample_image():
    # uses whatever real photo you've dropped into images/ - only falls
    # back to generating a fake sample document if that folder is empty
    _ensure_folders()
    existing = sorted(f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(VALID_EXT))
    if existing:
        return cv2.imread(os.path.join(IMAGES_DIR, existing[0]))

    path = os.path.join(IMAGES_DIR, "sample_document.jpg")
    cv2.imwrite(path, create_sample_document())
    return cv2.imread(path)


def add_noise(img):
    # sprinkles random black/white dots (salt & pepper noise), useful for
    # showing what morphological opening/closing cleans up
    noisy = img.copy()
    h, w = noisy.shape[:2]
    n = int(h * w * 0.02)  # noisy pixels = 2% of the image
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    values = np.random.choice([0, 255], size=n)
    if noisy.ndim == 2:  # grayscale/binary - one value per pixel
        noisy[ys, xs] = values
    else:  # color - same value on every channel, so the dot is pure black/white
        noisy[ys, xs] = values[:, None]
    return noisy


def save_and_show(name, img):
    # saves the result into outputs/ AND shows it in a popup window
    _ensure_folders()
    path = os.path.join(OUTPUTS_DIR, name)
    cv2.imwrite(path, img)
    print(f"Saved: {path}")
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def build_collage(images, labels, label_height=30):
    # lays images side by side with a text label above each one, so
    # comparisons (Sobel vs Laplacian vs Canny, before vs after, etc.) are
    # easy to see in a single saved file
    normalized = []
    for img in images:
        if img.ndim == 2:  # grayscale -> BGR so everything can sit in one collage
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        normalized.append(img)

    height = max(img.shape[0] for img in normalized)
    panels = []
    for img, label in zip(normalized, labels):
        h, w = img.shape[:2]
        if h != height:  # pad shorter images so all panels line up
            pad = np.full((height - h, w, 3), 255, dtype=np.uint8)
            img = np.vstack([img, pad])

        bar = np.full((label_height, w, 3), 30, dtype=np.uint8)
        cv2.putText(bar, label, (8, label_height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(np.vstack([bar, img]))

        divider = np.full((height + label_height, 4, 3), 200, dtype=np.uint8)
        panels.append(divider)

    return np.hstack(panels[:-1])  # drop trailing divider
