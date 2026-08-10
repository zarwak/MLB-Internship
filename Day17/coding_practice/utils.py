"""
Helper functions shared by all practice scripts.
We don't have a real document dataset yet, so we fake one here.
(The mini project later will use real document photos instead.)
"""

import os
import cv2
import numpy as np

# folder this file lives in, so images/outputs always save HERE
# no matter which folder you were in when you ran "python 01_translation.py"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

DOC_W, DOC_H = 600, 800  # size of our fake "flat" document

# the 4 corners (top-left, top-right, bottom-right, bottom-left) where the
# document will appear to sit INSIDE the fake tilted photo
TILTED_POINTS = np.float32([
    [150, 120],   # top-left
    [750, 80],    # top-right
    [800, 750],   # bottom-right
    [100, 800],   # bottom-left
])


def _ensure_folders():
    # make sure images/ and outputs/ exist before we save anything
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def create_flat_document():
    # a plain white "page" with a border and a few lines of fake text
    doc = np.full((DOC_H, DOC_W, 3), 255, dtype=np.uint8)  # white background
    cv2.rectangle(doc, (10, 10), (DOC_W - 10, DOC_H - 10), (0, 0, 0), 3)  # black page border
    for i, y in enumerate(range(80, DOC_H - 60, 60)):
        text = f"This is line {i + 1} of the sample document."
        cv2.putText(doc, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return doc


def create_tilted_photo(doc):
    # pretends the document was photographed at an angle, lying on a table
    canvas = np.full((900, 900, 3), 160, dtype=np.uint8)  # gray "table" background

    src = np.float32([[0, 0], [DOC_W, 0], [DOC_W, DOC_H], [0, DOC_H]])  # doc's real corners
    M = cv2.getPerspectiveTransform(src, TILTED_POINTS)  # map real corners -> tilted spot
    warped_doc = cv2.warpPerspective(doc, M, (900, 900))  # the tilted-looking document

    # figure out WHERE the document landed (a plain white shape warped the same way)
    white_doc = np.full((DOC_H, DOC_W, 3), 255, dtype=np.uint8)
    warped_shape = cv2.warpPerspective(white_doc, M, (900, 900))
    doc_area = cv2.cvtColor(warped_shape, cv2.COLOR_BGR2GRAY) > 0

    canvas[doc_area] = warped_doc[doc_area]  # paste the tilted document onto the table
    return canvas


def get_sample_image():
    # returns the flat sample document (creates + saves it once if missing)
    _ensure_folders()
    path = os.path.join(IMAGES_DIR, "sample.jpg")
    if not os.path.exists(path):
        cv2.imwrite(path, create_flat_document())
    return cv2.imread(path)


def get_tilted_sample():
    # returns the fake tilted "photo" of the document (creates + saves it once if missing)
    _ensure_folders()
    path = os.path.join(IMAGES_DIR, "tilted_sample.jpg")
    if not os.path.exists(path):
        tilted = create_tilted_photo(create_flat_document())
        cv2.imwrite(path, tilted)
    return cv2.imread(path)


def add_noise(img):
    # sprinkles random black/white dots (salt & pepper noise) so blur/sharpen effects are visible
    noisy = img.copy()
    h, w = noisy.shape[:2]
    n = int(h * w * 0.02)  # noisy pixels = 2% of the image
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    noisy[ys, xs] = np.random.choice([0, 255], size=n)[:, None]  # set random pixels to black or white
    return noisy


def save_and_show(name, img):
    # saves the result into outputs/ AND shows it in a popup window
    _ensure_folders()
    path = os.path.join(OUTPUTS_DIR, name)
    cv2.imwrite(path, img)
    print(f"Saved: {path}")
    cv2.imshow(name, img)
    cv2.waitKey(0)          # wait for a key press
    cv2.destroyAllWindows()  # close the window before showing the next one
