"""
Helper functions shared by all Day 19 practice scripts.
We don't have the real shape dataset here yet (those photos go in
shape_detection_system/input_images/), so like Day17/Day18 we generate one
fake sample image with a handful of shapes on it - enough to practice
contour and shape detection on before touching real photos.
"""
import os
import cv2
import numpy as np

# folder this file lives in, so images/outputs always save HERE
# no matter which folder you were in when you ran "python 01_contours_basics.py"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

CANVAS_W, CANVAS_H = 800, 500  # size of our fake sample shapes image


def _ensure_folders():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def create_sample_shapes():
    # a plain white "page" with a handful of different shapes scattered on
    # it - one of each kind we want to be able to detect, in different
    # colors and sizes, plus one shape that overlaps another
    img = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)  # white background

    cv2.rectangle(img, (40, 40), (180, 150), (200, 80, 30), -1)      # rectangle (wide)
    cv2.rectangle(img, (230, 40), (350, 160), (30, 150, 30), -1)     # square
    cv2.circle(img, (500, 100), 65, (30, 30, 200), -1)               # circle
    triangle = np.array([[620, 170], [690, 170], [655, 50]], np.int32)
    cv2.fillPoly(img, [triangle], (180, 30, 180))                    # triangle

    pentagon = np.array([[100, 300], [140, 270], [180, 300], [165, 350], [115, 350]], np.int32)
    cv2.fillPoly(img, [pentagon], (200, 150, 0))                     # pentagon (polygon)

    hexagon_center, hex_r = (330, 330), 60
    hexagon = np.array([
        (int(hexagon_center[0] + hex_r * np.cos(np.pi / 3 * k)),
         int(hexagon_center[1] + hex_r * np.sin(np.pi / 3 * k)))
        for k in range(6)
    ], np.int32)
    cv2.fillPoly(img, [hexagon], (0, 120, 200))                      # hexagon (polygon)

    cv2.circle(img, (520, 330), 45, (60, 180, 200), -1)               # small circle
    cv2.rectangle(img, (600, 290), (700, 390), (100, 100, 100), -1)   # small square

    return img


VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def get_sample_image():
    # uses whatever real photo you've dropped into images/ - only falls
    # back to generating the fake sample shapes image if that folder is empty
    _ensure_folders()
    existing = sorted(f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(VALID_EXT))
    if existing:
        return cv2.imread(os.path.join(IMAGES_DIR, existing[0]))

    path = os.path.join(IMAGES_DIR, "sample_shapes.jpg")
    cv2.imwrite(path, create_sample_shapes())
    return cv2.imread(path)


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
    # comparisons are easy to see in a single saved file
    normalized = []
    for img in images:
        if img.ndim == 2:  # grayscale/binary -> BGR so everything can sit in one collage
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
