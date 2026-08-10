"""
Optional helper: makes ONE fake tilted document photo inside input_images/,
so you can test the tool right now, before you have real document photos.
"""
import os
import cv2
import numpy as np

# folder this file lives in, so the test image always lands HERE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_images")


def make_flat_doc(w=600, h=800):
    # a plain white "page" with a border and a few lines of fake text
    doc = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(doc, (10, 10), (w - 10, h - 10), (0, 0, 0), 3)
    for i, y in enumerate(range(80, h - 60, 60)):
        text = f"Sample document line {i + 1}"
        cv2.putText(doc, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return doc


def make_tilted_photo(doc, canvas_size=900):
    # pretends the page was photographed at an angle, lying on a table
    canvas = np.full((canvas_size, canvas_size, 3), 160, dtype=np.uint8)  # gray table
    h, w = doc.shape[:2]

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])                       # doc's real corners
    dst = np.float32([[150, 120], [750, 80], [800, 750], [100, 800]])         # tilted spot on the table
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc, M, (canvas_size, canvas_size))          # tilted-looking doc

    white = np.full((h, w, 3), 255, dtype=np.uint8)                            # used to find doc's shape
    warped_mask = cv2.warpPerspective(white, M, (canvas_size, canvas_size))
    mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY) > 0

    canvas[mask] = warped[mask]  # paste the tilted document onto the table
    return canvas


if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    tilted = make_tilted_photo(make_flat_doc())
    path = os.path.join(INPUT_DIR, "test_tilted_sample.jpg")
    cv2.imwrite(path, tilted)
    print(f"Created a test image: {path}")
    print("Now run: python process.py")
