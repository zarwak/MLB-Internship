"""
Makes 5 different fake tilted document photos for the Challenge Task.
We don't have real scanned documents yet - real photos work too, just drop
them into tilted_inputs/ yourself and skip running this script.
"""
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "tilted_inputs")

# a different set of "tilted corner" positions for each fake photo
# order in each set: top-left, top-right, bottom-right, bottom-left
TILT_SETS = [
    [[150, 120], [750, 80], [800, 750], [100, 800]],
    [[100, 60], [780, 150], [820, 800], [60, 720]],
    [[200, 40], [820, 100], [780, 830], [90, 760]],
    [[60, 180], [700, 40], [850, 700], [150, 850]],
    [[130, 90], [790, 130], [770, 810], [110, 780]],
]


def make_flat_doc(index, w=600, h=800):
    # a plain white "page" with a border and a few lines of fake text
    doc = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(doc, (10, 10), (w - 10, h - 10), (0, 0, 0), 3)
    cv2.putText(doc, f"Document {index}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for i, y in enumerate(range(120, h - 60, 60)):
        text = f"This is line {i + 1} of document {index}."
        cv2.putText(doc, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return doc


def make_tilted_photo(doc, tilt_points, canvas_size=900):
    # pretends the page was photographed at an angle, lying on a table
    canvas = np.full((canvas_size, canvas_size, 3), 160, dtype=np.uint8)  # gray table
    h, w = doc.shape[:2]

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])   # doc's real corners
    dst = np.float32(tilt_points)                          # where those corners land on the table
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc, M, (canvas_size, canvas_size))

    white = np.full((h, w, 3), 255, dtype=np.uint8)         # used to find the doc's shape
    warped_mask = cv2.warpPerspective(white, M, (canvas_size, canvas_size))
    mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY) > 0

    canvas[mask] = warped[mask]  # paste the tilted document onto the table
    return canvas


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, tilt_points in enumerate(TILT_SETS, start=1):
        doc = make_flat_doc(i)
        tilted = make_tilted_photo(doc, tilt_points)
        path = os.path.join(OUTPUT_DIR, f"tilted_document_{i}.jpg")
        cv2.imwrite(path, tilted)
        print(f"Created: {path}")
