"""
Day 26 practice 5 - watershed for touching objects, and background removal.

Two demos:
  1. Watershed splitting two/three touching circles into separate objects -
     something a plain threshold cannot do, since touching blobs merge into
     one connected component.
  2. Foreground/background segmentation + background removal on a shadowed
     object, showing the morphology cleanup step doing its job.

Run:  python coding_practice/05_watershed_and_background_removal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation import (draw_watershed, hstack_padded, imwrite, label_image,  # noqa: E402
                          load_image, remove_background, resize_max_side,
                          segment_foreground, to_gray, watershed_segmentation)

ROOT = Path(__file__).resolve().parent.parent
WATERSHED_IMAGE = ROOT / "sample_images" / "obj06_overlapping_circles_watershed.jpg"
SHADOW_IMAGE = ROOT / "sample_images" / "shadow01_circle_cast_shadow.jpg"
OUT_WATERSHED = ROOT / "sample_outputs" / "practice_05_watershed.jpg"
OUT_BG_REMOVAL = ROOT / "sample_outputs" / "practice_05_background_removal.jpg"


def demo_watershed() -> None:
    image = resize_max_side(load_image(WATERSHED_IMAGE), 700)
    mask, markers, n_objects = watershed_segmentation(image, invert=True)
    overlay = draw_watershed(image, markers)

    n_blobs_before = cv2.connectedComponents(mask)[0] - 1
    print(f"Before watershed: threshold + connected-components finds "
          f"{n_blobs_before} blob(s) (the touching circles count as one).")
    print(f"After watershed:  {n_objects} separated object(s).")

    panels = [
        label_image(image, "original"),
        label_image(mask, f"otsu mask ({n_blobs_before} blob before split)"),
        label_image(overlay, f"watershed ({n_objects} objects separated)"),
    ]
    imwrite(OUT_WATERSHED, hstack_padded(panels))
    print(f"Wrote {OUT_WATERSHED}\n")


def demo_background_removal() -> None:
    image = resize_max_side(load_image(SHADOW_IMAGE), 600)
    seg = segment_foreground(image, method="otsu", invert=True)
    removed_white = remove_background(image, method="otsu", invert=True, bg_color=(255, 255, 255))
    removed_green = remove_background(image, method="otsu", invert=True, bg_color=(0, 200, 0))

    print(f"Foreground segmentation: {seg.n_components} component(s) kept "
          f"after morphology cleanup, {seg.elapsed_ms:.1f} ms.")

    panels = [
        label_image(image, "original (with cast shadow)"),
        label_image(seg.mask, "cleaned mask"),
        label_image(removed_white, "background -> white"),
        label_image(removed_green, "background -> green"),
    ]
    imwrite(OUT_BG_REMOVAL, hstack_padded(panels))
    print(f"Wrote {OUT_BG_REMOVAL}")


def main() -> None:
    print("=== Watershed: separating touching objects ===")
    demo_watershed()
    print("=== Foreground/background segmentation ===")
    demo_background_removal()


if __name__ == "__main__":
    main()
