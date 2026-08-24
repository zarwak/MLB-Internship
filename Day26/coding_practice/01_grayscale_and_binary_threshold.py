"""
Day 26 practice 1 - grayscale conversion and binary thresholding.

Reads an image, converts it to grayscale, and applies a plain binary
threshold at a handful of cutoff values so you can see how sensitive the
result is to picking the "right" number by hand.

Run:  python coding_practice/01_grayscale_and_binary_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation import (binary_threshold, hstack_padded, imwrite,  # noqa: E402
                          label_image, load_image, resize_max_side, to_gray)

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "sample_images" / "obj01_circle_on_white.jpg"
OUT = ROOT / "sample_outputs" / "practice_01_binary_thresholds.jpg"

THRESHOLDS = [60, 100, 127, 160, 200]


def main() -> None:
    image = resize_max_side(load_image(IMAGE), 600)
    gray = to_gray(image)
    print(f"Loaded {IMAGE.name}: {image.shape[1]}x{image.shape[0]}, "
          f"converted to single-channel grayscale (shape {gray.shape}).")

    panels = [label_image(gray, "grayscale")]
    for t in THRESHOLDS:
        result = binary_threshold(gray, thresh=t, invert=True)
        print(f"  thresh={t:3d}  ->  {result.foreground_ratio:5.1f}% white pixels")
        panels.append(label_image(result.mask, f"t={t}"))

    imwrite(OUT, hstack_padded(panels))
    print(f"\nSweeping the cutoff shows the core problem with binary "
          f"thresholding: there is no single 'correct' value, only a "
          f"trade-off you pick by eye. Wrote {OUT}")


if __name__ == "__main__":
    main()
