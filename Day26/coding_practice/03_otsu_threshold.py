"""
Day 26 practice 3 - Otsu's automatic threshold.

Shows Otsu picking its own cutoff from the grayscale histogram, and plots
the histogram with the chosen threshold marked so the "minimise within-class
variance" idea is visible rather than just quoted.

Run:  python coding_practice/03_otsu_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation import (hstack_padded, imwrite, label_image, load_image,  # noqa: E402
                          otsu_threshold, resize_max_side, to_gray)

ROOT = Path(__file__).resolve().parent.parent
IMAGES = [
    ROOT / "sample_images" / "obj01_circle_on_white.jpg",
    ROOT / "sample_images" / "doc03_report_coffee_stain.jpg",
]
OUT_PANELS = ROOT / "sample_outputs" / "practice_03_otsu_masks.jpg"
OUT_HIST = ROOT / "sample_outputs" / "practice_03_otsu_histogram.png"


def main() -> None:
    panels = []
    fig, axes = plt.subplots(1, len(IMAGES), figsize=(6 * len(IMAGES), 4))
    if len(IMAGES) == 1:
        axes = [axes]

    for ax, path in zip(axes, IMAGES):
        image = resize_max_side(load_image(path), 500)
        gray = to_gray(image)
        result = otsu_threshold(gray, invert=True)
        print(f"{path.name:<32} Otsu picked t={result.threshold_value:.0f}  "
              f"({result.foreground_ratio:.1f}% foreground, {result.elapsed_ms:.2f} ms)")

        panels.append(label_image(gray, path.stem))
        panels.append(label_image(result.mask, f"otsu t={result.threshold_value:.0f}"))

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        ax.plot(hist, color="steelblue")
        ax.axvline(result.threshold_value, color="crimson", linestyle="--",
                  label=f"t={result.threshold_value:.0f}")
        ax.set_title(path.stem)
        ax.set_xlabel("pixel intensity")
        ax.set_ylabel("count")
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_HIST, dpi=130)
    imwrite(OUT_PANELS, hstack_padded(panels))
    print(f"\nOtsu's threshold lands in the valley between the two histogram "
          f"peaks - background pixels clustered on one side, foreground on "
          f"the other. Wrote {OUT_PANELS} and {OUT_HIST}")


if __name__ == "__main__":
    main()
