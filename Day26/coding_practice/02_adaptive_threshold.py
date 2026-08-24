"""
Day 26 practice 2 - adaptive thresholding on an unevenly lit image.

Runs binary thresholding side by side with adaptive mean and adaptive
Gaussian thresholding on a page that has a lighting gradient across it, to
show why a single global cutoff fails where a local one doesn't.

Run:  python coding_practice/02_adaptive_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation import (adaptive_threshold, binary_threshold, hstack_padded,  # noqa: E402
                          imwrite, label_image, load_image, resize_max_side, to_gray)

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "sample_images" / "doc05_report_uneven_light.jpg"
OUT = ROOT / "sample_outputs" / "practice_02_adaptive_thresholds.jpg"


def main() -> None:
    image = resize_max_side(load_image(IMAGE), 700)
    gray = to_gray(image)

    binary = binary_threshold(gray, thresh=127, invert=True)
    mean = adaptive_threshold(gray, method="mean", block_size=35, c=10, invert=True)
    gaussian = adaptive_threshold(gray, method="gaussian", block_size=35, c=10, invert=True)

    for result in (binary, mean, gaussian):
        print(f"{result.method:<18} {result.foreground_ratio:5.1f}% foreground "
              f"in {result.elapsed_ms:5.2f} ms  params={result.params}")

    panels = [
        label_image(gray, "grayscale (uneven light)"),
        label_image(binary.mask, f"binary t=127 ({binary.foreground_ratio:.1f}%)"),
        label_image(mean.mask, f"adaptive mean ({mean.foreground_ratio:.1f}%)"),
        label_image(gaussian.mask, f"adaptive gaussian ({gaussian.foreground_ratio:.1f}%)"),
    ]
    imwrite(OUT, hstack_padded(panels))
    print(f"\nBinary thresholding turns the bright half of the page solid "
          f"white - it has no notion of 'local' brightness. Both adaptive "
          f"variants stay legible across the whole gradient because each "
          f"pixel is compared only to its own neighbourhood. Wrote {OUT}")


if __name__ == "__main__":
    main()
