"""
Coding practice 1 - Detect corners using Harris Corner Detection.

Harris asks a simple question about every pixel: if I slide this little
window around, does the image underneath change a lot in *every* direction?

  - flat region  -> nothing changes           -> not a corner
  - edge         -> changes in one direction  -> not a corner
  - corner       -> changes in all directions -> corner

It answers it by building the structure tensor M from the x and y gradients
inside the window and scoring it with

    R = det(M) - k * trace(M)^2

Large positive R means both eigenvalues of M are large, which is the "changes
in every direction" case.

Run:  python coding_practice/01_harris_corners.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_detection import (  # noqa: E402
    VIS_EXT,
    detect_harris,
    draw_harris,
    draw_harris_response,
    hstack_padded,
    imwrite,
    label_image,
    load_image,
    resize_max_side,
)

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "sample_images" / "pair01_box_product_b.png"
OUT_DIR = ROOT / "sample_outputs"


def main() -> int:
    image = resize_max_side(load_image(IMAGE))
    print(f"image: {IMAGE.name}  {image.shape[1]}x{image.shape[0]}\n")

    # Sweeping the threshold shows what that 0.01 is actually doing: it is the
    # only thing standing between "a few hundred real corners" and "every
    # slightly textured pixel in the picture".
    panels = []
    print(f"{'threshold':>10} {'corners':>9} {'time (ms)':>10}")
    for threshold in (0.001, 0.01, 0.05, 0.20):
        result = detect_harris(image, threshold=threshold)
        print(f"{threshold:>10.3f} {result.count:>9d} {result.elapsed_ms:>10.2f}")
        panels.append(label_image(
            draw_harris(image, result),
            f"threshold {threshold:g} -> {result.count} corners"))

    imwrite(OUT_DIR / f"practice_01_harris_thresholds{VIS_EXT}", hstack_padded(panels))

    # The response map itself, before any thresholding. Red is a strong
    # corner response, blue is flat. This is what Harris really produces -
    # a picture, not a list of points.
    default = detect_harris(image)
    heat = label_image(draw_harris_response(image, default),
                       "raw Harris response R (red = corner-like)")
    corners = label_image(draw_harris(image, default),
                          f"after threshold + centroids -> {default.count} corners")
    imwrite(OUT_DIR / f"practice_01_harris_response{VIS_EXT}", hstack_padded([heat, corners]))

    print(f"\nk = {default.params['k']}, block size = {default.params['block_size']}, "
          f"Sobel ksize = {default.params['ksize']}")
    print(f"wrote {OUT_DIR / ('practice_01_harris_thresholds' + VIS_EXT)}")
    print(f"wrote {OUT_DIR / ('practice_01_harris_response' + VIS_EXT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
