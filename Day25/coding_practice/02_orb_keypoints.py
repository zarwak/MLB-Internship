"""
Coding practice 2 - Detect keypoints using ORB.

ORB = Oriented FAST and Rotated BRIEF. Three parts:

  1. FAST finds candidate corners fast (it compares a pixel against a ring of
     16 neighbours and bails out early), run once per level of an image
     pyramid so the same corner is found whatever size it appears at.
  2. Each surviving corner gets an orientation from the intensity centroid of
     its patch - the direction from the centre of the patch to its "centre of
     mass".
  3. BRIEF then describes the patch as 256 binary intensity comparisons,
     rotated by that orientation so the description does not change when the
     image does.

The output is therefore richer than Harris': every keypoint carries a
position, a size, an angle and a 32-byte descriptor.

Run:  python coding_practice/02_orb_keypoints.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from feature_detection import (  # noqa: E402
    VIS_EXT,
    detect_orb,
    draw_orb,
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

    result = detect_orb(image, n_features=1000)
    print(f"keypoints requested : {result.params['n_features']}")
    print(f"keypoints returned  : {result.count}")
    print(f"detection time      : {result.elapsed_ms:.2f} ms")
    print(f"descriptor matrix   : {result.descriptors.shape} of {result.descriptors.dtype}")
    print(f"                      {result.descriptors.shape[1]} bytes = "
          f"{result.descriptors.shape[1] * 8} bits per keypoint\n")

    # What a single keypoint actually holds - this is the part that Harris
    # has no equivalent for.
    first = result.keypoints[0]
    print("one keypoint, unpacked:")
    print(f"  position   (x, y) : ({first.pt[0]:.1f}, {first.pt[1]:.1f})")
    print(f"  size (diameter)   : {first.size:.1f} px")
    print(f"  angle             : {first.angle:.1f} degrees")
    print(f"  response strength : {first.response:.6f}")
    print(f"  pyramid octave    : {first.octave}")
    print(f"  descriptor        : {result.descriptors[0][:8]} ... (first 8 of 32 bytes)\n")

    # Keypoints spread across the pyramid: this is ORB's scale invariance
    # made visible. Level 0 is the full-size image, each level up is 1/1.2
    # the size of the one below.
    octaves = np.array([kp.octave for kp in result.keypoints])
    sizes = np.array([kp.size for kp in result.keypoints])
    print(f"{'octave':>7} {'keypoints':>10} {'mean patch size':>17}")
    for octave in sorted(set(octaves.tolist())):
        selected = octaves == octave
        print(f"{octave:>7} {int(selected.sum()):>10} {sizes[selected].mean():>16.1f} px")

    angles = np.array([kp.angle for kp in result.keypoints])
    print(f"\nangles span {angles.min():.1f} to {angles.max():.1f} degrees - "
          "every keypoint is oriented, which is what makes the descriptor "
          "rotation invariant")

    # Compare how many features you ask for against how many you get. On a
    # low-texture image ORB simply cannot find its full quota.
    panels = []
    print(f"\n{'requested':>10} {'returned':>9} {'time (ms)':>10}")
    for requested in (100, 500, 1000, 2000):
        run = detect_orb(image, n_features=requested)
        print(f"{requested:>10} {run.count:>9} {run.elapsed_ms:>10.2f}")
        panels.append(label_image(
            draw_orb(image, run, max_draw=300),
            f"nfeatures={requested} -> {run.count} keypoints"))

    imwrite(OUT_DIR / f"practice_02_orb_budgets{VIS_EXT}", hstack_padded(panels))
    print(f"\nwrote {OUT_DIR / ('practice_02_orb_budgets' + VIS_EXT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
