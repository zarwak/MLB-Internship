"""
Coding practice 3 - Visualise the detected keypoints.

Runs both detectors over several of the sample images and writes one
side-by-side sheet per image:

    left  : Harris corners, drawn as plain dots (a Harris corner is only a
            position, so a dot is the whole truth about it)
    right : ORB keypoints, drawn as circles with a radius line (the circle
            size is the patch the descriptor was computed over, the line is
            the measured orientation)

Seeing them next to each other is the quickest way to understand what ORB
adds: the same kind of corner, but with a scale and an angle attached.

Run:  python coding_practice/03_visualize_keypoints.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_detection import (  # noqa: E402
    VIS_EXT,
    detect_harris,
    detect_orb,
    draw_harris,
    draw_orb,
    hstack_padded,
    imwrite,
    label_image,
    load_image,
    resize_max_side,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_images"
OUT_DIR = ROOT / "sample_outputs"

IMAGES = [
    "pair01_box_product_a.png",
    "pair02_graffiti_wall_a.png",
    "pair03_leuven_landmark_a.jpg",
    "pair07_text_page_a.png",
]


def main() -> int:
    print(f"{'image':<32} {'Harris':>8} {'ORB':>8}   {'Harris ms':>10} {'ORB ms':>8}")
    print("-" * 74)

    for name in IMAGES:
        path = SAMPLES / name
        if not path.exists():
            print(f"{name:<32} missing - run download_samples.py first")
            continue

        image = resize_max_side(load_image(path))
        harris = detect_harris(image)
        orb = detect_orb(image)

        print(f"{name:<32} {harris.count:>8} {orb.count:>8}   "
              f"{harris.elapsed_ms:>10.1f} {orb.elapsed_ms:>8.1f}")

        left = label_image(draw_harris(image, harris),
                           f"Harris - {harris.count} corners (position only)")
        right = label_image(draw_orb(image, orb, rich=True, max_draw=250),
                            f"ORB - {orb.count} keypoints (position + scale + angle)")

        out = OUT_DIR / f"practice_03_keypoints_{Path(name).stem}{VIS_EXT}"
        imwrite(out, hstack_padded([left, right]))

    print(f"\nwrote {len(IMAGES)} comparison sheets to {OUT_DIR}")
    print("\nWhat to look for:")
    print("  - Harris dots and ORB circle centres land on the same kinds of")
    print("    corners; ORB uses the Harris score internally to rank its")
    print("    FAST candidates, so this is expected, not a coincidence.")
    print("  - ORB circles come in many sizes. Big circles were found on")
    print("    small (blurred-down) pyramid levels and correspond to coarse,")
    print("    large-scale structure. Harris has one fixed window and so")
    print("    only ever sees one scale.")
    print("  - On the flat sky or plain wall regions both find nothing:")
    print("    no gradient in two directions means no corner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
