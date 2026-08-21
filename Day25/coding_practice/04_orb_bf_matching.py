"""
Coding practice 4 and 5 - Match features between two similar images using
ORB + Brute Force matcher, and display the matched keypoints.

The pipeline, end to end:

    ORB on image A  ->  N_a keypoints + N_a x 32 descriptors
    ORB on image B  ->  N_b keypoints + N_b x 32 descriptors
    BFMatcher(NORM_HAMMING).knnMatch(k=2)
        for every descriptor in A, find its two closest in B
    Lowe's ratio test
        keep it only if  best.distance < 0.75 * second_best.distance
    RANSAC homography
        keep the survivors that all agree on one transform

The last two steps are the whole game. Brute force always returns a nearest
neighbour for every descriptor - even for keypoints that have no counterpart
at all - so unfiltered "matches" are meaningless.

Run:  python coding_practice/04_orb_bf_matching.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from feature_detection import (  # noqa: E402
    VIS_EXT,
    imwrite,
    label_image,
    load_image,
    resize_max_side,
)
from feature_matching import (  # noqa: E402
    draw_detected_object,
    draw_matches,
    match_images,
    summarise,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_images"
OUT_DIR = ROOT / "sample_outputs"

PAIR_A = SAMPLES / "pair01_box_product_a.png"
PAIR_B = SAMPLES / "pair01_box_product_b.png"


def show_filtering_effect(image_a, image_b, result) -> None:
    """Print how many matches survive each stage of filtering."""
    n_raw = len(result.raw_matches)
    print("\nwhat each filtering stage removes")
    print(f"  every descriptor in A gets a nearest neighbour : {n_raw}")
    print(f"  after Lowe's ratio test @ {result.ratio:.2f}                : "
          f"{result.n_good}  ({100.0 * result.n_good / n_raw:.1f}% kept)")
    if result.n_good:
        print(f"  after RANSAC geometric verification            : "
              f"{result.n_inliers}  ({result.inlier_rate:.1f}% of the good ones kept)")


def compare_ratios(image_a, image_b) -> None:
    """The ratio threshold is a precision/recall dial - show both ends of it."""
    print(f"\n{'ratio':>7} {'good':>6} {'verified':>10} {'inlier rate':>13}")
    for ratio in (0.60, 0.70, 0.75, 0.80, 0.90):
        run = match_images(image_a, image_b, ratio=ratio)
        print(f"{ratio:>7.2f} {run.n_good:>6} {run.n_inliers:>10} "
              f"{run.inlier_rate:>12.1f}%")
    print("  lower ratio  = fewer matches, but the ones you get are reliable")
    print("  higher ratio = more matches, more of them wrong")


def main() -> int:
    image_a = resize_max_side(load_image(PAIR_A))
    image_b = resize_max_side(load_image(PAIR_B))

    print(f"A: {PAIR_A.name}  {image_a.shape[1]}x{image_a.shape[0]}")
    print(f"B: {PAIR_B.name}  {image_b.shape[1]}x{image_b.shape[0]}\n")

    result = match_images(image_a, image_b)
    print(summarise(result, PAIR_A.name, PAIR_B.name))

    show_filtering_effect(image_a, image_b, result)
    compare_ratios(image_a, image_b)

    # ---- display the matched keypoints -------------------------------------
    # Three views, because they answer three different questions.

    # 1. The verified matches only - "where did the good correspondences go?"
    verified = draw_matches(image_a, image_b, result, max_draw=50, use_inliers=True)
    imwrite(OUT_DIR / f"practice_04_matches_verified{VIS_EXT}", label_image(
        verified,
        f"RANSAC-verified matches - {result.n_inliers} of {result.n_good} good "
        f"(top {min(50, result.n_inliers)} drawn)"))

    # 2. Ratio-test matches without the geometry check - "what does RANSAC
    #    actually throw away?" On a clean pair the two look nearly identical;
    #    on a hard pair the difference is obvious.
    unverified = draw_matches(image_a, image_b, result, max_draw=50, use_inliers=False)
    imwrite(OUT_DIR / f"practice_04_matches_ratio_only{VIS_EXT}", label_image(
        unverified,
        f"ratio test only, no geometry check - {result.n_good} good "
        f"(top {min(50, result.n_good)} drawn)"))

    # 3. Unfiltered nearest neighbours - the cautionary picture. Every one of
    #    these is a "match" as far as brute force is concerned.
    raw_best = sorted([pair[0] for pair in result.raw_matches if pair],
                      key=lambda m: m.distance)[:50]
    raw_canvas = cv2.drawMatches(
        image_a, result.keypoints_a, image_b, result.keypoints_b,
        raw_best, None, matchColor=(0, 140, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    imwrite(OUT_DIR / f"practice_04_matches_unfiltered{VIS_EXT}", label_image(
        raw_canvas,
        f"no filtering - closest 50 of {len(result.raw_matches)} nearest neighbours"))

    # 4. The homography, applied: project A's outline onto B.
    located = draw_detected_object(image_a, image_b, result)
    if located is not None:
        imwrite(OUT_DIR / f"practice_04_object_located{VIS_EXT}", label_image(
            located, "image A's border projected into image B"))
        print("\nthe homography places image A's outline on the object in image B")
    else:
        print("\nnot enough verified matches to trust a homography")

    print(f"\nwrote 4 match visualisations to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
