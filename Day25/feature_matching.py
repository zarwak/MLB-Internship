"""
Feature matching: ORB descriptors + Brute Force matcher.

Given two images this finds ORB keypoints in each, matches the binary
descriptors with `cv2.BFMatcher(NORM_HAMMING)`, filters the raw matches down
to the trustworthy ones, and optionally checks that the survivors agree on a
single homography.

Run it directly on a pair:

    python feature_matching.py sample_images/pair01_box_product_a.png \
                               sample_images/pair01_box_product_b.png
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from feature_detection import (
    ORB_N_FEATURES,
    VIS_EXT,
    detect_orb,
    imwrite,
    label_image,
    load_image,
    resize_max_side,
)

# Lowe's ratio. 0.75 is the usual starting point: a match is kept only when
# the best candidate is clearly better than the runner-up.
DEFAULT_RATIO = 0.75

# RANSAC reprojection tolerance in pixels when fitting the homography.
RANSAC_REPROJ_THRESHOLD = 5.0

# Below this many verified inliers a homography is not worth believing.
MIN_INLIERS_FOR_HOMOGRAPHY = 10


@dataclass
class MatchResult:
    """Everything the app and the CLI need to report on one image pair."""

    keypoints_a: tuple
    keypoints_b: tuple
    raw_matches: list                    # best-of-2 candidates, before filtering
    good_matches: list                   # survived the ratio test, sorted by distance
    inlier_matches: list                 # of those, the ones RANSAC agreed with
    homography: np.ndarray | None
    detect_ms: float
    match_ms: float
    ratio: float

    notes: list = field(default_factory=list)

    @property
    def count_a(self) -> int:
        return len(self.keypoints_a)

    @property
    def count_b(self) -> int:
        return len(self.keypoints_b)

    @property
    def n_good(self) -> int:
        return len(self.good_matches)

    @property
    def n_inliers(self) -> int:
        return len(self.inlier_matches)

    @property
    def match_rate(self) -> float:
        """Good matches as a share of the smaller keypoint set, in percent.

        Dividing by the smaller set is the fair denominator: you can never get
        more matches than the image with fewer keypoints can supply.
        """
        smaller = min(self.count_a, self.count_b)
        return 100.0 * self.n_good / smaller if smaller else 0.0

    @property
    def inlier_rate(self) -> float:
        """Share of good matches that survived RANSAC, in percent.

        This is the honest quality signal. A high `n_good` with a low inlier
        rate means the ratio test let a pile of coincidences through.
        """
        return 100.0 * self.n_inliers / self.n_good if self.n_good else 0.0

    @property
    def total_ms(self) -> float:
        return self.detect_ms + self.match_ms


def match_descriptors(descriptors_a: np.ndarray | None,
                      descriptors_b: np.ndarray | None,
                      ratio: float = DEFAULT_RATIO):
    """Brute-force match two sets of ORB descriptors and apply the ratio test.

    Returns `(raw_matches, good_matches, notes)`.

    Hamming distance is the right metric here because ORB descriptors are
    256-bit strings, not vectors - comparing them is a popcount of the XOR,
    which is why brute force stays cheap even at a thousand keypoints each.

    `crossCheck` is left off deliberately: it cannot be combined with
    `knnMatch`, and the ratio test is the stronger filter of the two. The
    ratio test compares the best match against the second best and throws the
    pair away when they are too close, which is exactly the situation where a
    repeated texture would otherwise produce a confident-looking mistake.
    """
    notes: list[str] = []

    if descriptors_a is None or descriptors_b is None:
        notes.append("one of the images produced no ORB descriptors at all")
        return [], [], notes
    if len(descriptors_a) == 0 or len(descriptors_b) == 0:
        notes.append("one of the images produced no ORB descriptors at all")
        return [], [], notes

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    if len(descriptors_b) < 2:
        # knnMatch with k=2 needs two candidates to compare; fall back to a
        # plain match and keep everything, noting that no filtering happened.
        plain = matcher.match(descriptors_a, descriptors_b)
        plain = sorted(plain, key=lambda m: m.distance)
        notes.append("second image had fewer than 2 descriptors - ratio test skipped")
        return [[m] for m in plain], plain, notes

    knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, runner_up = pair
        if best.distance < ratio * runner_up.distance:
            good.append(best)

    good.sort(key=lambda m: m.distance)
    return knn, good, notes


def estimate_homography(keypoints_a: tuple, keypoints_b: tuple, good_matches: list,
                        reproj_threshold: float = RANSAC_REPROJ_THRESHOLD):
    """Fit a homography through the good matches with RANSAC.

    Returns `(homography, inlier_matches)`. This is the geometric sanity
    check: individual descriptor matches can be wrong, but wrong matches
    rarely agree on the same plane-to-plane transform, so RANSAC separates
    the real correspondences from the lucky ones.

    Returns `(None, [])` when there are too few matches to fit anything, or
    when the fit fails. Note this assumes a roughly planar scene or a pure
    camera rotation - for the stereo pairs below, a fundamental matrix would
    model the geometry better, so a lowish inlier rate there is expected
    rather than a sign that matching failed.
    """
    if len(good_matches) < MIN_INLIERS_FOR_HOMOGRAPHY:
        return None, []

    src = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj_threshold)
    if homography is None or mask is None:
        return None, []

    flat = mask.ravel().astype(bool)
    inliers = [m for m, keep in zip(good_matches, flat) if keep]
    return homography, inliers


def match_images(image_a: np.ndarray, image_b: np.ndarray,
                 n_features: int = ORB_N_FEATURES,
                 ratio: float = DEFAULT_RATIO,
                 verify_geometry: bool = True) -> MatchResult:
    """Full pipeline for one pair: detect, describe, match, filter, verify."""
    orb_a = detect_orb(image_a, n_features=n_features)
    orb_b = detect_orb(image_b, n_features=n_features)
    detect_ms = orb_a.elapsed_ms + orb_b.elapsed_ms

    start = time.perf_counter()
    raw, good, notes = match_descriptors(orb_a.descriptors, orb_b.descriptors, ratio)
    match_ms = (time.perf_counter() - start) * 1000.0

    homography, inliers = (None, [])
    if verify_geometry and good:
        homography, inliers = estimate_homography(orb_a.keypoints, orb_b.keypoints, good)
        if homography is None and len(good) >= MIN_INLIERS_FOR_HOMOGRAPHY:
            notes.append("RANSAC could not fit a homography through the good matches")

    return MatchResult(
        keypoints_a=orb_a.keypoints,
        keypoints_b=orb_b.keypoints,
        raw_matches=raw,
        good_matches=good,
        inlier_matches=inliers,
        homography=homography,
        detect_ms=detect_ms,
        match_ms=match_ms,
        ratio=ratio,
        notes=notes,
    )


def draw_matches(image_a: np.ndarray, image_b: np.ndarray, result: MatchResult,
                 max_draw: int = 50, use_inliers: bool = True) -> np.ndarray:
    """Draw the top matches as lines between the two images.

    Drawing all of them is unreadable, so this shows the best `max_draw`.
    When RANSAC found inliers those are drawn in preference to the raw good
    matches, because they are the ones actually worth looking at.
    """
    matches = result.inlier_matches if (use_inliers and result.inlier_matches) else result.good_matches
    matches = matches[:max_draw]

    if not matches:
        canvas = np.hstack([
            image_a,
            np.full((image_a.shape[0], 8, 3), 32, np.uint8),
            cv2.resize(image_b, (image_b.shape[1],
                                 image_a.shape[0])) if image_b.shape[0] != image_a.shape[0] else image_b,
        ])
        return label_image(canvas, "no matches to draw")

    return cv2.drawMatches(
        image_a, result.keypoints_a,
        image_b, result.keypoints_b,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(160, 160, 160),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_detected_object(image_a: np.ndarray, image_b: np.ndarray,
                         result: MatchResult) -> np.ndarray | None:
    """Project image A's border into image B using the homography.

    This is the payoff shot: if the match is genuine, the quadrilateral lands
    exactly on the object. Returns None when there is no trustworthy
    homography to draw.
    """
    if result.homography is None or result.n_inliers < MIN_INLIERS_FOR_HOMOGRAPHY:
        return None

    height, width = image_a.shape[:2]
    corners = np.float32([[0, 0], [0, height - 1],
                          [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
    try:
        projected = cv2.perspectiveTransform(corners, result.homography)
    except cv2.error:
        return None

    canvas = image_b.copy()
    cv2.polylines(canvas, [np.int32(projected)], True, (0, 255, 255), 3, cv2.LINE_AA)
    return canvas


def summarise(result: MatchResult, name_a: str = "image A", name_b: str = "image B") -> str:
    """Plain-text report - reused by the CLI and by the Gradio app."""
    lines = [
        f"Keypoints in {name_a} : {result.count_a}",
        f"Keypoints in {name_b} : {result.count_b}",
        f"Good matches (ratio test @ {result.ratio:.2f}) : {result.n_good}",
        f"Geometrically verified inliers (RANSAC)     : {result.n_inliers}",
        f"Match rate (good / smaller keypoint set)    : {result.match_rate:.1f}%",
        f"Inlier rate (verified / good)               : {result.inlier_rate:.1f}%",
        f"Time - detect {result.detect_ms:.1f} ms, match {result.match_ms:.1f} ms, "
        f"total {result.total_ms:.1f} ms",
    ]
    if result.good_matches:
        distances = [m.distance for m in result.good_matches]
        lines.append(f"Hamming distance of good matches - best {min(distances):.0f}, "
                     f"mean {sum(distances) / len(distances):.1f}, worst {max(distances):.0f}")
    for note in result.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match two images with ORB + Brute Force matcher.")
    parser.add_argument("image_a")
    parser.add_argument("image_b")
    parser.add_argument("--out", default=None, help="where to write the match visualisation")
    parser.add_argument("--features", type=int, default=ORB_N_FEATURES)
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--max-draw", type=int, default=50)
    parser.add_argument("--no-geometry", action="store_true",
                        help="skip the RANSAC homography check")
    args = parser.parse_args()

    image_a = resize_max_side(load_image(args.image_a))
    image_b = resize_max_side(load_image(args.image_b))

    result = match_images(image_a, image_b, n_features=args.features,
                          ratio=args.ratio, verify_geometry=not args.no_geometry)

    print(summarise(result, Path(args.image_a).name, Path(args.image_b).name))

    canvas = draw_matches(image_a, image_b, result, max_draw=args.max_draw)
    shown = min(args.max_draw, result.n_inliers or result.n_good)
    canvas = label_image(
        canvas,
        f"ORB + BFMatcher - {result.count_a} vs {result.count_b} keypoints, "
        f"{result.n_good} good, {result.n_inliers} verified, showing {shown}")

    out = args.out
    if out is None:
        out = Path("sample_outputs") / f"match_{Path(args.image_a).stem}{VIS_EXT}"
    imwrite(out, canvas)
    print(f"wrote {out}")

    box = draw_detected_object(image_a, image_b, result)
    if box is not None:
        box_out = Path(str(out).replace(VIS_EXT, "_located" + VIS_EXT))
        imwrite(box_out, label_image(box, "image A projected into image B via the homography"))
        print(f"wrote {box_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
