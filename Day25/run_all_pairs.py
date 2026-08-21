"""
Run the ORB + Brute Force matching pipeline over all 10 sample pairs.

Writes one match visualisation per pair into `sample_outputs/`, prints a
results table, and saves that table as markdown so the README can quote real
numbers instead of invented ones.

Run:  python run_all_pairs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from download_samples import PAIRS
from feature_detection import (
    VIS_EXT,
    imwrite,
    label_image,
    load_image,
    resize_max_side,
)
from feature_matching import (
    draw_detected_object,
    draw_matches,
    match_images,
)

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "sample_images"
OUT_DIR = ROOT / "sample_outputs"


def find_pair(pair_id: str):
    """Locate the two files for a pair, whatever extension they ended up with."""
    matches_a = sorted(SAMPLES.glob(f"{pair_id}_a.*"))
    matches_b = sorted(SAMPLES.glob(f"{pair_id}_b.*"))
    if not matches_a or not matches_b:
        return None
    return matches_a[0], matches_b[0]


def main() -> int:
    if not SAMPLES.exists() or not any(SAMPLES.glob("pair*")):
        print("no sample images found - run `python download_samples.py` first")
        return 1

    OUT_DIR.mkdir(exist_ok=True)

    header = (f"{'pair':<24} {'category':<28} {'kp A':>6} {'kp B':>6} "
              f"{'good':>6} {'verified':>9} {'inlier%':>8} {'ms':>7}")
    print(header)
    print("-" * len(header))

    rows = []
    for pair_id, category, _, _, note in PAIRS:
        found = find_pair(pair_id)
        if found is None:
            print(f"{pair_id:<24} missing")
            continue

        path_a, path_b = found
        image_a = resize_max_side(load_image(path_a))
        image_b = resize_max_side(load_image(path_b))

        result = match_images(image_a, image_b)

        print(f"{pair_id:<24} {category:<28} {result.count_a:>6} {result.count_b:>6} "
              f"{result.n_good:>6} {result.n_inliers:>9} {result.inlier_rate:>7.1f}% "
              f"{result.total_ms:>7.0f}")

        rows.append({
            "pair_id": pair_id,
            "category": category,
            "note": note,
            "count_a": result.count_a,
            "count_b": result.count_b,
            "n_good": result.n_good,
            "n_inliers": result.n_inliers,
            "inlier_rate": result.inlier_rate,
            "match_rate": result.match_rate,
            "total_ms": result.total_ms,
        })

        canvas = draw_matches(image_a, image_b, result, max_draw=40)
        caption = (f"{pair_id} - {result.count_a} vs {result.count_b} keypoints, "
                   f"{result.n_good} good, {result.n_inliers} verified "
                   f"({result.inlier_rate:.0f}% inliers)")
        imwrite(OUT_DIR / f"{pair_id}_matches{VIS_EXT}", label_image(canvas, caption))

        located = draw_detected_object(image_a, image_b, result)
        if located is not None:
            imwrite(OUT_DIR / f"{pair_id}_located{VIS_EXT}",
                    label_image(located, f"{pair_id} - A projected into B"))

    if not rows:
        print("nothing processed")
        return 1

    # ---- ranking ----------------------------------------------------------
    # Rank on verified inliers rather than raw good matches. A pair can rack
    # up good matches on repeated texture and still be geometrically wrong;
    # inliers are the ones that survived agreeing on a single transform.
    ranked = sorted(rows, key=lambda r: (r["n_inliers"], r["inlier_rate"]), reverse=True)

    print("\nranked by geometrically verified matches")
    print(f"{'#':>2} {'pair':<24} {'verified':>9} {'inlier%':>8}")
    for index, row in enumerate(ranked, 1):
        print(f"{index:>2} {row['pair_id']:<24} {row['n_inliers']:>9} "
              f"{row['inlier_rate']:>7.1f}%")

    best, worst = ranked[0], ranked[-1]
    print(f"\nbest : {best['pair_id']} - {best['n_inliers']} verified matches "
          f"at {best['inlier_rate']:.0f}% inliers")
    print(f"       ({best['note']})")
    print(f"worst: {worst['pair_id']} - {worst['n_inliers']} verified matches "
          f"at {worst['inlier_rate']:.0f}% inliers")
    print(f"       ({worst['note']})")

    # ---- markdown for the README -----------------------------------------
    lines = [
        "| Pair | Category | Keypoints A | Keypoints B | Good | Verified | Inlier % |",
        "|------|----------|------------:|------------:|-----:|---------:|---------:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['pair_id']}` | {row['category']} | {row['count_a']} | "
            f"{row['count_b']} | {row['n_good']} | {row['n_inliers']} | "
            f"{row['inlier_rate']:.1f}% |")

    table_path = OUT_DIR / "results_table.md"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {len(rows)} match sheets and {table_path.name} to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
