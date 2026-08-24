"""
Day 26 practice 4 - compare binary, adaptive and Otsu across several images.

Runs all three thresholding families against every category in
sample_images/ (documents, plain objects, uneven lighting, shadows) and
writes one big comparison grid plus a markdown results table, so the
strengths/weaknesses of each method are visible across conditions rather
than on a single cherry-picked image.

Run:  python coding_practice/04_compare_thresholding_methods.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation import (THRESHOLD_METHODS, grid_padded, imwrite, label_image,  # noqa: E402
                          load_image, resize_max_side, to_gray)

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_images"
OUT_GRID = ROOT / "sample_outputs" / "practice_04_comparison_grid.jpg"
OUT_TABLE = ROOT / "sample_outputs" / "practice_04_comparison_table.md"

# One representative image per required category.
CATEGORY_IMAGES = {
    "document (clean)": "doc01_invoice_clean.jpg",
    "document (uneven light)": "doc05_report_uneven_light.jpg",
    "plain object": "obj01_circle_on_white.jpg",
    "spotlight / uneven light": "light01_circle_spotlight.jpg",
    "cast shadow": "shadow01_circle_cast_shadow.jpg",
}


def main() -> None:
    rows = []
    table = ["| Image | Binary | Adaptive-Mean | Adaptive-Gaussian | Otsu |",
            "|---|---|---|---|---|"]

    for label, filename in CATEGORY_IMAGES.items():
        path = SAMPLES / filename
        if not path.exists():
            print(f"skip {filename} (missing - run generate_samples.py)")
            continue
        image = resize_max_side(load_image(path), 420)
        gray = to_gray(image)

        row = [label_image(gray, label)]
        cells = []
        for name, fn in THRESHOLD_METHODS.items():
            result = fn(gray, invert=True)
            row.append(label_image(result.mask, f"{name} ({result.foreground_ratio:.0f}%)"))
            cells.append(f"{result.foreground_ratio:.1f}%")
        rows.append(row)
        table.append(f"| {label} | " + " | ".join(cells) + " |")
        print(f"{label:<28} " + "  ".join(f"{n}={c}" for n, c in
                                          zip(THRESHOLD_METHODS, cells)))

    imwrite(OUT_GRID, grid_padded(rows))
    OUT_TABLE.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_GRID}")
    print(f"Wrote {OUT_TABLE}")
    print("\nRead down each column: binary is stable only when foreground-% "
          "stays close across every row (it doesn't - see the uneven-light "
          "rows). Otsu tracks binary closely on clean bimodal images but "
          "spikes on gradients. Adaptive stays in a tighter band throughout, "
          "at the cost of a noisier mask on flat regions.")


if __name__ == "__main__":
    main()
