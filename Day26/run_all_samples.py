"""
Run every thresholding method against every sample image, save a comparison
panel per image into sample_outputs/, and write the results table quoted in
README.md.

Run:  python run_all_samples.py
"""

from __future__ import annotations

from pathlib import Path

from segmentation import (THRESHOLD_METHODS, hstack_padded, imwrite, label_image,
                          load_image, resize_max_side, segment_foreground, to_gray)

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "sample_images"
OUTPUTS = ROOT / "sample_outputs"

CATEGORY_BY_PREFIX = {
    "doc": "document",
    "obj": "plain object",
    "light": "uneven lighting",
    "shadow": "cast shadow",
}


def category_of(name: str) -> str:
    for prefix, label in CATEGORY_BY_PREFIX.items():
        if name.startswith(prefix):
            return label
    return "other"


def main() -> None:
    paths = sorted(SAMPLES.glob("*.jpg"))
    if not paths:
        print("No sample images found - run generate_samples.py first.")
        return

    table = ["| Image | Category | Binary fg% | Adaptive-Mean fg% | "
            "Adaptive-Gaussian fg% | Otsu fg% | Otsu t | Cleaned components |",
            "|---|---|---|---|---|---|---|---|"]

    for path in paths:
        image = resize_max_side(load_image(path), 700)
        gray = to_gray(image)
        category = category_of(path.stem)

        panels = [label_image(gray, "grayscale")]
        ratios = {}
        otsu_t = None
        for name, fn in THRESHOLD_METHODS.items():
            result = fn(gray, invert=True)
            ratios[name] = result.foreground_ratio
            if name == "otsu":
                otsu_t = result.threshold_value
            panels.append(label_image(result.mask, f"{name} ({result.foreground_ratio:.1f}%)"))

        seg = segment_foreground(image, method="otsu", invert=True)
        panels.append(label_image(seg.foreground, f"fg cut-out ({seg.n_components} comp.)"))

        out_path = OUTPUTS / f"compare_{path.stem}.jpg"
        imwrite(out_path, hstack_padded(panels))

        table.append(
            f"| {path.name} | {category} | {ratios['binary']:.1f} | "
            f"{ratios['adaptive_mean']:.1f} | {ratios['adaptive_gaussian']:.1f} | "
            f"{ratios['otsu']:.1f} | {otsu_t:.0f} | {seg.n_components} |"
        )
        print(f"{path.name:<38} {category:<16} components={seg.n_components}")

    table_path = OUTPUTS / "results_table.md"
    table_path.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"\nWrote {len(paths)} comparison panels to {OUTPUTS}/")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
