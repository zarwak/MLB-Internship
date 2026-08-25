"""
Day 27 practice 4 - observe detected classes and confidence scores.

Re-runs detection over every sample image, aggregates the results (which
classes showed up, how often, at what confidence), and writes a markdown
summary table to sample_outputs/results_table.md - the table quoted in
README.md.

Run:  python coding_practice/04_analyze_results.py
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection import detect_image, load_image, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "sample_images"
OUT_TABLE = ROOT / "sample_outputs" / "results_table.md"
CONF = 0.25


def main() -> None:
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        print("No images found in sample_images/. Run download_samples.py first.")
        return

    model = load_model()

    per_image_rows = []
    class_confidences: dict[str, list[float]] = defaultdict(list)
    class_counts: Counter[str] = Counter()

    for path in images:
        image = load_image(path)
        result = detect_image(model, image, conf=CONF)
        names = sorted({d.class_name for d in result.detections})
        per_image_rows.append((path.name, len(result.detections), ", ".join(names) or "-",
                               result.elapsed_ms))
        for d in result.detections:
            class_confidences[d.class_name].append(d.confidence)
            class_counts[d.class_name] += 1

    lines = ["# Day 27 - Detection results summary", "",
             f"Model: YOLO11n, confidence threshold: {CONF}", "",
             "## Per-image results", "",
             "| Image | Objects found | Classes | Inference time |",
             "|---|---|---|---|"]
    for name, n, classes, ms in per_image_rows:
        lines.append(f"| {name} | {n} | {classes} | {ms:.0f} ms |")

    lines += ["", "## Per-class summary (across all sample images)", "",
              "| Class | Times detected | Min conf | Mean conf | Max conf |",
              "|---|---|---|---|---|"]
    for name, n in class_counts.most_common():
        confs = class_confidences[name]
        lines.append(f"| {name} | {n} | {min(confs):.2f} | {statistics.mean(confs):.2f} | {max(confs):.2f} |")

    total = sum(class_counts.values())
    lines += ["", f"**{total} total detections across {len(images)} images, "
                  f"{len(class_counts)} unique classes.**"]

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
