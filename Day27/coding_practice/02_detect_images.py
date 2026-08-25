"""
Day 27 practice 2 - run YOLO object detection on every sample image.

Loads YOLO11n once, runs it on every image in sample_images/, prints the
detected classes + confidence scores for each, and writes an annotated copy
of every image into sample_outputs/detected_images/.

Run:  python coding_practice/02_detect_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection import detect_image, imwrite, load_image, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "sample_images"
OUT_DIR = ROOT / "sample_outputs" / "detected_images"
CONF = 0.25


def main() -> None:
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        print("No images found in sample_images/. Run download_samples.py first.")
        return

    model = load_model()
    print(f"Loaded model, {len(images)} image(s) to process (conf >= {CONF})\n")

    total_detections = 0
    for path in images:
        image = load_image(path)
        result = detect_image(model, image, conf=CONF)
        total_detections += len(result.detections)

        out_path = OUT_DIR / f"{path.stem}_detected.jpg"
        imwrite(out_path, result.annotated)

        print(f"{path.name:<28} {image.shape[1]}x{image.shape[0]:<6} "
              f"{len(result.detections):>2} object(s)  ({result.elapsed_ms:.0f} ms)")
        if result.detections:
            for d in result.detections:
                print(f"    {d.class_name:<15} conf={d.confidence:.2f}  box={d.box}")
        else:
            print("    (nothing detected at this confidence threshold)")

    print(f"\n{total_detections} total detection(s) across {len(images)} image(s).")
    print(f"Annotated images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
