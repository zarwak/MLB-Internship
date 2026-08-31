"""
Day 29 - Run inference on test images and save annotated results + a table.

Picks images from the test split, runs the trained model via detection.py
(same annotated-box drawing the Streamlit app uses), and saves annotated
copies plus a markdown table of every detection to sample_outputs/.

Run: python coding_practice/04_inference.py --weights best.pt --n 12
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detection import DEFAULT_CONF, detect_image, imwrite, load_image, load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--source", default="sample_images",
                    help="folder of images to run on (default: the curated sample_images/ set)")
    p.add_argument("--n", type=int, default=18, help="number of images to run inference on")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--out-dir", default="sample_outputs/predictions")
    args = p.parse_args()

    src_dir = ROOT / args.source
    images = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
    random.seed(0)
    sample = random.sample(images, min(args.n, len(images)))

    out_dir = ROOT / args.out_dir
    model = load_model(str(ROOT / args.weights))

    rows = ["| Image | Class | Confidence | Box (x1,y1,x2,y2) |", "|---|---|---|---|"]
    for img_path in sample:
        image = load_image(img_path)
        result = detect_image(model, image, conf=args.conf)
        out_path = out_dir / img_path.name
        imwrite(out_path, result.annotated)

        if not result.detections:
            rows.append(f"| {img_path.name} | *(no detections)* | - | - |")
        for d in result.detections:
            rows.append(f"| {img_path.name} | {d.class_name} | {d.confidence:.3f} | {d.box} |")

        print(f"{img_path.name}: {len(result.detections)} detection(s) -> {out_path.relative_to(ROOT)}")

    table_path = out_dir / "results_table.md"
    table_path.write_text(f"# Inference results ({len(sample)} images)\n\n" + "\n".join(rows) + "\n")
    print(f"\nWrote {table_path}")


if __name__ == "__main__":
    main()
