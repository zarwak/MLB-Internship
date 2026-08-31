"""
Day 29 - Evaluate a trained model on the held-out test split.

Run: python coding_practice/03_evaluate.py --weights best.pt --data dataset/data.yaml
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--out", default="sample_outputs/metrics.md")
    args = p.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, split="test", device="cpu", plots=True)

    names = metrics.names
    lines = [
        "# Evaluation metrics (test split)",
        "",
        f"- weights: `{args.weights}`",
        f"- data: `{args.data}`",
        "",
        "## Overall",
        "",
        f"- mAP@50: **{metrics.box.map50:.4f}**",
        f"- mAP@50-95: **{metrics.box.map:.4f}**",
        f"- Precision: {metrics.box.mp:.4f}",
        f"- Recall: {metrics.box.mr:.4f}",
        "",
        "## Per-class AP@50",
        "",
        "| Class | AP@50 | AP@50-95 |",
        "|---|---|---|",
    ]
    for i, cls_idx in enumerate(metrics.ap_class_index):
        lines.append(f"| {names[cls_idx]} | {metrics.box.ap50[i]:.4f} | {metrics.box.ap[i]:.4f} |")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
