"""
Day 27 practice 1 - install check + loading pretrained YOLO models.

Confirms the ultralytics package is installed, loads both a YOLOv8 and a
YOLO11 nano model (downloading the ~5-6 MB weights on first run), and prints
a side-by-side comparison of the two variants plus the 80 COCO classes they
were both trained to detect.

Run:  python coding_practice/01_install_and_load_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ultralytics  # noqa: E402

from detection import load_model, model_info  # noqa: E402


def main() -> None:
    print(f"ultralytics package version: {ultralytics.__version__}\n")

    print("Loading YOLOv8n (nano) ...")
    v8 = load_model("yolov8n.pt")
    info8 = model_info(v8)

    print("Loading YOLO11n (nano) ...")
    v11 = load_model("yolo11n.pt")
    info11 = model_info(v11)

    print("\nModel variant comparison (both trained on the same 80-class COCO dataset):")
    print(f"  {'':<12}{'YOLOv8n':>15}{'YOLO11n':>15}")
    print(f"  {'classes':<12}{info8['class_count']:>15}{info11['class_count']:>15}")
    print(f"  {'parameters':<12}{info8['param_count']:>15,}{info11['param_count']:>15,}")

    print(f"\nAll {info11['class_count']} COCO classes this pretrained model can detect "
          f"(no training required, this is inference-only):")
    for i, name in enumerate(info11["classes"]):
        end = "\n" if (i + 1) % 8 == 0 else "  "
        print(f"{name:<14}", end=end)
    print()

    print("\nNote: 'n' (nano) is the smallest/fastest variant. Larger variants "
          "(s/m/l/x) trade inference speed for accuracy - see README for the "
          "full size/speed/accuracy table.")


if __name__ == "__main__":
    main()
