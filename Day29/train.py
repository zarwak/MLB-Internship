"""
Day 29 - Train a custom YOLO model on the Road Damage Detection dataset.

Thin wrapper around the Ultralytics Python API so the same script can be
reused for the CPU-timing calibration run and for each class-taxonomy
variant we compare (see coding_practice/02_prepare_class_variants.py and
README.md "Challenges" section for why there are multiple variants).

Examples:
    python train.py --data dataset/data.yaml --epochs 2 --name calibration
    python train.py --data dataset_dropped5/data.yaml --epochs 40 --imgsz 512 --name dropped5
    python train.py --resume runs/detect/road_damage_8class/weights/last.pt
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
# Absolute, so it can't get silently re-joined onto the machine's global
# Ultralytics `runs_dir` setting (that setting is left over from another
# day's project in this repo and pointed training output at the wrong place
# during our first calibration run when we passed a relative "runs/detect").
DEFAULT_PROJECT = str(ROOT / "runs" / "detect")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="path to a data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="base weights to start from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=15, help="early-stop patience")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--name", default="train")
    p.add_argument("--resume", help="path to a run's weights/last.pt to continue an interrupted run "
                                     "(reuses that run's original data/epochs/imgsz/etc. automatically)")
    args = p.parse_args()

    start = time.time()
    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        if not args.data:
            p.error("--data is required unless --resume is given")
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device="cpu",
            project=args.project,
            name=args.name,
            exist_ok=True,
            plots=True,
            verbose=True,
        )
    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed / 60:.1f} min")
    print(f"Best weights: {results.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
