"""
Day 32 - external validation: check the occupancy pipeline against
independent ground truth instead of this project's own videos/layouts.

Uses the Kaggle "Parking Space Detection & Classification Dataset"
(trainingdatapro) - 30 drone top-down photos, 903 hand-labeled parking-space
polygons (free / not_free / partially_free), built by a different team for
a different purpose. See README.md "External validation against
independent ground truth" for what this found and why it matters.

Reuses parking_detection.py's own _order_ccw + cv2.intersectConvexConvex
overlap-ratio code directly against each labeled polygon, plus the same
Hungarian one-to-one vehicle<->space assignment ParkingLotState.update()
uses (scipy.optimize.linear_sum_assignment) - no video, no debounce (a
still-image "hit" is just assigned overlap >= threshold) - so this measures
the actual shipped occupancy logic, not a reimplementation of it.

The dataset itself is CC BY-NC-ND 4.0 (non-commercial, no-derivatives) and
is NOT committed to this repo - get your own copy before running this:

    kaggle datasets download -d trainingdatapro/parking-space-detection-dataset
    unzip parking-space-detection-dataset.zip -d coding_practice/kaggle_parking

(kaggle_parking/ is gitignored - see README.md for the license caveat.)

Run:  python coding_practice/03_validate_kaggle_dataset.py [weights_file]
      weights_file defaults to vehicle_detection_aerial.pt - these images
      are true nadir drone shots, the same camera angle that checkpoint was
      fine-tuned for. Pass yolov8n.pt to reproduce the negative-control
      result (the general-purpose detector finds almost nothing here).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "kaggle_parking"

sys.path.insert(0, str(ROOT))
from parking_detection import (DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_IOU,  # noqa: E402
                               _order_ccw, load_model, vehicle_class_ids)

THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]


def main() -> None:
    if not (DATA_DIR / "annotations.xml").exists():
        print(f"No dataset at {DATA_DIR} - see this file's docstring for how to get one.")
        raise SystemExit(1)

    weights = sys.argv[1] if len(sys.argv) > 1 else "vehicle_detection_aerial.pt"
    model = load_model(str(ROOT / weights))
    veh_ids = vehicle_class_ids(model.names)
    print(f"using {weights}")

    root = ET.parse(DATA_DIR / "annotations.xml").getroot()
    records: list[tuple[str, float]] = []  # (gt_label, best_overlap_ratio)
    n_images = 0

    for image_el in root.findall("image"):
        img_path = DATA_DIR / image_el.get("name")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"skip (missing): {img_path}")
            continue
        n_images += 1

        res = model(img, conf=DEFAULT_CONF, iou=DEFAULT_IOU, imgsz=DEFAULT_IMGSZ,
                    classes=veh_ids, verbose=False)[0]
        vehicle_polys = []
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            vehicle_polys.append(_order_ccw(poly))

        spaces = []  # (label, space_poly, space_area)
        for poly_el in image_el.findall("polygon"):
            pts = np.array([[float(v) for v in p.split(",")]
                            for p in poly_el.get("points").split(";")], dtype=np.float32)
            space_poly = _order_ccw(pts)
            spaces.append((poly_el.get("label"), space_poly, cv2.contourArea(space_poly)))

        # Same one-to-one constraint as ParkingLotState.update(): a vehicle
        # overlapping two adjacent labeled spaces (common in a tightly packed
        # real lot) can only be assigned to the single best one, not counted
        # as a hit for both - otherwise this validation would be checking
        # different occupancy logic than what's actually shipped.
        overlap_matrix = np.zeros((len(vehicle_polys), len(spaces)))
        for i, vpoly in enumerate(vehicle_polys):
            for j, (_, space_poly, space_area) in enumerate(spaces):
                if space_area <= 0:
                    continue
                inter_area, _ = cv2.intersectConvexConvex(space_poly, vpoly)
                overlap_matrix[i, j] = inter_area / space_area

        best_overlaps = [0.0] * len(spaces)
        if overlap_matrix.size > 0:
            vehicle_idx, space_idx = linear_sum_assignment(-overlap_matrix)
            for i, j in zip(vehicle_idx, space_idx):
                best_overlaps[j] = overlap_matrix[i, j]

        for (label, _, _), best_overlap in zip(spaces, best_overlaps):
            records.append((label, best_overlap))

    gt_map = {"free_parking_space": 0, "not_free_parking_space": 1}
    labeled = [(gt_map[label], ov) for label, ov in records if label in gt_map]
    partial = [ov for label, ov in records if label == "partially_free_parking_space"]

    gt = np.array([g for g, _ in labeled])
    ov = np.array([o for _, o in labeled])

    print(f"\n{n_images} images, {len(records)} labeled spaces "
          f"({(gt == 1).sum()} occupied, {(gt == 0).sum()} free, {len(partial)} partial - excluded below)")
    print(f"{'threshold':>9} {'accuracy':>9} {'precision':>10} {'recall':>8} {'f1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    for t in THRESHOLDS:
        pred = (ov >= t).astype(int)
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())
        acc = (tp + tn) / len(gt)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
        print(f"{t:>9.2f} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>6.3f} {tp:>4} {fp:>4} {fn:>4} {tn:>4}")

    if partial:
        pa = np.array(partial)
        print(f"\npartially_free_parking_space overlap stats (n={len(pa)}): "
              f"min={pa.min():.3f} median={np.median(pa):.3f} max={pa.max():.3f}")


if __name__ == "__main__":
    main()
