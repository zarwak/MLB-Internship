"""
Day 29 - Explore the downloaded YOLO dataset.

Walks the train/valid/test folders, parses data.yaml, counts images/labels
and per-class instances, and draws a few sample images with their ground
truth boxes so we can eyeball the annotation quality before training.

Run: python coding_practice/01_explore_dataset.py
"""

import random
from collections import Counter
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "sample_outputs" / "dataset_preview"
SPLITS = ["train", "valid", "test"]

# A fixed, distinct BGR colour per class index so boxes are easy to tell apart.
BOX_COLORS = [
    (255, 99, 71), (60, 179, 113), (255, 165, 0), (30, 144, 255),
    (238, 130, 238), (255, 215, 0), (0, 206, 209), (220, 20, 60),
]


def load_data_yaml() -> dict:
    with open(DATASET_DIR / "data.yaml") as f:
        return yaml.safe_load(f)


def yolo_line_to_box(line: str, img_w: int, img_h: int):
    """
    YOLO annotation format: one line per object, space-separated
        class_id  x_center  y_center  width  height
    all four geometry values normalized to [0, 1] relative to image size.
    """
    cls_id, xc, yc, w, h = (float(v) for v in line.split())
    cls_id = int(cls_id)
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return cls_id, x1, y1, x2, y2


def main():
    data = load_data_yaml()
    class_names = data["names"]
    print(f"data.yaml -> nc={data['nc']}, classes={class_names}")
    print(f"  train: {data['train']}")
    print(f"  val:   {data['val']}")
    print(f"  test:  {data['test']}")

    print("\n=== Split sizes ===")
    class_counts = Counter()
    split_image_counts = {}
    for split in SPLITS:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        images = sorted(img_dir.glob("*.jpg"))
        labels = sorted(lbl_dir.glob("*.txt"))
        split_image_counts[split] = len(images)
        print(f"  {split:6s}: {len(images):5d} images, {len(labels):5d} label files")

        for lbl_path in labels:
            text = lbl_path.read_text().strip()
            if not text:
                continue
            for line in text.splitlines():
                cls_id = int(line.split()[0])
                class_counts[class_names[cls_id]] += 1

    print("\n=== Instance count per class (all splits) ===")
    for name in class_names:
        print(f"  {name:20s} {class_counts.get(name, 0)}")
    total_instances = sum(class_counts.values())
    print(f"  {'TOTAL':20s} {total_instances}")

    print("\n=== Drawing sample ground-truth boxes ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    train_images = sorted((DATASET_DIR / "train" / "images").glob("*.jpg"))
    sample = random.sample(train_images, min(6, len(train_images)))

    for img_path in sample:
        lbl_path = DATASET_DIR / "train" / "labels" / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for line in lbl_path.read_text().strip().splitlines():
            cls_id, x1, y1, x2, y2 = yolo_line_to_box(line, w, h)
            color = BOX_COLORS[cls_id % len(BOX_COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_names[cls_id], (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        out_path = OUT_DIR / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"  wrote {out_path.relative_to(ROOT)}")

    print(f"\nDone. {len(sample)} preview images with ground-truth boxes in "
          f"{OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
