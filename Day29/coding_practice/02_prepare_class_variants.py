"""
Day 29 - Build two alternative class-taxonomy copies of the dataset.

The raw 8-class dataset is badly imbalanced (see 01_explore_dataset.py):
pothole=4620 and Longitudinal-Crack=2616 instances vs. Ravelling=24,
Rutting=13, Striping=30. Those three near-empty classes drag the mAP@50
macro-average down regardless of how well the common classes are learned,
so we compare the original taxonomy against two reduced ones and let
train_all_variants.py pick a winner by actual validation mAP.

Variants (images are shared/copied as-is, only labels + data.yaml differ):
  dataset            - original, 8 classes, untouched
  dataset_merged6    - Ravelling+Rutting+Striping merged into "Other-Damage" (6 classes)
  dataset_dropped5   - Ravelling/Rutting/Striping removed entirely (5 classes)
  dataset_2class     - pothole vs. everything else ("crack") - the taxonomy that ended up
                       winning the GPU experiments (README.md "Challenges" / "Results")

Run: python coding_practice/02_prepare_class_variants.py
"""

import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dataset"
SPLITS = ["train", "valid", "test"]

ORIGINAL_NAMES = ["Alligator", "Edge Cracking", "Lateral-Crack", "Longitudinal-Crack",
                   "Ravelling", "Rutting", "Striping", "pothole"]
RARE = {"Ravelling", "Rutting", "Striping"}  # indices 4, 5, 6


def build_variant(dest_name: str, new_names: list[str], remap: dict[int, int | None]):
    """
    remap: old_class_id -> new_class_id, or None to drop that class's boxes.
    Images are copied once; label .txt files are rewritten with remapped ids
    (and dropped lines removed). Images that end up with zero boxes are kept
    as background/negative examples, which YOLO training handles natively.
    """
    dest = ROOT / dest_name
    print(f"\n=== Building {dest_name} ({len(new_names)} classes) ===")
    for split in SPLITS:
        src_img_dir = SRC / split / "images"
        src_lbl_dir = SRC / split / "labels"
        dst_img_dir = dest / split / "images"
        dst_lbl_dir = dest / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not any(dst_img_dir.iterdir()):
            for img in src_img_dir.iterdir():
                shutil.copy2(img, dst_img_dir / img.name)

        n_boxes_kept, n_boxes_dropped = 0, 0
        for lbl_path in src_lbl_dir.glob("*.txt"):
            out_lines = []
            for line in lbl_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                old_id = int(parts[0])
                new_id = remap[old_id]
                if new_id is None:
                    n_boxes_dropped += 1
                    continue
                n_boxes_kept += 1
                out_lines.append(" ".join([str(new_id), *parts[1:]]))
            (dst_lbl_dir / lbl_path.name).write_text(
                ("\n".join(out_lines) + "\n") if out_lines else ""
            )
        print(f"  {split}: {n_boxes_kept} boxes kept, {n_boxes_dropped} boxes dropped")

    data_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(new_names),
        "names": new_names,
    }
    with open(dest / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print(f"  wrote {dest / 'data.yaml'}")


def main():
    # merged6: Ravelling/Rutting/Striping -> "Other-Damage"
    merged_names = ["Alligator", "Edge Cracking", "Lateral-Crack", "Longitudinal-Crack",
                     "Other-Damage", "pothole"]
    merged_remap = {}
    for i, name in enumerate(ORIGINAL_NAMES):
        if name in RARE:
            merged_remap[i] = 4  # Other-Damage
        elif name == "pothole":
            merged_remap[i] = 5
        else:
            merged_remap[i] = i
    build_variant("dataset_merged6", merged_names, merged_remap)

    # dropped5: Ravelling/Rutting/Striping removed
    dropped_names = ["Alligator", "Edge Cracking", "Lateral-Crack", "Longitudinal-Crack", "pothole"]
    dropped_remap = {}
    for i, name in enumerate(ORIGINAL_NAMES):
        if name in RARE:
            dropped_remap[i] = None
        elif name == "pothole":
            dropped_remap[i] = 4
        else:
            dropped_remap[i] = i
    build_variant("dataset_dropped5", dropped_names, dropped_remap)

    # 2class: pothole vs. everything else ("crack") - no data dropped, just coarser.
    # This is the taxonomy the final shipped model (best.pt) was trained on.
    two_class_names = ["crack", "pothole"]
    two_class_remap = {i: (1 if name == "pothole" else 0) for i, name in enumerate(ORIGINAL_NAMES)}
    build_variant("dataset_2class", two_class_names, two_class_remap)

    print("\nDone. dataset/, dataset_merged6/, dataset_dropped5/, dataset_2class/ are ready to train.")


if __name__ == "__main__":
    main()
