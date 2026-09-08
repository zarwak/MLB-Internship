"""
Day 32 - batch runner: both demo variants x every sample video.

Saves every annotated output into outputs/variant-1/ (Parking Monitor) and
outputs/variant-2/ (Parking Analytics), and writes
outputs/occupancy_results.md summarizing what came out of each run - see
README.md "Full results" for the numbers this produced.

Run:  python coding_practice/02_batch_all_videos.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parking_detection import load_layout, load_model, process_video  # noqa: E402

VIDEO_DIR = ROOT / "sample_videos"
OUT_DIR = ROOT / "outputs"
VARIANT_DIRS = {"monitor": OUT_DIR / "variant-1", "analytics": OUT_DIR / "variant-2"}

# aerial_* clips are a steep top-down angle that plain COCO-pretrained YOLOv8n
# was measured to find ~0 vehicles on (see README "Real challenges faced") - they
# use a separate, custom-trained checkpoint and a layout recalibrated to real
# per-stall size (not hand-eyeballed zones), which is what makes this default
# stable rather than a fragile near-zero value.
AERIAL_OVERLAP_THRESHOLD = 0.15


def model_for(video_stem: str):
    if video_stem.startswith("aerial_"):
        return load_model(str(ROOT / "vehicle_detection_aerial.pt")), AERIAL_OVERLAP_THRESHOLD
    return load_model(str(ROOT / "yolov8n.pt")), None  # None -> process_video's own default


def main() -> None:
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    for d in VARIANT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for video_path in videos:
        layout_path = VIDEO_DIR / f"{video_path.stem}_spaces.json"
        if not layout_path.exists():
            print(f"skipping {video_path.name}: no {layout_path.name}")
            continue
        layout = load_layout(layout_path)
        model, overlap_threshold = model_for(video_path.stem)
        kwargs = {} if overlap_threshold is None else {"overlap_threshold": overlap_threshold}

        for variant, out_dir in VARIANT_DIRS.items():
            out_path = out_dir / f"{video_path.stem}_{variant}.mp4"
            print(f"{video_path.name} [{variant}] -> {out_path.name}")
            start = time.perf_counter()
            result = process_video(model, video_path, out_path, layout, variant=variant, **kwargs)
            elapsed = time.perf_counter() - start
            rows.append((video_path.stem, variant, result, elapsed))

    lines = ["# Day 32 - Batch Occupancy Results", "",
             "| Video | Variant | Frames | Spaces | Occupied | Free | Occupancy % | "
             "Unique vehicles | Time (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, variant, result, elapsed in rows:
        lines.append(f"| {name} | {variant} | {result.n_frames} | {result.total_spaces} | "
                     f"{result.final_occupied} | {result.final_free} | "
                     f"{result.final_occupancy_pct:.1f} | {result.unique_vehicles_seen} | {elapsed:.1f} |")

    report = OUT_DIR / "occupancy_results.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
