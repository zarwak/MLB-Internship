"""
Day 31 coding practice - run full vehicle counting (car/bus/truck/motorcycle,
horizontal counting line) on every sample video, save annotated outputs, and
write a results table.

This is what generates the real, measured numbers in README.md's "Full
results" section - same rigor as Day30's 01_track_videos.py (no invented
numbers).

Run: python coding_practice/02_batch_count_all_videos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from counting import VEHICLE_CLASS_IDS, CountingLine, count_video, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "sample_videos"
OUT_DIR = ROOT / "sample_outputs"

# (filename, counting-line position as a fraction of frame height) - all 5
# clips look down a road/intersection with traffic crossing roughly
# horizontally in frame, so every one uses a horizontal line; position is
# tuned per clip to sit across the lanes rather than off in the sky/verge.
VIDEOS = [
    ("highway_evening.mp4", 0.55),
    ("highway_cars_buses.mp4", 0.5),
    ("highway_fast_paced.mp4", 0.5),
    ("highway_many_cars.mp4", 0.5),
    ("urban_intersection_motorcycles.mp4", 0.6),
]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    model = load_model(str(ROOT / "yolov8n.pt"))
    rows = []

    for video_name, position in VIDEOS:
        in_path = VIDEO_DIR / video_name
        out_path = OUT_DIR / f"02_{Path(video_name).stem}_counted.mp4"
        line = CountingLine(orientation="horizontal", position=position)

        result = count_video(model, in_path, out_path, line=line, classes=VEHICLE_CLASS_IDS)
        class_summary = ", ".join(f"{k}={v}" for k, v in sorted(result.counts_by_class.items())) or "none"
        dir_summary = ", ".join(f"{k}={v}" for k, v in sorted(result.counts_by_direction.items())) or "none"
        print(f"{video_name:<38} total={result.total_count:<4} "
              f"({class_summary})  [{dir_summary}]  {result.elapsed_s:.1f}s")
        rows.append((video_name, result.n_frames, result.total_count, class_summary,
                     dir_summary, f"{result.elapsed_s:.1f}s"))

    lines = ["| Video | Frames | Total counted | Per-class | Per-direction | Time |",
             "|---|---|---|---|---|---|"]
    for video_name, n_frames, total, class_summary, dir_summary, elapsed in rows:
        lines.append(f"| {video_name} | {n_frames} | {total} | {class_summary} | "
                     f"{dir_summary} | {elapsed} |")
    (OUT_DIR / "counting_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'counting_results.md'}")


if __name__ == "__main__":
    main()
