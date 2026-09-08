"""
Day 36 - Coding Practice: run both demo variants on every sample video.

Saves every annotated output into outputs/variant-1/ (Wrong-Way Detection)
and outputs/variant-2/ (Traffic Violation Analytics), and writes
outputs/violation_results.md - see README.md "Results" for what came out
of this.

Run:  python coding_practice/02_batch_all_videos.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from traffic_violation import load_config, load_model, process_video  # noqa: E402

VIDEO_DIR = ROOT / "sample_videos"
OUT_DIR = ROOT / "outputs"


def main() -> None:
    model = load_model(str(ROOT / "yolov8n.pt"))
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    rows = []

    for video in videos:
        config_path = VIDEO_DIR / f"{video.stem}_config.json"
        if not config_path.exists():
            print(f"skipping {video.name}: no *_config.json preset")
            continue
        direction, zone, direction_roi = load_config(config_path)

        for variant, subdir in (("wrong_way", "variant-1"), ("analytics", "variant-2")):
            out_path = OUT_DIR / subdir / f"{video.stem}_{variant}.mp4"
            print(f"{video.name:<28} [{variant:<10}] -> {out_path.relative_to(ROOT)}")
            start = time.perf_counter()
            result = process_video(model, video, out_path, direction, zone, direction_roi, variant=variant)
            elapsed = time.perf_counter() - start
            print(f"  {result.n_frames} frames, {elapsed:.1f}s, "
                  f"{result.total_vehicles} vehicles, {result.total_violations} violations "
                  f"(wrong-way {result.wrong_way_count}, zone {result.zone_count})")
            if variant == "wrong_way":  # stats are identical either way; record once per video
                rows.append((video.stem, direction.name, result))

    report_path = OUT_DIR / "violation_results.md"
    lines = ["# Day 36 - Batch Results\n",
             "| Video | Direction | Vehicles | Violations | Wrong-way | Zone | Frames |",
             "|---|---|---|---|---|---|---|"]
    for name, direction_name, r in rows:
        lines.append(f"| {name} | {direction_name} | {r.total_vehicles} | {r.total_violations} | "
                     f"{r.wrong_way_count} | {r.zone_count} | {r.n_frames} |")
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
