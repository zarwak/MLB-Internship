"""
Day 36 - Coding Practice: the brief's literal checklist.

Detects vehicles using YOLO, tracks them with ByteTrack, defines a road
direction, detects vehicles moving the wrong way, draws a virtual
restricted zone, detects entries into it, records the vehicle ID + frame
for each violation, and saves the processed video into sample_outputs/.

Run:  python coding_practice/01_traffic_violation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from traffic_violation import RegionBox, RoadDirection, load_config, load_model, process_video  # noqa: E402

VIDEO = ROOT / "sample_videos" / "bangkok_boulevard.mp4"
CONFIG = ROOT / "sample_videos" / "bangkok_boulevard_config.json"
OUT = ROOT / "sample_outputs" / "bangkok_boulevard_wrong_way.mp4"


def main() -> None:
    model = load_model(str(ROOT / "yolov8n.pt"))
    direction, zone, direction_roi = load_config(CONFIG)

    print(f"Road direction: {direction.name} {direction.arrow}")
    print(f"Restricted zone: {zone}")

    result = process_video(model, VIDEO, OUT, direction, zone, direction_roi, variant="wrong_way")

    print(f"\nProcessed {result.n_frames} frames in {result.elapsed_s:.1f}s")
    print(f"Total vehicles tracked : {result.total_vehicles}")
    print(f"Total violations       : {result.total_violations}")
    print(f"  Wrong-way            : {result.wrong_way_count}")
    print(f"  Restricted zone      : {result.zone_count}")
    print("\nVehicle type stats:")
    for cls, n in result.vehicle_counts_by_class.items():
        print(f"  {cls:<12} {n}")

    print("\nViolation events (vehicle ID, type, frame):")
    if not result.events:
        print("  none recorded on this clip")
    for ev in result.events:
        print(f"  #{ev.track_id:<4} {ev.violation_type:<16} frame {ev.frame_idx} (t={ev.time_s:.1f}s)")

    print(f"\nSaved annotated video to {OUT}")


if __name__ == "__main__":
    main()
