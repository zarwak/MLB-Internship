"""
Day 31 coding practice - detect, track and classify vehicles as car and
truck in one video; draw a counting line; count vehicles as they cross it;
display the running total on the video; save the processed video.

This is the literal "Coding Practice" checklist from the brief, restricted
to car+truck (CAR_TRUCK_CLASS_IDS) - the fuller car/bus/truck/motorcycle
classification lives in the Mini Project (app.py) and in
02_batch_count_all_videos.py below.

Run: python coding_practice/01_count_vehicles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from counting import CAR_TRUCK_CLASS_IDS, CountingLine, count_video, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "sample_videos" / "highway_many_cars.mp4"
OUT = ROOT / "sample_outputs" / "01_highway_many_cars_car_truck.mp4"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    model = load_model(str(ROOT / "yolov8n.pt"))
    line = CountingLine(orientation="horizontal", position=0.5)

    result = count_video(model, VIDEO, OUT, line=line, classes=CAR_TRUCK_CLASS_IDS)

    print(f"{VIDEO.name}: {result.n_frames} frames, {result.elapsed_s:.1f}s")
    print(f"Total vehicles counted (car+truck only): {result.total_count}")
    for cls, n in sorted(result.counts_by_class.items()):
        print(f"  {cls:<10} {n}")
    for direction, n in sorted(result.counts_by_direction.items()):
        print(f"  {direction:<10} {n}")
    print(f"Saved annotated video to {OUT}")


if __name__ == "__main__":
    main()
