"""
Day 32 - Coding Practice: the brief's checklist, literally.

  * Load a parking lot video
  * Define multiple parking spaces
  * Detect vehicles using YOLO
  * Check which parking spaces are occupied
  * Track vehicles across frames
  Display: total spaces / occupied / available / occupancy % / save the video

Run:  python coding_practice/01_parking_occupancy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parking_detection import load_layout, load_model, process_video  # noqa: E402

VIDEO = ROOT / "sample_videos" / "mall_curbside_lot.mp4"
LAYOUT = ROOT / "sample_videos" / "mall_curbside_lot_spaces.json"
OUT = ROOT / "sample_outputs" / "mall_curbside_lot_occupancy.mp4"


def main() -> None:
    # Load a parking lot video
    model = load_model(str(ROOT / "yolov8n.pt"))

    # Define multiple parking spaces (a preset grid, calibrated once against
    # this clip's first frame - see README "How parking spaces are defined")
    layout = load_layout(LAYOUT)

    # Detect vehicles using YOLO + check which spaces are occupied + track
    # vehicles across frames - all three happen together inside
    # process_video(): model.track() detects AND tracks in one call, and
    # ParkingLotState.update() (parking_detection.py) does the occupancy check.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = process_video(model, VIDEO, OUT, layout, variant="monitor")

    # Display the 5 required stats
    print(f"Total spaces:        {result.total_spaces}")
    print(f"Occupied spaces:     {result.final_occupied}")
    print(f"Available spaces:    {result.final_free}")
    print(f"Occupancy:           {result.final_occupancy_pct:.1f}%")
    print(f"Saved processed video to: {OUT}")


if __name__ == "__main__":
    main()
