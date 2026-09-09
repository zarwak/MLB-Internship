from __future__ import annotations

import argparse
from pathlib import Path

from people_counter import CountingLine, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_TRACKER, load_model, process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO person detection and counting on one video.")
    parser.add_argument("--video", type=str, required=True, help="Path to a video file")
    parser.add_argument("--out", type=str, default="output/people_counted.mp4", help="Output annotated video")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--tracker", type=str, default=DEFAULT_TRACKER)
    parser.add_argument("--line-orientation", choices=["horizontal", "vertical"], default="horizontal")
    parser.add_argument("--line-position", type=float, default=0.5)
    args = parser.parse_args()

    model = load_model()
    line = CountingLine(orientation=args.line_orientation, position=args.line_position)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = process_video(
        model=model,
        in_path=args.video,
        out_path=out_path,
        conf=args.conf,
        iou=args.iou,
        tracker=args.tracker,
        line=line,
    )

    print(f"Processed {result.n_frames} frames")
    print(f"Peak count: {result.peak_count}")
    print(f"Current frame count: {result.current_count}")
    print(f"Line crossings: {result.line_crossings}")
    print(f"Saved annotated video to: {result.out_path}")


if __name__ == "__main__":
    main()
