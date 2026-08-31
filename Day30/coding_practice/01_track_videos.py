"""
Day 30 coding practice - run tracking on every sample video with every
supported tracker, save annotated outputs, and report unique-object counts.

Fulfills the brief's "Coding Practice" checklist: load a model, run
tracking on >=5 videos, show a unique ID per object, count unique objects
per video, save the output videos.

Run: python coding_practice/01_track_videos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracking import TRACKERS, load_model, track_video  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "sample_videos"
OUT_DIR = ROOT / "sample_outputs"
VIDEOS = [
    "pedestrians_cctv.mp4",
    "pedestrians_crosswalk.mp4",
    "pedestrians_mall.mp4",
    "sports_soccer.mp4",
    "sports_basketball.mp4",
]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    model = load_model()
    rows = []

    for video_name in VIDEOS:
        in_path = VIDEO_DIR / video_name
        for tracker in TRACKERS:
            tracker_tag = tracker.split(".")[0]
            out_path = OUT_DIR / f"{Path(video_name).stem}_{tracker_tag}.mp4"
            result = track_video(model, in_path, out_path, tracker=tracker)
            class_summary = ", ".join(f"{k}={v}" for k, v in result.class_counts.items()) or "none"
            print(f"{video_name:<28} {tracker_tag:<10} "
                  f"{len(result.unique_ids):>3} unique  ({class_summary})  "
                  f"{result.elapsed_s:.1f}s")
            rows.append((video_name, tracker_tag, result.n_frames,
                        len(result.unique_ids), class_summary, f"{result.elapsed_s:.1f}s"))

    lines = ["| Video | Tracker | Frames | Unique objects | Per-class | Time |",
             "|---|---|---|---|---|---|"]
    for video_name, tracker_tag, n_frames, n_unique, class_summary, elapsed in rows:
        lines.append(f"| {video_name} | {tracker_tag} | {n_frames} | {n_unique} | "
                     f"{class_summary} | {elapsed} |")
    (OUT_DIR / "tracking_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'tracking_results.md'}")


if __name__ == "__main__":
    main()
