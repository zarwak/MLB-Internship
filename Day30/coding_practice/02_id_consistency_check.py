"""
Day 30 coding practice - heuristic ID-switch counter.

There's no ground-truth ID annotation for these clips, so this isn't a
formal MOTA/IDF1 score. Instead: for each track ID, find the last frame it
appeared in. Count it as a suspected "ID switch" if, within the next
GAP_FRAMES frames, a *new* track ID (one that had never appeared before)
first appears whose class matches and whose box center is within
DIST_THRESHOLD_FRAC of the frame's shorter side from where the old ID was
last seen. That pattern - one ID vanishing right as a nearby new one of the
same class appears - is what a tracker losing an object and relabeling it
looks like.

Run: python coding_practice/02_id_consistency_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2

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
GAP_FRAMES = 5
DIST_THRESHOLD_FRAC = 0.08


def _centroid(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def count_suspected_switches(tracks_per_frame: list[list], frame_shape_short_side: float) -> int:
    last_seen: dict[int, tuple[int, str, tuple[float, float]]] = {}
    first_seen_frame: dict[int, int] = {}
    threshold = frame_shape_short_side * DIST_THRESHOLD_FRAC
    switches = 0

    for frame_idx, boxes in enumerate(tracks_per_frame):
        for t in boxes:
            if t.track_id not in first_seen_frame:
                first_seen_frame[t.track_id] = frame_idx
                # a brand-new ID: check if it lines up with a recently-lost one
                cx, cy = _centroid(t.box)
                for old_id, (old_frame, old_class, (ox, oy)) in list(last_seen.items()):
                    if old_class != t.class_name:
                        continue
                    if 0 < frame_idx - old_frame <= GAP_FRAMES:
                        if math.hypot(cx - ox, cy - oy) <= threshold:
                            switches += 1
                            del last_seen[old_id]
                            break
            last_seen[t.track_id] = (frame_idx, t.class_name, _centroid(t.box))

    return switches


def main() -> None:
    model = load_model()
    lines = ["| Video | Tracker | Unique IDs | Suspected ID switches |",
             "|---|---|---|---|"]

    for video_name in VIDEOS:
        in_path = VIDEO_DIR / video_name
        for tracker in TRACKERS:
            tracker_tag = tracker.split(".")[0]
            out_path = OUT_DIR / f"_scratch_{Path(video_name).stem}_{tracker_tag}.mp4"
            result = track_video(model, in_path, out_path, tracker=tracker)
            cap = cv2.VideoCapture(str(in_path))
            short_side = min(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            switches = count_suspected_switches(result.tracks_per_frame, short_side)
            print(f"{video_name:<28} {tracker_tag:<10} "
                  f"{len(result.unique_ids):>3} unique  {switches} suspected switch(es)")
            lines.append(f"| {video_name} | {tracker_tag} | {len(result.unique_ids)} | {switches} |")
            out_path.unlink(missing_ok=True)  # this script only needs the stats, not the video

    lines.append("")
    lines.append("Heuristic, not a formal MOTA/IDF1 score - no ground-truth ID "
                 "annotations exist for these clips. A 'suspected switch' is a new "
                 "track ID appearing within "
                 f"{GAP_FRAMES} frames and {DIST_THRESHOLD_FRAC:.0%} of the frame's "
                 "shorter side of where a same-class ID was last seen.")
    (OUT_DIR / "id_consistency.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'id_consistency.md'}")


if __name__ == "__main__":
    main()
