"""
Day 27 practice 3 - run YOLO object detection on the sample videos.

Runs YOLO11n frame-by-frame on every video in sample_videos/, writes an
annotated mp4 for each into sample_outputs/detected_videos/, and prints
aggregate per-video stats (frames, objects seen, most common classes).

Run:  python coding_practice/03_detect_videos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection import detect_video, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "sample_videos"
OUT_DIR = ROOT / "sample_outputs" / "detected_videos"
CONF = 0.25


def main() -> None:
    videos = sorted(p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in {".avi", ".mp4", ".mov"})
    if not videos:
        print("No videos found in sample_videos/. Run download_samples.py first.")
        return

    model = load_model()
    print(f"Loaded model, {len(videos)} video(s) to process (conf >= {CONF})\n")

    for path in videos:
        out_path = OUT_DIR / f"{path.stem}_detected.mp4"

        def progress(done: int, total: int) -> None:
            if done % 25 == 0 or done == total:
                print(f"\r  frame {done}/{total}", end="", flush=True)

        print(f"{path.name}")
        result = detect_video(model, path, out_path, conf=CONF, progress_cb=progress)
        print()

        counts = result.class_counts
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        top_str = ", ".join(f"{name}x{n}" for name, n in top) or "(nothing detected)"

        print(f"  {result.n_frames} frames @ {result.fps:.1f} fps, "
              f"{result.frames_with_detection} frame(s) with >=1 detection, "
              f"{result.elapsed_s:.1f}s to process")
        print(f"  classes seen (detection count, not unique objects): {top_str}")
        print(f"  saved -> {out_path}\n")


if __name__ == "__main__":
    main()
