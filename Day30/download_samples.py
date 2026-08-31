"""
Fetch the 5 sample tracking videos used by the Day-30 object tracking project.

Categories (see README "Dataset" / brainstorming discussion): we deliberately
picked **people walking** and **sports** over traffic/parking-lot, because
those two categories are the ones that actually stress-test the thing this
task cares about - IDs surviving objects crossing paths, occluding each
other, and re-entering frame. A parking lot with mostly-parked cars barely
exercises a tracker at all.

Sources (all real footage, permissive licenses - fine to commit into a
public repo):

  - vtest.avi              OpenCV's classic pedestrian-CCTV sample (BSD
                            licensed test data, samples/data/vtest.avi) -
                            same source Day27 used for detection. Runs ~80s;
                            trimmed to 15s here (brief asks for "short" clips).
  - 4 Pexels videos         real stock footage, Pexels License (free for
                            commercial/personal use, no attribution
                            required). Fetched at their SD 960x540 encode
                            (the direct CDN link Pexels' own "Free Download"
                            size picker resolves to) - already 14-19s long,
                            no trimming needed.

Run:  python download_samples.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).parent
VIDEO_DIR = ROOT / "sample_videos"

OPENCV_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"

# (filename, remote_url, category, one-line note)
PEXELS_VIDEOS = [
    ("pedestrians_crosswalk.mp4",
     "https://videos.pexels.com/video-files/38298405/16262173_960_540_25fps.mp4",
     "people walking", "busy Tokyo-style crosswalk, dense crowd crossing (Musa Ortac)"),
    ("pedestrians_mall.mp4",
     "https://videos.pexels.com/video-files/15760904/15760904-sd_960_540_24fps.mp4",
     "people walking", "shopping mall, crowd walking in both directions (khanhhoangminh)"),
    ("sports_soccer.mp4",
     "https://videos.pexels.com/video-files/6077718/6077718-sd_960_540_25fps.mp4",
     "sports", "soccer players training, frequent overlap (Tima Miroshnichenko)"),
    ("sports_basketball.mp4",
     "https://videos.pexels.com/video-files/5586533/5586533-sd_960_540_25fps.mp4",
     "sports", "basketball players on court, frequent overlap (Tima Miroshnichenko)"),
]

TRIMMED_VTEST_SECONDS = 15  # vtest.avi alone runs ~80s


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name}")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "day30-object-tracking"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    print(f"  saved {dest.name:<28} ({dest.stat().st_size / 1024:6.1f} KB)")


def trim_video(in_path: Path, out_path: Path, seconds: float) -> None:
    """Cut the first `seconds` off a video (used only for vtest.avi, which
    runs much longer than the brief's "short clip" ask)."""
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  already have {out_path.name}")
        return
    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(seconds * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"  saved  {out_path.name:<28} ({out_path.stat().st_size / 1024:6.1f} KB, "
          f"first {seconds}s of {in_path.name})")


def main() -> int:
    VIDEO_DIR.mkdir(exist_ok=True)
    failures = []

    print("1) Downloading OpenCV's pedestrian-CCTV sample (vtest.avi)")
    full_path = VIDEO_DIR / "_vtest_full.avi"
    try:
        _download(OPENCV_URL, full_path)
    except (urllib.error.URLError, OSError) as exc:
        print(f"  FAILED vtest.avi: {exc}")
        failures.append("vtest.avi")

    print(f"\n1b) Trimming to a {TRIMMED_VTEST_SECONDS}s clip (the one the app/scripts actually use)")
    if full_path.exists():
        trim_video(full_path, VIDEO_DIR / "pedestrians_cctv.mp4", TRIMMED_VTEST_SECONDS)
        full_path.unlink()  # only the trimmed clip is kept as a project deliverable
    else:
        failures.append("pedestrians_cctv.mp4")

    print(f"\n2) Downloading {len(PEXELS_VIDEOS)} Pexels videos (already short, no trimming needed)")
    for name, url, category, note in PEXELS_VIDEOS:
        try:
            _download(url, VIDEO_DIR / name)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  FAILED {name}: {exc}")
            failures.append(name)

    videos = sorted(p.name for p in VIDEO_DIR.iterdir() if p.suffix.lower() in {".avi", ".mp4"})
    print(f"\nDone. {len(videos)} videos in sample_videos/.")
    if failures:
        print(f"{len(failures)} item(s) failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
