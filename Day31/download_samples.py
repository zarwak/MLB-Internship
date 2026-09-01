"""
Fetch the 5 sample traffic videos used by the Day-31 vehicle counting project.

All 5 are real Pexels footage (Pexels License - free for commercial/personal
use, no attribution required, fine to commit into a public repo), fetched at
their SD 960x540 encode - same sourcing approach as Day30's
download_samples.py. Picked for a mix of camera angles (elevated/oblique
highway views + a static urban intersection) and vehicle classes (the
intersection clip is the only one with motorcycles - highway footage is
almost entirely cars/buses/trucks).

Note: two straight-down AERIAL/nadir clips were tried first and dropped -
YOLOv8n's COCO weights detected zero vehicles on either (0/30 frames tested,
even at conf=0.1). COCO's "car"/"truck"/"bus" examples are essentially all
ground-level or oblique photos, so a pretrained detector doesn't generalize
to a straight-overhead viewpoint - a real, measured limitation, not a config
mistake (see README "Challenges"). Replaced with two more oblique/eye-level
highway clips, which detect normally.

Run:  python download_samples.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
VIDEO_DIR = ROOT / "sample_videos"

# (filename, remote_url, scene, note)
PEXELS_VIDEOS = [
    ("highway_evening.mp4",
     "https://videos.pexels.com/video-files/36369997/15425609_960_540_25fps.mp4",
     "highway, oblique angle", "evening traffic on a busy urban highway"),
    ("highway_cars_buses.mp4",
     "https://videos.pexels.com/video-files/28349571/12365005_960_540_30fps.mp4",
     "highway, side angle", "cars and buses on a highway (Sururi Balliday)"),
    ("highway_fast_paced.mp4",
     "https://videos.pexels.com/video-files/38854050/16516326_960_540_25fps.mp4",
     "highway, oblique angle", "fast-paced highway traffic"),
    ("highway_many_cars.mp4",
     "https://videos.pexels.com/video-files/18553048/18553048-sd_960_540_30fps.mp4",
     "highway, side angle", "dense car traffic"),
    ("urban_intersection_motorcycles.mp4",
     "https://videos.pexels.com/video-files/36306512/15397234_960_540_60fps.mp4",
     "urban intersection", "static camera, mixed traffic including motorcycles"),
]


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name}")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "day31-vehicle-counting"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    print(f"  saved {dest.name:<38} ({dest.stat().st_size / 1024:6.1f} KB)")


def main() -> int:
    VIDEO_DIR.mkdir(exist_ok=True)
    failures = []

    print(f"Downloading {len(PEXELS_VIDEOS)} Pexels traffic videos")
    for name, url, scene, note in PEXELS_VIDEOS:
        print(f"  {name}  [{scene}]  {note}")
        try:
            _download(url, VIDEO_DIR / name)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  FAILED {name}: {exc}")
            failures.append(name)

    videos = sorted(p.name for p in VIDEO_DIR.iterdir() if p.suffix.lower() == ".mp4")
    print(f"\nDone. {len(videos)} videos in sample_videos/.")
    if failures:
        print(f"{len(failures)} item(s) failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
