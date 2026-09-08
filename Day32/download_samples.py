"""
Fetch 2 of the 4 sample parking-lot videos used by the Day-32 occupancy
project (the 2 general-purpose-detector ones - courtyard_lot.mp4,
mall_curbside_lot.mp4). The 2 aerial_* videos are free stock footage from
the same kind of search but their exact source URLs weren't captured when
they were added - see README.md "The 4 sample videos".

Both videos below are free stock clips (Pixabay License - free for commercial and
personal use, no attribution required). Picked after evaluating 40+
candidates across Pexels and Pixabay for the property that matters most
for a fixed-grid-of-space-polygons system and turned out to be the
hardest to find: a genuinely, measurably STATIC camera - no pan, tilt, or
zoom. See README.md "Challenges" for the full account of what that search
found - most "parking lot" stock footage is either continuously-moving
drone cinematography, or a straight-down nadir angle that defeats a
COCO-pretrained detector (same finding Day31 made for traffic footage).
A third candidate (a clear, well-marked, oblique drone shot) was dropped
after review specifically because it had a slow zoom - correct occupancy
numbers throughout, but "fixed camera" was a hard requirement here, and it
didn't meet it.

Run:  python download_samples.py
"""

from __future__ import annotations

import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"

ROOT = Path(__file__).parent
VIDEO_DIR = ROOT / "sample_videos"

# (filename, remote_url, trim_frames_or_None, scene, note)
SOURCES = [
    ("courtyard_lot.mp4",
     "https://cdn.pixabay.com/video/2021/07/11/81014-574269765_large.mp4",
     None,
     "residential courtyard, elevated window/balcony view",
     "fully static for its whole 9s runtime - measured 0.0-0.1% frame-to-frame "
     "drift throughout via ORB feature matching"),
    ("mall_curbside_lot.mp4",
     "https://cdn.pixabay.com/video/2019/12/07/29947-378294525_large.mp4",
     300,
     "ground-level, mall curbside row",
     "fully static tripod shot; trimmed to the first 300 frames (10s) to "
     "match the app's own processing cap"),
]


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "day32-parking-occupancy"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())


def _transcode(src: Path, dest: Path, trim_frames: int | None) -> None:
    args = [FFMPEG, "-y", "-i", str(src)]
    if trim_frames is not None:
        args += ["-frames:v", str(trim_frames)]
    args += ["-vf", "scale=1280:-2", "-r", "30", "-c:v", "libx264", "-preset", "medium",
             "-crf", "24", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dest)]
    subprocess.run(args, check=True, capture_output=True)


def main() -> int:
    VIDEO_DIR.mkdir(exist_ok=True)
    failures = []

    print(f"Downloading and trimming {len(SOURCES)} sample videos")
    for name, url, trim_frames, scene, note in SOURCES:
        dest = VIDEO_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  already have {name}")
            continue
        print(f"  {name}  [{scene}]  {note}")
        raw = VIDEO_DIR / f"_raw_{name}"
        try:
            _download(url, raw)
            _transcode(raw, dest, trim_frames)
            print(f"  saved {name:<28} ({dest.stat().st_size / 1024:6.1f} KB)")
        except (urllib.error.URLError, OSError, subprocess.CalledProcessError) as exc:
            print(f"  FAILED {name}: {exc}")
            failures.append(name)
        finally:
            raw.unlink(missing_ok=True)

    videos = sorted(p.name for p in VIDEO_DIR.iterdir() if p.suffix.lower() == ".mp4")
    print(f"\nDone. {len(videos)} videos in sample_videos/.")
    print("Note: *_spaces.json layouts are already committed - they were hand-calibrated "
          "against each clip's first frame and won't regenerate from this script.")
    if failures:
        print(f"{len(failures)} item(s) failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
