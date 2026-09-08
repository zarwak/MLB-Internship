"""
Fetch and trim the 6 sample traffic videos used by the Day-36 violation
detection project. All 6 are free stock clips (Pixabay License - free for
commercial and personal use, no attribution required), picked after
measuring that a plain COCO-pretrained YOLOv8n reliably detects vehicles in
them - see README.md "Real challenges faced" for several rejected
candidates (steep drone highway shots the model hallucinated as a "train";
a tilt-shift/miniature-effect clip; night hyperlapse clips whose long
exposure turns every car into a light streak) and why moderate-elevation
or street-level fixed-camera footage was chosen instead.

Each video is trimmed to ~8-14s and transcoded to 1280px-wide H.264 so the
shipped repo stays small and every clip decodes consistently regardless of
its original resolution/frame rate.

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

# (filename, remote_url, trim_seconds, scene)
SOURCES = [
    ("madrid_intersection.mp4",
     "https://cdn.pixabay.com/video/2021/01/31/63621-506830674_large.mp4",
     12,
     "street-level, multi-lane approach to a Madrid junction"),
    ("hillside_street.mp4",
     "https://cdn.pixabay.com/video/2019/10/25/28293-369325244_medium.mp4",
     12,
     "street-level, two-way palm-lined hill street"),
    ("bangkok_boulevard.mp4",
     "https://cdn.pixabay.com/video/2020/01/06/30949-383991398_large.mp4",
     12,
     "elevated fixed camera over a divided Bangkok boulevard"),
    ("river_bridge_junction.mp4",
     "https://cdn.pixabay.com/video/2022/11/23/140222-774508021_medium.mp4",
     14,
     "aerial view of a Dutch river-town bridge + junction"),
    ("nhatrang_street.mp4",
     "https://cdn.pixabay.com/video/2020/04/12/35835-408654119_large.mp4",
     8,
     "street-level, motorcycle-heavy Vietnamese street (clip is only 8s total)"),
    ("atlanta_commute.mp4",
     "https://cdn.pixabay.com/video/2024/04/26/209541_large.mp4",
     12,
     "dusk drone shot of an Atlanta commuter highway"),
]


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "day36-traffic-violation"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        dest.write_bytes(resp.read())


def _transcode(src: Path, dest: Path, trim_seconds: int) -> None:
    args = [FFMPEG, "-y", "-ss", "0", "-i", str(src), "-t", str(trim_seconds),
             "-vf", "scale=1280:-2", "-r", "25", "-c:v", "libx264", "-preset", "medium",
             "-crf", "22", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dest)]
    subprocess.run(args, check=True, capture_output=True)


def main() -> int:
    VIDEO_DIR.mkdir(exist_ok=True)
    failures = []

    print(f"Downloading and trimming {len(SOURCES)} sample videos")
    for name, url, trim_seconds, scene in SOURCES:
        dest = VIDEO_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  already have {name}")
            continue
        print(f"  {name}  [{scene}]")
        raw = VIDEO_DIR / f"_raw_{name}"
        try:
            _download(url, raw)
            _transcode(raw, dest, trim_seconds)
            print(f"  saved {name:<28} ({dest.stat().st_size / 1024:6.1f} KB)")
        except (urllib.error.URLError, OSError, subprocess.CalledProcessError) as exc:
            print(f"  FAILED {name}: {exc}")
            failures.append(name)
        finally:
            raw.unlink(missing_ok=True)

    videos = sorted(p.name for p in VIDEO_DIR.iterdir() if p.suffix.lower() == ".mp4")
    print(f"\nDone. {len(videos)} videos in sample_videos/.")
    print("Note: *_config.json presets (direction/zone/direction_roi) are already committed - "
          "they were hand-calibrated against each clip's first frames and won't regenerate from this script.")
    if failures:
        print(f"{len(failures)} item(s) failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
