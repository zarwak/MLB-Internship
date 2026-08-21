"""
Fetch the 10 sample image pairs used by the Day-25 feature matching project.

Every image comes from the OpenCV repository's public `samples/data` folder
(BSD-licensed test data). They are real photographs / renders, not synthetic
warps, so the matching results below are honest ones.

Run:  python download_samples.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data"
SAMPLE_DIR = Path(__file__).parent / "sample_images"

# (pair_id, category, remote_a, remote_b, what changes between the two views)
PAIRS = [
    ("pair01_box_product",   "Product / object in clutter", "box.png",              "box_in_scene.png",      "object photographed alone, then lying in a cluttered scene"),
    ("pair02_graffiti_wall", "Building wall",               "graf1.png",            "graf3.png",             "strong viewpoint change (~40 degrees) across a graffiti wall"),
    ("pair03_leuven_landmark", "Landmark",                  "leuvenA.jpg",          "leuvenB.jpg",           "same Leuven town-hall facade under very different exposure"),
    ("pair04_aloe_plant",    "Object, two angles",          "aloeL.jpg",            "aloeR.jpg",             "stereo pair: left and right camera of the same plant"),
    ("pair05_suzanne_3d",    "Object, two angles",          "Blender_Suzanne1.jpg", "Blender_Suzanne2.jpg",  "same 3D model rendered from two different camera angles"),
    ("pair06_rubber_whale",  "Product / toys",              "rubberwhale1.png",     "rubberwhale2.png",      "small object motion between two consecutive frames"),
    ("pair07_text_page",     "Book cover / text",           "imageTextN.png",       "imageTextR.png",        "the same printed page, second one rotated"),
    ("pair08_aerial",        "Landmark from the air",       "aero1.jpg",            "aero3.jpg",             "two aerial passes over the same ground"),
    ("pair09_indoor_stereo", "Indoor scene",                "left.jpg",             "right.jpg",             "stereo pair of the same room"),
    ("pair10_basketball",    "Scene, camera motion",        "basketball1.png",      "basketball2.png",       "two consecutive frames with player and camera motion"),
]


def _download(remote_name: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name}")
        return
    url = f"{BASE_URL}/{remote_name}"
    req = urllib.request.Request(url, headers={"User-Agent": "day25-feature-matching"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    print(f"  saved {dest.name:<34} ({dest.stat().st_size / 1024:6.1f} KB)")


def main() -> int:
    SAMPLE_DIR.mkdir(exist_ok=True)
    print(f"Downloading 10 image pairs into {SAMPLE_DIR}\n")

    failures = []
    for pair_id, category, remote_a, remote_b, note in PAIRS:
        print(f"{pair_id}  [{category}]")
        for suffix, remote in (("a", remote_a), ("b", remote_b)):
            dest = SAMPLE_DIR / f"{pair_id}_{suffix}{Path(remote).suffix}"
            try:
                _download(remote, dest)
            except (urllib.error.URLError, OSError) as exc:
                print(f"  FAILED {remote}: {exc}")
                failures.append(remote)
        print(f"  -> {note}\n")

    if failures:
        print(f"{len(failures)} file(s) failed: {', '.join(failures)}")
        return 1

    have = sorted(p.name for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg"})
    print(f"Done. {len(have)} images ({len(have) // 2} pairs) in sample_images/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
