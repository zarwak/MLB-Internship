"""
Fetch the sample images and videos used by the Day-27 object detection project.

Sources (all real photographs, not synthetic renders - YOLO is trained on
real photos, so synthetic geometric shapes like Day-26's samples wouldn't be
detected as anything):

  - bus.jpg / zidane.jpg   copied straight out of the installed `ultralytics`
                           package (it ships its own two canonical demo
                           images) - zero network dependency for these two.
  - 8 images               real photographs from the COCO val2017 set
                           (images.cocodataset.org), the same dataset YOLO's
                           COCO-pretrained weights were trained on.
  - 2 images                OpenCV's public samples/data folder (BSD-licensed
                           test data), for extra class variety (person, ball).
  - 2 videos               vtest.avi, OpenCV's classic pedestrian-CCTV sample
                           (real footage, person detection), plus a slideshow
                           video assembled from several of the photos above
                           (see make_slideshow_video() / README "Challenges"
                           for why the second clip is synthesized rather than
                           downloaded).

Run:  python download_samples.py
"""

from __future__ import annotations

import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import ultralytics

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "sample_images"
VIDEO_DIR = ROOT / "sample_videos"

OPENCV_BASE = "http://raw.githubusercontent.com/opencv/opencv/master/samples/data"
COCO_BASE = "http://images.cocodataset.org/val2017"

# (filename, remote_url, one-line note - filled in after actually looking at
# the detections, see README "Dataset")
COCO_IMAGES = [
    ("coco_cats_couch.jpg", f"{COCO_BASE}/000000039769.jpg", "two cats + remotes on a couch"),
    ("coco_skier.jpg", f"{COCO_BASE}/000000000785.jpg", "person on skis"),
    ("coco_scene_632.jpg", f"{COCO_BASE}/000000000632.jpg", "COCO val2017 photo"),
    ("coco_scene_724.jpg", f"{COCO_BASE}/000000000724.jpg", "COCO val2017 photo"),
    ("coco_scene_776.jpg", f"{COCO_BASE}/000000000776.jpg", "COCO val2017 photo"),
    ("coco_scene_802.jpg", f"{COCO_BASE}/000000000802.jpg", "COCO val2017 photo"),
    ("coco_scene_872.jpg", f"{COCO_BASE}/000000000872.jpg", "COCO val2017 photo"),
    ("coco_scene_885.jpg", f"{COCO_BASE}/000000000885.jpg", "COCO val2017 photo"),
]

OPENCV_IMAGES = [
    ("opencv_messi.jpg", f"{OPENCV_BASE}/messi5.jpg", "footballer + ball"),
    ("opencv_basketball.png", f"{OPENCV_BASE}/basketball1.png", "players on court"),
]

VIDEO_SOURCES = [
    ("vtest_pedestrians_full.avi", f"{OPENCV_BASE}/vtest.avi", "CCTV footage of people walking"),
]
TRIMMED_VIDEO_SECONDS = 15  # brief asks for "short" videos; vtest.avi alone runs ~80s

SLIDESHOW_SOURCE_IMAGES = [
    "coco_cats_couch.jpg", "coco_skier.jpg", "opencv_messi.jpg",
    "coco_scene_632.jpg", "coco_scene_776.jpg", "coco_scene_802.jpg",
]


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name}")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "day27-object-detection"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    print(f"  saved {dest.name:<28} ({dest.stat().st_size / 1024:6.1f} KB)")


def copy_ultralytics_assets() -> list[str]:
    """bus.jpg / zidane.jpg ship inside the ultralytics package itself."""
    assets_dir = Path(ultralytics.__file__).parent / "assets"
    copied = []
    for name in ("bus.jpg", "zidane.jpg"):
        src = assets_dir / name
        dest = IMAGE_DIR / f"ultralytics_{name}"
        if src.exists():
            shutil.copy(src, dest)
            copied.append(dest.name)
            print(f"  copied {dest.name:<28} (from installed ultralytics package)")
        else:
            print(f"  SKIPPED {name}: not found in installed ultralytics package")
    return copied


def trim_video(in_path: Path, out_path: Path, seconds: float) -> None:
    """Cut the first `seconds` off a video (brief asks for "short" clips;
    vtest.avi on its own runs ~80s)."""
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


def make_slideshow_video(image_paths: list[Path], out_path: Path,
                          seconds_per_image: float = 2.5, fps: int = 20,
                          size: tuple[int, int] = (960, 540)) -> None:
    """Assemble a short slideshow video (gentle zoom-in per photo) out of
    real downloaded photographs. Used as the second sample video instead of
    a second download - see README 'Challenges' for why."""
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  already have {out_path.name}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    n_frames = int(seconds_per_image * fps)

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = max(size[0] / w, size[1] / h)
        img = cv2.resize(img, (round(w * scale), round(h * scale)))
        for i in range(n_frames):
            zoom = 1.0 + 0.12 * (i / n_frames)  # slow zoom-in, 1.0x -> 1.12x
            zh, zw = img.shape[:2]
            ch, cw = round(zh / zoom), round(zw / zoom)
            y0, x0 = (zh - ch) // 2, (zw - cw) // 2
            crop = img[y0:y0 + ch, x0:x0 + cw]
            frame = cv2.resize(crop, size)
            writer.write(frame)
    writer.release()
    print(f"  built  {out_path.name:<28} ({out_path.stat().st_size / 1024:6.1f} KB, "
          f"{len(image_paths)} photos x {seconds_per_image}s)")


def main() -> int:
    IMAGE_DIR.mkdir(exist_ok=True)
    VIDEO_DIR.mkdir(exist_ok=True)
    failures = []

    print(f"1) Copying ultralytics' bundled demo images into {IMAGE_DIR}")
    copy_ultralytics_assets()

    print(f"\n2) Downloading {len(COCO_IMAGES)} real photos from COCO val2017 "
          f"(the dataset YOLO's weights were trained on)")
    for name, url, note in COCO_IMAGES:
        try:
            _download(url, IMAGE_DIR / name)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  FAILED {name}: {exc}")
            failures.append(name)

    print(f"\n3) Downloading {len(OPENCV_IMAGES)} photos from OpenCV's public samples/data")
    for name, url, note in OPENCV_IMAGES:
        try:
            _download(url, IMAGE_DIR / name)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  FAILED {name}: {exc}")
            failures.append(name)

    print(f"\n4) Downloading {len(VIDEO_SOURCES)} sample video(s)")
    for name, url, note in VIDEO_SOURCES:
        try:
            _download(url, VIDEO_DIR / name)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  FAILED {name}: {exc}")
            failures.append(name)

    print(f"\n4b) Trimming to a {TRIMMED_VIDEO_SECONDS}s clip (the one the app/scripts actually use)")
    full_path = VIDEO_DIR / VIDEO_SOURCES[0][0]
    if full_path.exists():
        trim_video(full_path, VIDEO_DIR / "vtest_pedestrians.mp4", TRIMMED_VIDEO_SECONDS)
        full_path.unlink()  # only the trimmed clip is kept as a project deliverable
    else:
        failures.append("vtest_pedestrians.mp4")

    print("\n5) Building a second sample video (slideshow of real photos above)")
    source_paths = [IMAGE_DIR / n for n in SLIDESHOW_SOURCE_IMAGES if (IMAGE_DIR / n).exists()]
    if source_paths:
        make_slideshow_video(source_paths, VIDEO_DIR / "coco_slideshow.mp4")
    else:
        print("  SKIPPED: none of the slideshow source images downloaded successfully")
        failures.append("coco_slideshow.mp4")

    images = sorted(p.name for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    videos = sorted(p.name for p in VIDEO_DIR.iterdir() if p.suffix.lower() in {".avi", ".mp4"})
    print(f"\nDone. {len(images)} images in sample_images/, {len(videos)} videos in sample_videos/.")
    if failures:
        print(f"{len(failures)} item(s) failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
