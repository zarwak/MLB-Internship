"""
Day 29 - Curate a small, committable sample set from the (gitignored) dataset.

Copies a fixed, seeded selection of test-split images into sample_images/
(small enough to commit, so the app/deliverables work without needing the
full ~330 MB dataset/ or a Roboflow API key), and assembles a short
slideshow video out of them into sample_videos/ so the app's Video mode has
something to demo against out of the box.

Run: python prepare_samples.py
"""

import random
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
DATASET_TEST_IMAGES = ROOT / "dataset" / "test" / "images"
IMAGE_DIR = ROOT / "sample_images"
VIDEO_DIR = ROOT / "sample_videos"
N_SAMPLES = 18


def make_slideshow_video(image_paths: list[Path], out_path: Path,
                          seconds_per_image: float = 2.0, fps: int = 20,
                          size: tuple[int, int] = (960, 540)) -> None:
    """Assemble a short slideshow (gentle zoom-in per photo) out of the
    curated sample images, so Video mode has a ready-made demo clip."""
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
            zoom = 1.0 + 0.12 * (i / n_frames)
            zh, zw = img.shape[:2]
            ch, cw = round(zh / zoom), round(zw / zoom)
            y0, x0 = (zh - ch) // 2, (zw - cw) // 2
            crop = img[y0:y0 + ch, x0:x0 + cw]
            writer.write(cv2.resize(crop, size))
    writer.release()


def main():
    if not DATASET_TEST_IMAGES.exists():
        raise SystemExit(
            f"{DATASET_TEST_IMAGES} not found - run download_dataset.py first."
        )

    IMAGE_DIR.mkdir(exist_ok=True)
    VIDEO_DIR.mkdir(exist_ok=True)

    all_images = sorted(DATASET_TEST_IMAGES.glob("*.jpg"))
    random.seed(0)
    chosen = random.sample(all_images, min(N_SAMPLES, len(all_images)))

    print(f"Copying {len(chosen)} test images into {IMAGE_DIR}/ ...")
    for path in chosen:
        dest = IMAGE_DIR / path.name
        dest.write_bytes(path.read_bytes())

    slideshow_path = VIDEO_DIR / "road_damage_slideshow.mp4"
    print(f"Building {slideshow_path} ...")
    make_slideshow_video(chosen[:8], slideshow_path)

    print("Done.")


if __name__ == "__main__":
    main()
