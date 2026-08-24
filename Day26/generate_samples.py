"""
Generate the 15+ sample images used by the Day-26 segmentation project.

Real "documents with uneven lighting" and "objects with soft shadows" photo
sets are hard to source reliably from a public URL (licensing, category
labels, network flakiness during grading). So instead of downloading, this
script *renders* the four required categories directly with Pillow/OpenCV,
which guarantees:

  - exact category coverage (documents, plain-background objects, uneven
    lighting, shadows) instead of hoping a downloaded set happens to match
  - deterministic, reproducible output (fixed random seed)
  - no network dependency at grading time

Run:  python generate_samples.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAMPLE_DIR = Path(__file__).parent / "sample_images"
RNG = random.Random(26)
NP_RNG = np.random.default_rng(26)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _save(img: Image.Image, name: str) -> None:
    path = SAMPLE_DIR / name
    img.convert("RGB").save(path, quality=95)
    print(f"  wrote {name:<32} {img.size[0]}x{img.size[1]}")


def _add_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    noise = NP_RNG.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Category 1: documents - dark text on a light page, the classic
# binarize-a-scan case that Otsu was designed for.
# ---------------------------------------------------------------------------

LOREM = [
    "INVOICE #A-2049",
    "Segmentation Report - Day 26",
    "Section 1: Introduction to thresholding",
    "The quick brown fox jumps over the lazy dog.",
    "Binary, Adaptive and Otsu methods compared.",
    "Signed: ___________________  Date: __/__/____",
]


def make_document(name: str, *, stained: bool, tinted_bg: bool, blurred: bool) -> None:
    w, h = 900, 1160
    bg = (245, 240, 225) if tinted_bg else (252, 252, 250)
    img = Image.new("L", (w, h), color=bg[0])
    draw = ImageDraw.Draw(img)

    y = 90
    draw.text((70, 40), LOREM[1] if "invoice" not in name else LOREM[0],
              font=_font(40), fill=20)
    y = 160
    body_font = _font(24)
    for line_no in range(18):
        text = LOREM[(line_no + 2) % len(LOREM)] if line_no % 4 == 0 else \
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit sed."
        draw.text((70, y), text, font=body_font, fill=25)
        y += 42

    if stained:
        # A soft brown "coffee stain" blob - the classic reason a single
        # global threshold fails on a real scanned document.
        stain = Image.new("L", (w, h), 0)
        sdraw = ImageDraw.Draw(stain)
        cx, cy = int(w * 0.72), int(h * 0.35)
        for r, val in [(140, 40), (100, 70), (60, 110)]:
            sdraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=val)
        stain = stain.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(25))
        arr = np.asarray(img).astype(np.int16)
        stain_arr = np.asarray(stain).astype(np.int16)
        arr = np.clip(arr - stain_arr // 2, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    if blurred:
        img = img.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(1.2))

    img = _add_gaussian_noise(img.convert("L"), sigma=4).convert("L")
    _save(img, name)


def make_document_uneven_light(name: str) -> None:
    """A page lit from one corner - defeats binary/Otsu, needs adaptive."""
    w, h = 900, 1160
    img = Image.new("L", (w, h), color=250)
    draw = ImageDraw.Draw(img)
    y = 160
    body_font = _font(24)
    draw.text((70, 40), "Section 3: Adaptive Thresholding", font=_font(38), fill=15)
    for line_no in range(18):
        draw.text((70, y), "Uneven lighting text line for adaptive threshold demo.",
                  font=body_font, fill=15)
        y += 42

    # Multiplicative lighting gradient from bright (top-left) to dark
    # (bottom-right), simulating a desk lamp / window light on one side.
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = 1.0 - 0.65 * ((xx / w) * 0.6 + (yy / h) * 0.4)
    arr = np.asarray(img).astype(np.float32) * gradient
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    _save(img, name)


# ---------------------------------------------------------------------------
# Category 2: simple objects on a plain background - the easy case where
# binary thresholding alone should already work well.
# ---------------------------------------------------------------------------

SHAPES = ["circle", "square", "star", "triangle"]


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, cx: int, cy: int, r: int, fill) -> None:
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
    elif shape == "star":
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rad = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        draw.polygon(pts, fill=fill)


def make_plain_object(name: str, shape: str, dark_on_light: bool = True) -> None:
    w, h = 800, 800
    bg, fg = (235, 40) if dark_on_light else (30, 220)
    img = Image.new("L", (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    _draw_shape(draw, shape, w // 2, h // 2, 220, fg)
    img = _add_gaussian_noise(img, sigma=3)
    _save(img, name)


def make_overlapping_objects(name: str) -> None:
    """Two touching circles - the classic watershed 'split touching blobs' demo."""
    w, h = 800, 500
    img = Image.new("L", (w, h), color=235)
    draw = ImageDraw.Draw(img)
    centers = [(260, 250, 150), (430, 250, 150), (560, 180, 90)]
    for cx, cy, r in centers:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=45)
    img = _add_gaussian_noise(img, sigma=3)
    _save(img, name)


# ---------------------------------------------------------------------------
# Category 3: uneven lighting on an object photo - defeats a single global
# threshold, motivates adaptive thresholding.
# ---------------------------------------------------------------------------

def make_uneven_light_object(name: str, shape: str) -> None:
    w, h = 800, 800
    img = Image.new("L", (w, h), color=230)
    draw = ImageDraw.Draw(img)
    _draw_shape(draw, shape, w // 2, h // 2, 220, 50)

    yy, xx = np.mgrid[0:h, 0:w]
    # Radial spotlight: bright centre-left, falling off toward the edges.
    lx, ly = RNG.choice([0.2, 0.8]) * w, RNG.choice([0.2, 0.8]) * h
    dist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2) / (0.75 * max(w, h))
    gradient = np.clip(1.25 - dist, 0.35, 1.25)
    arr = np.asarray(img).astype(np.float32) * gradient
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = _add_gaussian_noise(img, sigma=3)
    _save(img, name)


# ---------------------------------------------------------------------------
# Category 4: objects with cast shadows - the case that trips up naive
# background subtraction and needs morphology clean-up after thresholding.
# ---------------------------------------------------------------------------

def make_shadowed_object(name: str, shape: str) -> None:
    w, h = 800, 800
    img = Image.new("L", (w, h), color=235)
    draw = ImageDraw.Draw(img)

    # Soft elliptical shadow, offset down-right of the object.
    shadow = Image.new("L", (w, h), 0)
    sdraw = ImageDraw.Draw(shadow)
    sdraw.ellipse([w // 2 - 150, h // 2 + 60, w // 2 + 260, h // 2 + 260], fill=90)
    shadow = shadow.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(30))
    arr = np.asarray(img).astype(np.int16) - np.asarray(shadow).astype(np.int16)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(img)
    _draw_shape(draw, shape, w // 2, h // 2, 200, 35)
    img = _add_gaussian_noise(img, sigma=3)
    _save(img, name)


def main() -> None:
    SAMPLE_DIR.mkdir(exist_ok=True)
    print(f"Generating sample images into {SAMPLE_DIR}\n")

    print("documents:")
    make_document("doc01_invoice_clean.jpg", stained=False, tinted_bg=False, blurred=False)
    make_document("doc02_report_tinted_bg.jpg", stained=False, tinted_bg=True, blurred=False)
    make_document("doc03_report_coffee_stain.jpg", stained=True, tinted_bg=False, blurred=False)
    make_document("doc04_report_scanned_blur.jpg", stained=False, tinted_bg=True, blurred=True)
    make_document_uneven_light("doc05_report_uneven_light.jpg")

    print("plain-background objects:")
    for i, shape in enumerate(SHAPES, start=1):
        make_plain_object(f"obj0{i}_{shape}_on_white.jpg", shape, dark_on_light=True)
    make_plain_object("obj05_circle_light_on_dark.jpg", "circle", dark_on_light=False)
    make_overlapping_objects("obj06_overlapping_circles_watershed.jpg")

    print("uneven lighting:")
    for i, shape in enumerate(["circle", "square", "star"], start=1):
        make_uneven_light_object(f"light0{i}_{shape}_spotlight.jpg", shape)

    print("shadows:")
    for i, shape in enumerate(["circle", "square", "triangle"], start=1):
        make_shadowed_object(f"shadow0{i}_{shape}_cast_shadow.jpg", shape)

    total = sorted(SAMPLE_DIR.glob("*.jpg"))
    print(f"\nDone. {len(total)} images in sample_images/.")


if __name__ == "__main__":
    main()
