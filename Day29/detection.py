"""
Day 29 - Custom road-damage YOLO detection core module.

Wraps our own trained model (best.pt, not a pretrained COCO checkpoint) for
image/video inference, with the same custom box drawing used in Day 27 (a
deterministic colour per class, a burned-in confidence/count/timing badge).
Both the Streamlit app and the coding_practice scripts import this module -
it holds all the actual logic, nothing lives only in app.py.
"""

from __future__ import annotations

import colorsys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    import imageio

from ultralytics import YOLO

DEFAULT_MODEL = "best.pt"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
MAX_SIDE = 960  # frames/images are downscaled to this before inference

_MODEL_CACHE: dict[str, YOLO] = {}
_MODEL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(name: str = DEFAULT_MODEL) -> YOLO:
    """Load (and cache) our trained YOLO model.

    Fuses the model eagerly, under a lock, right after loading. Ultralytics'
    predict() path also fuses lazily on first inference - fine for a single
    thread, but Streamlit can run two sessions concurrently against the same
    cached model, and two threads racing that lazy fuse() both pass its
    `hasattr(m, "bn")` check before either finishes deleting `bn`, so the
    second one crashes with AttributeError. Fusing here first makes that
    lazy fuse() a no-op (is_fused() already True) by the time predict() ever
    reaches it.
    """
    if name not in _MODEL_CACHE:
        with _MODEL_LOCK:
            if name not in _MODEL_CACHE:  # re-check: another thread may have loaded it while we waited
                model = YOLO(name)
                model.fuse(verbose=False)
                _MODEL_CACHE[name] = model
    return _MODEL_CACHE[name]


def model_info(model: YOLO) -> dict:
    n_params = sum(p.numel() for p in model.model.parameters())
    return {
        "class_count": len(model.names),
        "param_count": n_params,
        "classes": list(model.names.values()),
    }


# ---------------------------------------------------------------------------
# Colours - one visually distinct, deterministic colour per class id
# ---------------------------------------------------------------------------

def class_color(class_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour for a class id, spread via the golden angle
    so consecutive class ids (which often co-occur) don't land on similar hues."""
    hue = (class_id * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise IOError(f"could not read image: {path}")
    return image


def resize_max_side(image: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise IOError("could not encode image")
    return buf.tobytes()


def imwrite(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"could not write image: {path}")


# ---------------------------------------------------------------------------
# Detection results
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords


@dataclass
class ImageResult:
    annotated: np.ndarray
    detections: list[Detection]
    elapsed_ms: float
    conf_threshold: float

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(Counter(d.class_name for d in self.detections))


@dataclass
class VideoResult:
    out_path: Path
    detections_per_frame: list[list[Detection]]
    n_frames: int
    fps: float
    elapsed_s: float

    @property
    def class_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for frame_dets in self.detections_per_frame:
            counts.update(d.class_name for d in frame_dets)
        return dict(counts)

    @property
    def frames_with_detection(self) -> int:
        return sum(1 for frame_dets in self.detections_per_frame if frame_dets)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)


def _draw_box(image: np.ndarray, det: Detection) -> None:
    x1, y1, x2, y2 = det.box
    color = class_color(det.class_id)
    short_side = min(image.shape[:2])
    thickness = max(2, round(short_side / 350))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    label = f"{det.class_name} {det.confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, short_side / 900)
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
    ty1 = max(y1 - th - baseline - 6, 0)
    ty2 = ty1 + th + baseline + 6
    cv2.rectangle(image, (x1, ty1), (x1 + tw + 8, ty2), color, -1)
    text_color = _text_color_for(color)
    cv2.putText(image, label, (x1 + 4, ty2 - baseline - 2), font, font_scale,
                text_color, 1, cv2.LINE_AA)


def _draw_threshold_badge(image: np.ndarray, conf: float, n_objects: int, elapsed_ms: float) -> None:
    """Small translucent banner, top-left, so the confidence threshold used
    for this result is always visible on the image itself - not just in the
    surrounding UI."""
    text = f"conf >= {conf:.2f}   |   {n_objects} object(s)   |   {elapsed_ms:.0f} ms"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(image.shape[:2]) / 900)
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    pad = 8
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 2 * pad, th + baseline + 2 * pad), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, dst=image)
    cv2.putText(image, text, (pad, th + pad), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _extract_detections(result, names: dict[int, str]) -> list[Detection]:
    detections = []
    boxes = result.boxes
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf_val, cls_id in zip(xyxy, confs, cls_ids):
            detections.append(Detection(int(cls_id), names[int(cls_id)], float(conf_val),
                                         (int(x1), int(y1), int(x2), int(y2))))
    detections.sort(key=lambda d: -d.confidence)
    return detections


def detect_image(model: YOLO, image_bgr: np.ndarray, conf: float = DEFAULT_CONF,
                  iou: float = DEFAULT_IOU, draw_badge: bool = True) -> ImageResult:
    """Run YOLO on a single BGR image and return the annotated copy + detections."""
    start = time.perf_counter()
    result = model.predict(image_bgr, conf=conf, iou=iou, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = _extract_detections(result, result.names)
    annotated = image_bgr.copy()
    for det in detections:
        _draw_box(annotated, det)
    if draw_badge:
        _draw_threshold_badge(annotated, conf, len(detections), elapsed_ms)

    return ImageResult(annotated, detections, elapsed_ms, conf)


def detect_video(model: YOLO, in_path: str | Path, out_path: str | Path,
                  conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                  max_side: int = MAX_SIDE, max_frames: int | None = None,
                  progress_cb: Callable[[int, int], None] | None = None) -> VideoResult:
    """Run YOLO frame-by-frame on a video, write an annotated H.264 mp4, and
    return per-frame detections + aggregate stats.

    Written with imageio (bundled ffmpeg via the imageio-ffmpeg wheel) rather
    than cv2.VideoWriter, because OpenCV's own build has no licensed H.264
    encoder - its mp4 output plays in almost nothing but itself. imageio's
    libx264 output is a normal browser-playable mp4.

    max_frames stops early after that many frames (used by the Streamlit app
    so an uploaded video can't stall a free, CPU-only, shared-resource host;
    the coding_practice script leaves it unset to process full videos).
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                 quality=6, macro_block_size=None)

    detections_per_frame: list[list[Detection]] = []
    start = time.perf_counter()
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_max_side(frame, max_side)
            result = detect_image(model, frame, conf=conf, iou=iou, draw_badge=True)
            detections_per_frame.append(result.detections)
            writer.append_data(bgr_to_rgb(result.annotated))
            frame_idx += 1
            if progress_cb:
                progress_cb(frame_idx, n_total or frame_idx)
            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        writer.close()

    elapsed_s = time.perf_counter() - start
    return VideoResult(out_path, detections_per_frame, frame_idx, fps, elapsed_s)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run custom road-damage YOLO detection on one image.")
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    m = load_model(args.model)
    img = load_image(args.image)
    res = detect_image(m, img, conf=args.conf)
    print(f"{len(res.detections)} object(s) in {res.elapsed_ms:.1f} ms:")
    for d in res.detections:
        print(f"  {d.class_name:<20} conf={d.confidence:.2f}  box={d.box}")

    out_path = args.out or (Path(args.image).stem + "_detected.jpg")
    imwrite(out_path, res.annotated)
    print(f"Saved {out_path}")
