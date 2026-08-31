"""
Day 30 - Multi-object tracking core module.

Wraps Ultralytics YOLO's built-in tracking (model.track(..., persist=True))
for video inference: draws a stable ID-colored box + trail per tracked
object, and reports unique-object counts (by track ID, not per-frame count -
a person visible for 200 frames is one object, not 200).

app.py and the coding_practice/ scripts both import this module - it holds
all the actual logic, nothing lives only in app.py (same split as Day29's
detection.py).
"""

from __future__ import annotations

import colorsys
import threading
import time
from collections import Counter, defaultdict, deque
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

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
TRACKERS = ["bytetrack.yaml", "botsort.yaml"]
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
MAX_SIDE = 960
TRAIL_LENGTH = 15  # centroids kept per track, for the fading motion trail

_MODEL_CACHE: dict[str, YOLO] = {}
_MODEL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(name: str = DEFAULT_MODEL) -> YOLO:
    """Load (and cache) the tracking model. Fuses eagerly under a lock -
    see Day29's detection.py::load_model for why (avoids a race between
    concurrent Streamlit sessions both lazily fusing the same cached model)."""
    if name not in _MODEL_CACHE:
        with _MODEL_LOCK:
            if name not in _MODEL_CACHE:
                model = YOLO(name)
                model.fuse(verbose=False)
                _MODEL_CACHE[name] = model
    return _MODEL_CACHE[name]


# ---------------------------------------------------------------------------
# Image/video helpers
# ---------------------------------------------------------------------------

def resize_max_side(image: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour keyed off TRACK ID (not class) so one
    physical object keeps one colour for its entire appearance in the clip -
    this is what makes ID persistence visible at a glance."""
    hue = (track_id * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


# ---------------------------------------------------------------------------
# Track records
# ---------------------------------------------------------------------------

@dataclass
class TrackedBox:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 pixel coords


def _extract_tracks(result, names: dict[int, str]) -> list[TrackedBox]:
    tracked = []
    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf_val, cls_id, tid in zip(xyxy, confs, cls_ids, ids):
            tracked.append(TrackedBox(int(tid), int(cls_id), names[int(cls_id)],
                                       float(conf_val), (int(x1), int(y1), int(x2), int(y2))))
    tracked.sort(key=lambda t: t.track_id)
    return tracked


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)


def _draw_track(image: np.ndarray, t: TrackedBox, trail: deque) -> None:
    x1, y1, x2, y2 = t.box
    color = track_color(t.track_id)
    short_side = min(image.shape[:2])
    thickness = max(2, round(short_side / 350))

    # fading trail of recent centroids, oldest = thinnest/dimmest
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        pt_color = tuple(int(c * alpha) for c in color)
        cv2.line(image, pts[i - 1], pts[i], pt_color, max(1, thickness // 2), cv2.LINE_AA)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    label = f"#{t.track_id} {t.class_name} {t.confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, short_side / 900)
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
    ty1 = max(y1 - th - baseline - 6, 0)
    ty2 = ty1 + th + baseline + 6
    cv2.rectangle(image, (x1, ty1), (x1 + tw + 8, ty2), color, -1)
    text_color = _text_color_for(color)
    cv2.putText(image, label, (x1 + 4, ty2 - baseline - 2), font, font_scale,
                text_color, 1, cv2.LINE_AA)


def _draw_badge(image: np.ndarray, tracker: str, n_active: int, n_unique_so_far: int) -> None:
    """Small translucent banner, top-left - live unique-object count burned
    into the video itself, not just shown in the surrounding UI."""
    text = f"{tracker}   |   {n_active} active   |   {n_unique_so_far} unique so far"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(image.shape[:2]) / 900)
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    pad = 8
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 2 * pad, th + baseline + 2 * pad), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, dst=image)
    cv2.putText(image, text, (pad, th + pad), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class VideoTrackResult:
    out_path: Path
    tracks_per_frame: list[list[TrackedBox]]
    n_frames: int
    fps: float
    elapsed_s: float
    tracker: str

    @property
    def unique_ids(self) -> set[int]:
        return {t.track_id for frame in self.tracks_per_frame for t in frame}

    @property
    def id_to_class(self) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for frame in self.tracks_per_frame:
            for t in frame:
                mapping[t.track_id] = t.class_name
        return mapping

    @property
    def id_best_conf(self) -> dict[int, float]:
        best: dict[int, float] = {}
        for frame in self.tracks_per_frame:
            for t in frame:
                best[t.track_id] = max(best.get(t.track_id, 0.0), t.confidence)
        return best

    @property
    def id_frame_count(self) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for frame in self.tracks_per_frame:
            counts.update(t.track_id for t in frame)
        return dict(counts)

    @property
    def class_counts(self) -> dict[str, int]:
        """Unique OBJECTS per class (by track ID) - not per-frame detections."""
        return dict(Counter(self.id_to_class.values()))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def track_video(model: YOLO, in_path: str | Path, out_path: str | Path,
                 tracker: str = DEFAULT_TRACKER, conf: float = DEFAULT_CONF,
                 iou: float = DEFAULT_IOU, max_side: int = MAX_SIDE,
                 max_frames: int | None = None,
                 progress_cb: Callable[[int, int], None] | None = None) -> VideoTrackResult:
    """Run YOLO tracking frame-by-frame on a video, write an annotated H.264
    mp4 (ID-coloured boxes + motion trail + live unique-count badge), and
    return per-frame track records + aggregate stats.

    persist=True is load-bearing: it keeps the tracker's internal state
    (and therefore every object's ID) alive across calls to .track() for
    this same video. Without it, each frame would start a brand-new
    tracking session and every object would get a new ID immediately.
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

    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    tracks_per_frame: list[list[TrackedBox]] = []
    seen_ids: set[int] = set()

    start = time.perf_counter()
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_max_side(frame, max_side)
            result = model.track(frame, persist=True, tracker=tracker,
                                  conf=conf, iou=iou, verbose=False)[0]
            tracked = _extract_tracks(result, result.names)
            tracks_per_frame.append(tracked)

            annotated = frame.copy()
            for t in tracked:
                x1, y1, x2, y2 = t.box
                trails[t.track_id].append(((x1 + x2) // 2, (y1 + y2) // 2))
                seen_ids.add(t.track_id)
                _draw_track(annotated, t, trails[t.track_id])
            _draw_badge(annotated, tracker, len(tracked), len(seen_ids))

            writer.append_data(bgr_to_rgb(annotated))
            frame_idx += 1
            if progress_cb:
                progress_cb(frame_idx, n_total or frame_idx)
            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        writer.close()

    elapsed_s = time.perf_counter() - start
    return VideoTrackResult(out_path, tracks_per_frame, frame_idx, fps, elapsed_s, tracker)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO tracking on one video.")
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, choices=TRACKERS)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    m = load_model(args.model)
    out = args.out or (Path(args.video).stem + f"_{args.tracker.split('.')[0]}_tracked.mp4")
    res = track_video(m, args.video, out, tracker=args.tracker, conf=args.conf)
    print(f"{res.n_frames} frames, {len(res.unique_ids)} unique object(s), "
          f"{res.elapsed_s:.1f}s ({res.tracker})")
    for cls, n in res.class_counts.items():
        print(f"  {cls:<15} {n}")
    print(f"Saved {out}")
