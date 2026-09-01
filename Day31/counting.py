"""
Day 31 - Vehicle counting core module.

Builds on Day30's tracking approach (model.track(..., persist=True) - see
Day30/tracking.py for the detection-vs-tracking background) and adds
line-crossing counting on top: a configurable line (horizontal or vertical)
plus an optional rectangular region of interest (ROI).

Counting logic, in one sentence: for every tracked vehicle, remember which
side of the line its centroid was on last frame; the moment that side
flips, count it - once, keyed by track ID, never again for that same ID.
That last part is what prevents duplicate counts: a per-frame detector with
no memory would see the same physical truck's box "cross" the line's pixel
row on several consecutive frames (its box is taller than one pixel row) and
could count it 3-4 times. A tracked ID crosses exactly once, because we
check "have we already counted this ID?" before counting again - see
CountState.update() below.

app.py and the coding_practice/ scripts both import this module.
"""

from __future__ import annotations

import colorsys
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    import imageio

from ultralytics import YOLO

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
MAX_SIDE = 960
TRAIL_LENGTH = 15  # centroids kept per track, for the fading motion trail

# COCO class ids this project cares about (yolov8n.pt is COCO-pretrained,
# no custom training needed - every vehicle type the brief asks for is
# already a COCO class).
VEHICLE_CLASS_IDS: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CAR_TRUCK_CLASS_IDS: dict[int, str] = {2: "car", 7: "truck"}

_MODEL_CACHE: dict[str, YOLO] = {}
_MODEL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(name: str = DEFAULT_MODEL) -> YOLO:
    """Load (and cache) the model. Fuses eagerly under a lock - see Day29's
    detection.py::load_model for why (avoids a race between concurrent
    Streamlit sessions both lazily fusing the same cached model)."""
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
    """Deterministic BGR colour keyed off TRACK ID (not class), same as
    Day30, so one physical vehicle keeps one colour for its whole clip."""
    hue = (track_id * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)


# ---------------------------------------------------------------------------
# Counting line & ROI
# ---------------------------------------------------------------------------

Side = Literal["horizontal", "vertical"]


@dataclass
class CountingLine:
    """A line spanning the frame, used to detect vehicle crossings.

    position is a fraction (0-1): the line's y-position (as a fraction of
    frame height) if horizontal, or x-position (fraction of frame width) if
    vertical - this keeps the line correctly placed regardless of the
    video's actual resolution.
    """
    orientation: Side = "horizontal"
    position: float = 0.5

    def side_of(self, point: tuple[int, int], frame_shape: tuple[int, ...]) -> int:
        h, w = frame_shape[:2]
        x, y = point
        threshold = self.position * (h if self.orientation == "horizontal" else w)
        coord = y if self.orientation == "horizontal" else x
        return 1 if coord >= threshold else -1

    def direction_label(self, prev_side: int, curr_side: int) -> str:
        if self.orientation == "horizontal":
            return "down" if curr_side > prev_side else "up"
        return "right" if curr_side > prev_side else "left"

    def pixel_coords(self, frame_shape: tuple[int, ...]) -> tuple[tuple[int, int], tuple[int, int]]:
        h, w = frame_shape[:2]
        if self.orientation == "horizontal":
            y = int(self.position * h)
            return (0, y), (w, y)
        x = int(self.position * w)
        return (x, 0), (x, h)


@dataclass
class ROI:
    """Rectangle (fractions of frame size) that counting is restricted to.
    A vehicle detected outside it is still drawn, just never counted -
    useful for ignoring a parking lane or sidewalk visible in-frame."""
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    def contains(self, point: tuple[int, int], frame_shape: tuple[int, ...]) -> bool:
        h, w = frame_shape[:2]
        x, y = point
        return (self.x_min * w <= x <= self.x_max * w) and (self.y_min * h <= y <= self.y_max * h)

    def pixel_rect(self, frame_shape: tuple[int, ...]) -> tuple[tuple[int, int], tuple[int, int]]:
        h, w = frame_shape[:2]
        return (int(self.x_min * w), int(self.y_min * h)), (int(self.x_max * w), int(self.y_max * h))


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

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2


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
# Counting state
# ---------------------------------------------------------------------------

@dataclass
class CountState:
    """Mutable per-video counting state, advanced one frame at a time."""
    line: CountingLine
    roi: ROI | None = None
    prev_side: dict[int, int] = field(default_factory=dict)
    counted_ids: set[int] = field(default_factory=set)
    counts_by_class: Counter = field(default_factory=Counter)
    counts_by_direction: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return len(self.counted_ids)

    def update(self, tracked: list[TrackedBox], frame_shape: tuple[int, ...]) -> list[TrackedBox]:
        """Advance counting state by one frame. Returns the tracks that were
        newly counted THIS frame (used to flash a "+1" on them)."""
        just_counted = []
        for t in tracked:
            point = t.centroid
            if self.roi is not None and not self.roi.contains(point, frame_shape):
                self.prev_side.pop(t.track_id, None)  # left the ROI - re-entry starts fresh
                continue

            side = self.line.side_of(point, frame_shape)
            prev = self.prev_side.get(t.track_id)
            if prev is not None and prev != side and t.track_id not in self.counted_ids:
                self.counted_ids.add(t.track_id)
                direction = self.line.direction_label(prev, side)
                self.counts_by_class[t.class_name] += 1
                self.counts_by_direction[direction] += 1
                just_counted.append(t)
            self.prev_side[t.track_id] = side
        return just_counted


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_track(image: np.ndarray, t: TrackedBox, trail: deque, flash: bool) -> None:
    x1, y1, x2, y2 = t.box
    color = track_color(t.track_id)
    short_side = min(image.shape[:2])
    thickness = max(2, round(short_side / 350))

    pts = list(trail)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        pt_color = tuple(int(c * alpha) for c in color)
        cv2.line(image, pts[i - 1], pts[i], pt_color, max(1, thickness // 2), cv2.LINE_AA)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness + (2 if flash else 0))

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

    if flash:
        cv2.putText(image, "+1", (x2 + 4, y1 + 20), font, max(0.6, font_scale * 1.3),
                    (0, 255, 0), 2, cv2.LINE_AA)


def _draw_line(image: np.ndarray, line: CountingLine) -> None:
    h, w = image.shape[:2]
    color = (0, 215, 255)
    thickness = max(2, min(h, w) // 250)
    p1, p2 = line.pixel_coords(image.shape)
    cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(h, w) / 900)
    if line.orientation == "horizontal":
        neg_label, pos_label = "UP", "DOWN"
        y = p1[1]
        neg_pos, pos_pos = (10, max(y - 12, 18)), (10, min(y + 28, h - 8))
    else:
        neg_label, pos_label = "LEFT", "RIGHT"
        x = p1[0]
        neg_pos, pos_pos = (max(x - 80, 4), 25), (min(x + 10, w - 60), 25)
    cv2.putText(image, neg_label, neg_pos, font, font_scale, color, 2, cv2.LINE_AA)
    cv2.putText(image, pos_label, pos_pos, font, font_scale, color, 2, cv2.LINE_AA)


def _draw_roi(image: np.ndarray, roi: ROI) -> None:
    (x1, y1), (x2, y2) = roi.pixel_rect(image.shape)
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, dst=image)


def _draw_badge(image: np.ndarray, state: CountState) -> None:
    """Translucent banner, top-left - live total/class/direction counts
    burned into the video itself, not just shown in the surrounding UI."""
    class_bits = "  ".join(f"{cls}={n}" for cls, n in sorted(state.counts_by_class.items()))
    dir_bits = "  ".join(f"{d}={n}" for d, n in sorted(state.counts_by_direction.items()))
    line1 = f"TOTAL {state.total}" + (f"   |   {class_bits}" if class_bits else "")
    line2 = dir_bits or "no crossings yet"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(image.shape[:2]) / 900)
    (tw1, th), baseline = cv2.getTextSize(line1, font, font_scale, 1)
    (tw2, _), _ = cv2.getTextSize(line2, font, font_scale, 1)
    tw = max(tw1, tw2)
    pad = 8
    line_h = th + baseline + 4
    box_h = 2 * line_h + pad
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 2 * pad, box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, dst=image)
    cv2.putText(image, line1, (pad, line_h - baseline), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, line2, (pad, 2 * line_h - baseline), font, font_scale, (0, 215, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class VideoCountResult:
    out_path: Path
    n_frames: int
    fps: float
    elapsed_s: float
    tracker: str
    total_count: int
    counts_by_class: dict[str, int]
    counts_by_direction: dict[str, int]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def count_video(model: YOLO, in_path: str | Path, out_path: str | Path,
                 line: CountingLine, roi: ROI | None = None,
                 classes: dict[int, str] | None = None,
                 tracker: str = DEFAULT_TRACKER, conf: float = DEFAULT_CONF,
                 iou: float = DEFAULT_IOU, max_side: int = MAX_SIDE,
                 max_frames: int | None = None,
                 progress_cb: Callable[[int, int], None] | None = None) -> VideoCountResult:
    """Run YOLO tracking restricted to `classes` (default: all 4 vehicle
    types), count crossings of `line` (optionally restricted to `roi`), and
    write an annotated H.264 mp4 (box + trail + line + live count badge).

    persist=True keeps the tracker's internal state alive across frames for
    this video, so IDs stay stable - see Day30/tracking.py::track_video for
    the full explanation of why that flag is load-bearing.
    """
    if classes is None:
        classes = VEHICLE_CLASS_IDS
    class_ids = list(classes)

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
    state = CountState(line=line, roi=roi)

    start = time.perf_counter()
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_max_side(frame, max_side)
            result = model.track(frame, persist=True, tracker=tracker, conf=conf,
                                  iou=iou, classes=class_ids, verbose=False)[0]
            tracked = _extract_tracks(result, result.names)
            just_counted = state.update(tracked, frame.shape)
            just_counted_ids = {t.track_id for t in just_counted}

            annotated = frame.copy()
            if roi is not None:
                _draw_roi(annotated, roi)
            _draw_line(annotated, line)
            for t in tracked:
                trails[t.track_id].append(t.centroid)
                _draw_track(annotated, t, trails[t.track_id], flash=t.track_id in just_counted_ids)
            _draw_badge(annotated, state)

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
    return VideoCountResult(out_path, frame_idx, fps, elapsed_s, tracker,
                             state.total, dict(state.counts_by_class), dict(state.counts_by_direction))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count vehicles crossing a line in one video.")
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, choices=["bytetrack.yaml", "botsort.yaml"])
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--orientation", default="horizontal", choices=["horizontal", "vertical"])
    parser.add_argument("--position", type=float, default=0.5, help="line position, 0-1 fraction of frame")
    parser.add_argument("--car-truck-only", action="store_true",
                         help="restrict detection/classification to car+truck (coding-practice scope)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    m = load_model(args.model)
    line = CountingLine(orientation=args.orientation, position=args.position)
    classes = CAR_TRUCK_CLASS_IDS if args.car_truck_only else VEHICLE_CLASS_IDS
    out = args.out or (Path(args.video).stem + "_counted.mp4")
    res = count_video(m, args.video, out, line=line, classes=classes,
                       tracker=args.tracker, conf=args.conf)
    print(f"{res.n_frames} frames, {res.total_count} vehicle(s) counted, "
          f"{res.elapsed_s:.1f}s ({res.tracker})")
    for cls, n in res.counts_by_class.items():
        print(f"  {cls:<12} {n}")
    for d, n in res.counts_by_direction.items():
        print(f"  {d:<12} {n}")
    print(f"Saved {out}")
