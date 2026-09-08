"""
Day 36 - Traffic Violation Detection: tracking core module.

Wraps Ultralytics YOLO's built-in tracking (model.track(..., persist=True)) -
same stack as Day30/31/32 (YOLOv8n + ByteTrack) - and adds the one thing this
project needs on top that those didn't: a short position HISTORY per track
ID, so a vehicle's current *movement direction* can be estimated frame to
frame. Day31 only needed "which side of a line is the centroid on"; Day36
needs "which way is this vehicle actually heading", which requires a few
frames of memory per track, not just the current one.

traffic_violation.py and analytics.py both import this module - it owns
detection + tracking + raw motion, nothing about violation RULES (direction
thresholds, zones) lives here, same separation Day32 drew between
parking_detection.py's tracking and its occupancy rules.
"""

from __future__ import annotations

import colorsys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
MAX_SIDE = 960
TRAIL_LENGTH = 20  # centroids kept per track - drives both the drawn motion trail and direction estimation

# COCO class ids this project treats as a vehicle (yolov8n.pt is
# COCO-pretrained - every vehicle type the brief asks for, including buses,
# is already a COCO class, no custom training needed).
VEHICLE_CLASS_IDS: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

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
    """Deterministic BGR colour keyed off track ID, same scheme as
    Day30/31/32, so one physical vehicle keeps one colour for its whole
    appearance in the clip."""
    hue = (track_id * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)


# ---------------------------------------------------------------------------
# Track records
# ---------------------------------------------------------------------------

@dataclass
class TrackedVehicle:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 pixel coords

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2


def _extract_vehicles(result, names: dict[int, str]) -> list[TrackedVehicle]:
    vehicles = []
    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf_val, cls_id, tid in zip(xyxy, confs, cls_ids, ids):
            vehicles.append(TrackedVehicle(int(tid), int(cls_id), names[int(cls_id)],
                                            float(conf_val), (int(x1), int(y1), int(x2), int(y2))))
    vehicles.sort(key=lambda v: v.track_id)
    return vehicles


# ---------------------------------------------------------------------------
# Motion history - direction estimation per track ID
# ---------------------------------------------------------------------------

MIN_DISPLACEMENT_PX = 6  # below this net movement across the history window, direction is "unknown" (still/idling)
                          # rather than a noisy near-zero vector - a stopped car's box jitters a few pixels
                          # frame to frame from detection noise alone, which would otherwise report a random heading


class MotionHistory:
    """Per-track rolling window of centroids, the only state needed to turn
    "where is this box now" into "which way is this vehicle moving".

    Direction is estimated from the NET displacement across the whole
    window (oldest centroid -> newest), not frame-to-frame deltas - a
    single-frame delta is dominated by detection jitter (a box's exact
    corner can wobble a couple of pixels even for a parked car), while the
    net vector over ~15-20 frames reflects real motion.
    """

    def __init__(self, length: int = TRAIL_LENGTH):
        self.length = length
        self._history: dict[int, deque] = {}

    def update(self, track_id: int, centroid: tuple[int, int]) -> None:
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.length)
        self._history[track_id].append(centroid)

    def trail(self, track_id: int) -> deque:
        return self._history.get(track_id, deque())

    def direction_vector(self, track_id: int) -> tuple[float, float] | None:
        """Unit vector (dx, dy) of net movement, or None if the track is too
        new or too nearly stationary to have a reliable heading."""
        hist = self._history.get(track_id)
        if hist is None or len(hist) < 2:
            return None
        (x0, y0), (x1, y1) = hist[0], hist[-1]
        dx, dy = x1 - x0, y1 - y0
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist < MIN_DISPLACEMENT_PX:
            return None
        return dx / dist, dy / dist

    def prune(self, active_ids: set[int]) -> None:
        """Drop history for tracks no longer present - keeps memory bounded
        on a long video with many transient IDs."""
        for tid in list(self._history):
            if tid not in active_ids:
                del self._history[tid]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_trail(image: np.ndarray, trail: deque, color: tuple[int, int, int], thickness: int) -> None:
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        pt_color = tuple(int(c * alpha) for c in color)
        cv2.line(image, pts[i - 1], pts[i], pt_color, max(1, thickness // 2), cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class VideoTrackResult:
    out_path: Path
    n_frames: int
    fps: float
    elapsed_s: float


# ---------------------------------------------------------------------------
# CLI - plain tracking smoke test (no violation rules; see traffic_violation.py for the real pipeline)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run plain YOLO+ByteTrack vehicle tracking on one video "
                                                  "(no violation rules - see traffic_violation.py for that).")
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    m = load_model(args.model)
    out = args.out or (Path(args.video).stem + "_tracked.mp4")

    try:
        import imageio.v2 as imageio
    except ImportError:  # pragma: no cover
        import imageio

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=6, macro_block_size=None)
    motion = MotionHistory()
    seen_ids: set[int] = set()
    start = time.perf_counter()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_max_side(frame)
        result = m.track(frame, persist=True, tracker=DEFAULT_TRACKER, conf=args.conf,
                          classes=list(VEHICLE_CLASS_IDS), verbose=False)[0]
        vehicles = _extract_vehicles(result, result.names)
        annotated = frame.copy()
        for v in vehicles:
            seen_ids.add(v.track_id)
            motion.update(v.track_id, v.centroid)
            color = track_color(v.track_id)
            draw_trail(annotated, motion.trail(v.track_id), color, 3)
            x1, y1, x2, y2 = v.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"#{v.track_id} {v.class_name}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        writer.append_data(bgr_to_rgb(annotated))
        frame_idx += 1
    cap.release()
    writer.close()
    print(f"{frame_idx} frames, {len(seen_ids)} unique vehicle(s), {time.perf_counter() - start:.1f}s")
    print(f"Saved {out}")
