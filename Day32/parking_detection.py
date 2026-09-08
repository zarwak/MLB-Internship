"""
Day 32 - Smart Parking Occupancy Detection: core module.

Builds on the YOLOv8n + ByteTrack stack from Day30 (tracking.py) and Day31
(counting.py) and adds a new idea on top: instead of counting objects that
cross a line, we test whether each tracked vehicle overlaps one of several
fixed *parking space* regions.

Parking spaces are polygons (4 points each), stored as fractions of frame
width/height - same resolution-independent convention as Day31's `ROI`/
`CountingLine` - so a layout calibrated once still lines up correctly no
matter what resolution the video is decoded/resized at.

Spaces are usually generated, not hand-drawn: `generate_grid_layout()`
takes a trapezoid (a top edge and a bottom edge, each an x-range, plus the
two y-positions) and subdivides it into a rows x cols grid of quadrilateral
cells with a bilinear interpolation - the same technique used to UV-map a
flat grid onto a perspective-distorted quad. A trapezoid rather than 4 free
points because that's the natural shape of an elevated/oblique parking-row
camera view (the near row is wide, the far row is narrower), and it
collapses the UI to two range sliders + two y-sliders + rows/cols instead
of dragging 8 independent points.

Occupancy, in one sentence: for every tracked vehicle, measure how much of
each parking space's area it covers (via `cv2.intersectConvexConvex` -
exact polygon-clipping, not a bounding-box approximation) and call a space
occupied once that coverage crosses a threshold - which is *why* a
partially visible vehicle (cut off by frame edge, half-hidden behind a
pillar) still marks its space occupied: we never require the whole vehicle
to be visible, just enough of it over the space to be confident.

app.py and the coding_practice/ scripts both import this module.
"""

from __future__ import annotations

import colorsys
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    import imageio

from ultralytics import YOLO

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_CONF = 0.15  # lower than Day30/31's 0.25 - these camera angles put more distance between the
                      # lens and each vehicle than close-up traffic footage does, so a stricter default
                      # would miss real, clearly-parked cars (see README "Challenges")
DEFAULT_IOU = 0.45
MAX_SIDE = 1280
DEFAULT_IMGSZ = 1280  # YOLO's internal inference resolution - see README "Challenges": model.track()
                      # silently resizes its input to imgsz (640 by default) before inference no
                      # matter how large the frame you hand it is, which was quietly shrinking every
                      # small/distant/aerial vehicle in this project to a few pixels before the model
                      # ever saw them. Matches MAX_SIDE so frames aren't resized twice to two
                      # different targets.

DEFAULT_OVERLAP_THRESHOLD = 0.15  # fraction of a space's area a vehicle must cover to count as "parked in it"
SMOOTH_WINDOW = 5      # frames of history kept per space, for debouncing
SMOOTH_ON_RATIO = 0.6  # >=60% of the window hit -> flip to occupied
SMOOTH_OFF_RATIO = 0.4  # <=40% of the window hit -> flip to free (else: keep previous state)
GRID_INSET = 0.08     # shrink each generated cell 8% toward its centroid, purely cosmetic (visual gap between spaces)

# Class names (lowercased) this project treats as a vehicle. Matched against
# whatever model gets loaded rather than hardcoded COCO ids, so swapping in a
# differently-labeled model (e.g. a custom-trained aerial detector with its
# own class list) doesn't silently filter out every detection.
VEHICLE_CLASS_NAMES = {"car", "motorcycle", "motorbike", "bus", "truck", "microbus", "pickup-van", "van"}


def vehicle_class_ids(model_names: dict[int, str]) -> list[int]:
    """Ids whose name matches VEHICLE_CLASS_NAMES, or every id if none do -
    a model with zero name matches (e.g. a custom-trained detector whose
    single class got exported unnamed, as "0" rather than "car") is a
    purpose-built vehicle detector, not one that finds no vehicles."""
    ids = [i for i, n in model_names.items() if n.lower() in VEHICLE_CLASS_NAMES]
    return ids if ids else list(model_names)

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
# Image helpers
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
    """Deterministic BGR colour keyed off track ID, same scheme as Day30/31,
    so one physical vehicle keeps one colour for its whole appearance."""
    hue = (track_id * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _order_ccw(points: np.ndarray) -> np.ndarray:
    """Sort 4 points into a consistent rotational order around their
    centroid. cv2.intersectConvexConvex needs both polygons it's handed to
    be simple (non-self-intersecting) convex contours - sorting by angle
    guarantees that regardless of the order points were generated in."""
    c = points.mean(axis=0)
    ang = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
    return points[np.argsort(ang)]


# ---------------------------------------------------------------------------
# Parking spaces
# ---------------------------------------------------------------------------

@dataclass
class ParkingSpace:
    """One parking space, as a 4-point polygon in FRACTION-of-frame
    coordinates (0-1) - resolution independent, same convention Day31 used
    for its counting line/ROI. Works equally well for a hand-built
    rectangle or one cell out of `generate_grid_layout()`."""
    id: int
    polygon: list[tuple[float, float]]

    def pixel_polygon(self, frame_shape: tuple[int, ...]) -> np.ndarray:
        h, w = frame_shape[:2]
        return np.array([[x * w, y * h] for x, y in self.polygon], dtype=np.float32)

    def centroid_fraction(self) -> tuple[float, float]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return sum(xs) / len(xs), sum(ys) / len(ys)


def _bilinear(tl, tr, br, bl, u: float, v: float) -> tuple[float, float]:
    top = (tl[0] + (tr[0] - tl[0]) * u, tl[1] + (tr[1] - tl[1]) * u)
    bot = (bl[0] + (br[0] - bl[0]) * u, bl[1] + (br[1] - bl[1]) * u)
    return top[0] + (bot[0] - top[0]) * v, top[1] + (bot[1] - top[1]) * v


def generate_grid_layout(top_range: tuple[float, float], bottom_range: tuple[float, float],
                          top_y: float, bottom_y: float, rows: int, cols: int,
                          inset: float = GRID_INSET) -> list[ParkingSpace]:
    """Subdivide the trapezoid (top_range/top_y, bottom_range/bottom_y - all
    fractions of frame size) into a rows x cols grid of quad-shaped parking
    spaces via bilinear interpolation, numbered row-major from 1.

    `inset` shrinks each cell toward its own centroid by that fraction so
    neighbouring spaces have a visible gap when drawn - cosmetic only, it
    doesn't change which vehicle a space "belongs" to.
    """
    tl = (top_range[0], top_y)
    tr = (top_range[1], top_y)
    bl = (bottom_range[0], bottom_y)
    br = (bottom_range[1], bottom_y)

    spaces = []
    space_id = 1
    for r in range(rows):
        for c in range(cols):
            u0, u1 = c / cols, (c + 1) / cols
            v0, v1 = r / rows, (r + 1) / rows
            corners = [
                _bilinear(tl, tr, br, bl, u0, v0),
                _bilinear(tl, tr, br, bl, u1, v0),
                _bilinear(tl, tr, br, bl, u1, v1),
                _bilinear(tl, tr, br, bl, u0, v1),
            ]
            cx = sum(p[0] for p in corners) / 4
            cy = sum(p[1] for p in corners) / 4
            shrunk = [(cx + (x - cx) * (1 - inset), cy + (y - cy) * (1 - inset)) for x, y in corners]
            spaces.append(ParkingSpace(space_id, shrunk))
            space_id += 1
    return spaces


def save_layout(path: str | Path, layout: list[ParkingSpace]) -> None:
    data = [{"id": s.id, "polygon": s.polygon} for s in layout]
    Path(path).write_text(json.dumps(data, indent=2))


def load_layout(path: str | Path) -> list[ParkingSpace]:
    data = json.loads(Path(path).read_text())
    return [ParkingSpace(d["id"], [tuple(p) for p in d["polygon"]]) for d in data]


# ---------------------------------------------------------------------------
# Vehicle tracks
# ---------------------------------------------------------------------------

@dataclass
class TrackedVehicle:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 pixel coords

    def pixel_polygon(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

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
# Occupancy state
# ---------------------------------------------------------------------------

@dataclass
class SpaceStatus:
    space: ParkingSpace
    occupied: bool
    overlap: float               # best current-frame overlap ratio, 0-1
    vehicle_track_id: int | None  # which track produced that overlap, if any


class ParkingLotState:
    """Mutable per-video occupancy state, advanced one frame at a time.

    Three things make this robust instead of just "is any box on top of this
    polygon": (1) occupancy is decided by AREA OVERLAP RATIO, not box
    containment, so a vehicle sticking out of frame or half-hidden behind
    another car still counts as long as enough of it lands on the space;
    (2) each vehicle can claim at most one space and vice versa, via a
    Hungarian-algorithm one-to-one assignment on the full vehicle x space
    overlap matrix - otherwise one real car overlapping two adjacent cells
    gets double-counted as two occupied spaces; (3) each space keeps a short
    rolling window of hit/miss and only flips state once a clear majority of
    recent frames agree, which absorbs a single missed detection instead of
    flickering free/occupied every frame - see `update()`.
    """

    def __init__(self, layout: list[ParkingSpace], overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
                 smooth_window: int = SMOOTH_WINDOW, smooth_on: float = SMOOTH_ON_RATIO,
                 smooth_off: float = SMOOTH_OFF_RATIO):
        self.layout = layout
        self.overlap_threshold = overlap_threshold
        self.smooth_on = smooth_on
        self.smooth_off = smooth_off
        self._history: dict[int, deque] = {s.id: deque(maxlen=smooth_window) for s in layout}
        self._occupied: dict[int, bool] = {s.id: False for s in layout}
        self.seen_track_ids: set[int] = set()

    def update(self, vehicles: list[TrackedVehicle], frame_shape: tuple[int, ...]) -> list[SpaceStatus]:
        vehicle_polys = [(v, _order_ccw(v.pixel_polygon())) for v in vehicles]
        space_polys = [_order_ccw(space.pixel_polygon(frame_shape)) for space in self.layout]
        space_areas = [cv2.contourArea(sp) for sp in space_polys]

        # overlap[i][j] = fraction of space j's area covered by vehicle i.
        overlap_matrix = np.zeros((len(vehicle_polys), len(self.layout)))
        for i, (_, vpoly) in enumerate(vehicle_polys):
            for j, space_poly in enumerate(space_polys):
                if space_areas[j] <= 0:
                    continue
                inter_area, _ = cv2.intersectConvexConvex(space_poly, vpoly)
                overlap_matrix[i, j] = inter_area / space_areas[j]

        # One vehicle can claim at most one space, and one space at most one
        # vehicle, per frame - the Hungarian algorithm on the overlap matrix
        # finds the assignment that maximizes total overlap. Without this, a
        # single car straddling two adjacent cells (or one big enough to
        # graze a neighboring space) would independently win "best overlap"
        # against both and get double-counted - measured to happen in every
        # frame of aerial_diagonal_lot before this fix (see README "Real
        # challenges faced"). A forced pairing with zero overlap (more
        # vehicles/spaces on one side than the other) is filtered out below.
        assigned: dict[int, tuple[int, float]] = {}  # space idx -> (vehicle idx, overlap)
        if overlap_matrix.size > 0:
            vehicle_idx, space_idx = linear_sum_assignment(-overlap_matrix)
            for i, j in zip(vehicle_idx, space_idx):
                ratio = overlap_matrix[i, j]
                if ratio > 0:
                    assigned[j] = (i, ratio)

        statuses = []
        for j, space in enumerate(self.layout):
            best_overlap, best_vehicle = 0.0, None
            if j in assigned:
                i, ratio = assigned[j]
                best_overlap, best_vehicle = ratio, vehicle_polys[i][0].track_id

            hit = best_overlap >= self.overlap_threshold
            if hit and best_vehicle is not None:
                # unique-vehicle count is by track ID, scoped to vehicles that actually
                # occupied a monitored space - not every vehicle YOLO sees anywhere in
                # frame, which over-counts once the ROI is a small slice of a busy lot
                # (see README "duplicate vehicle detection")
                self.seen_track_ids.add(best_vehicle)
            hist = self._history[space.id]
            hist.append(hit)
            frac = sum(hist) / len(hist)
            if frac >= self.smooth_on:
                self._occupied[space.id] = True
            elif frac <= self.smooth_off:
                self._occupied[space.id] = False
            # else: not enough of a majority yet - keep the previous state

            statuses.append(SpaceStatus(space, self._occupied[space.id], best_overlap, best_vehicle))
        return statuses


# ---------------------------------------------------------------------------
# Drawing - Variant 1 "Parking Monitor": full-frame overlay
# ---------------------------------------------------------------------------

FREE_COLOR = (80, 200, 80)     # BGR green
OCCUPIED_COLOR = (60, 60, 230)  # BGR red


def _text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)


def draw_monitor_spaces(frame: np.ndarray, statuses: list[SpaceStatus]) -> None:
    overlay = frame.copy()
    for status in statuses:
        color = OCCUPIED_COLOR if status.occupied else FREE_COLOR
        pts = status.space.pixel_polygon(frame.shape).astype(np.int32)
        cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, dst=frame)

    for status in statuses:
        color = OCCUPIED_COLOR if status.occupied else FREE_COLOR
        pts = status.space.pixel_polygon(frame.shape).astype(np.int32)
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        cx, cy = pts.mean(axis=0).astype(int)
        label = str(status.space.id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
        cv2.putText(frame, label, (cx - tw // 2, cy + th // 2), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_vehicle_box(frame: np.ndarray, v: TrackedVehicle, label: bool = True) -> None:
    x1, y1, x2, y2 = v.box
    color = track_color(v.track_id)
    short_side = min(frame.shape[:2])
    thickness = max(2, round(short_side / 350))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return
    text = f"#{v.track_id} {v.class_name} {v.confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, short_side / 1000)
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    ty1 = max(y1 - th - baseline - 6, 0)
    ty2 = ty1 + th + baseline + 6
    cv2.rectangle(frame, (x1, ty1), (x1 + tw + 8, ty2), color, -1)
    cv2.putText(frame, text, (x1 + 4, ty2 - baseline - 2), font, font_scale,
                _text_color_for(color), 1, cv2.LINE_AA)


def draw_monitor_badge(frame: np.ndarray, total: int, occupied: int) -> None:
    free = total - occupied
    text = f"SPACES  Total {total}   Occupied {occupied}   Free {free}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(frame.shape[:2]) / 900)
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    pad = 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 2 * pad, th + baseline + 2 * pad), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
    cv2.putText(frame, text, (pad, th + pad), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_monitor_frame(frame: np.ndarray, statuses: list[SpaceStatus], vehicles: list[TrackedVehicle]) -> np.ndarray:
    """Variant 1 look: translucent green/red fill per space, full vehicle
    boxes with track-ID labels on top, a top-left counts badge. Busy,
    operational "watch the lot" view."""
    annotated = frame.copy()
    draw_monitor_spaces(annotated, statuses)
    for v in vehicles:
        draw_vehicle_box(annotated, v)
    occupied = sum(1 for s in statuses if s.occupied)
    draw_monitor_badge(annotated, len(statuses), occupied)
    return annotated


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ParkingFrameSample:
    frame_idx: int
    time_s: float
    occupied: int
    total: int

    @property
    def occupancy_pct(self) -> float:
        return 100.0 * self.occupied / self.total if self.total else 0.0


@dataclass
class ParkingResult:
    out_path: Path
    n_frames: int
    fps: float
    elapsed_s: float
    total_spaces: int
    final_occupied: int
    unique_vehicles_seen: int
    history: list[ParkingFrameSample] = field(default_factory=list)
    final_statuses: list[SpaceStatus] = field(default_factory=list)

    @property
    def final_free(self) -> int:
        return self.total_spaces - self.final_occupied

    @property
    def final_occupancy_pct(self) -> float:
        return 100.0 * self.final_occupied / self.total_spaces if self.total_spaces else 0.0


# ---------------------------------------------------------------------------
# Inference pipeline - shared by both demo variants
# ---------------------------------------------------------------------------

def process_video(model: YOLO, in_path: str | Path, out_path: str | Path, layout: list[ParkingSpace],
                   variant: str = "monitor", tracker: str = DEFAULT_TRACKER, conf: float = DEFAULT_CONF,
                   iou: float = DEFAULT_IOU, overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
                   max_side: int = MAX_SIDE, imgsz: int = DEFAULT_IMGSZ, max_frames: int | None = None,
                   progress_cb: Callable[[int, int], None] | None = None) -> ParkingResult:
    """Run YOLO tracking frame-by-frame, test every tracked vehicle against
    every parking space (see `ParkingLotState`), draw either the "monitor"
    or "analytics" look (see render_monitor_frame / analytics.py's
    draw_analytics_frame), and write an annotated H.264 mp4.

    variant="analytics" does a local import of analytics.py rather than a
    module-level one, so this module - the shared detection/occupancy
    pipeline - has no dependency on the presentation layer built on top of
    it; analytics.py is the only file that imports the other way around.
    """
    if variant not in ("monitor", "analytics"):
        raise ValueError(f"unknown variant: {variant!r}")

    draw_analytics_frame = history_cls = None
    if variant == "analytics":
        from analytics import OccupancyHistory, draw_analytics_frame as _draw
        draw_analytics_frame = _draw
        history_cls = OccupancyHistory

    in_path, out_path = Path(in_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    state = ParkingLotState(layout, overlap_threshold=overlap_threshold)
    history = history_cls() if history_cls else None
    samples: list[ParkingFrameSample] = []
    track_class_ids = vehicle_class_ids(model.names)

    writer = None
    start = time.perf_counter()
    frame_idx = 0
    last_statuses: list[SpaceStatus] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_max_side(frame, max_side)
            result = model.track(frame, persist=True, tracker=tracker, conf=conf, iou=iou, imgsz=imgsz,
                                  max_det=1000, classes=track_class_ids, verbose=False)[0]
            vehicles = _extract_vehicles(result, result.names)
            statuses = state.update(vehicles, frame.shape)
            last_statuses = statuses
            occupied = sum(1 for s in statuses if s.occupied)
            samples.append(ParkingFrameSample(frame_idx, frame_idx / fps, occupied, len(layout)))

            if variant == "monitor":
                canvas = render_monitor_frame(frame, statuses, vehicles)
            else:
                history.append(frame_idx / fps, occupied, len(layout))
                canvas = draw_analytics_frame(frame, statuses, history, frame_idx, fps)

            if writer is None:
                h, w = canvas.shape[:2]
                writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                             quality=6, macro_block_size=None)
            writer.append_data(bgr_to_rgb(canvas))

            frame_idx += 1
            if progress_cb:
                progress_cb(frame_idx, n_total or frame_idx)
            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.close()

    elapsed_s = time.perf_counter() - start
    final_occupied = sum(1 for s in last_statuses if s.occupied)
    return ParkingResult(out_path, frame_idx, fps, elapsed_s, len(layout), final_occupied,
                          len(state.seen_track_ids), samples, last_statuses)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect parking-space occupancy in one video.")
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--layout", help="path to a *_spaces.json layout file")
    parser.add_argument("--top-range", nargs=2, type=float, default=[0.15, 0.85])
    parser.add_argument("--bottom-range", nargs=2, type=float, default=[0.0, 1.0])
    parser.add_argument("--top-y", type=float, default=0.4)
    parser.add_argument("--bottom-y", type=float, default=0.8)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--variant", choices=["monitor", "analytics"], default="monitor")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                         help="YOLO inference resolution - raise this (e.g. 1920) for aerial/top-down "
                              "footage where vehicles are small; see README Challenges")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    m = load_model(args.model)
    if args.layout:
        lot_layout = load_layout(args.layout)
    else:
        lot_layout = generate_grid_layout(tuple(args.top_range), tuple(args.bottom_range),
                                           args.top_y, args.bottom_y, args.rows, args.cols)
    out = args.out or (Path(args.video).stem + f"_{args.variant}.mp4")
    res = process_video(m, args.video, out, lot_layout, variant=args.variant,
                         conf=args.conf, overlap_threshold=args.overlap, imgsz=args.imgsz)
    print(f"{res.n_frames} frames, {res.elapsed_s:.1f}s")
    print(f"Spaces: {res.total_spaces}  Occupied: {res.final_occupied}  Free: {res.final_free}  "
          f"Occupancy: {res.final_occupancy_pct:.1f}%")
    print(f"Unique vehicles seen: {res.unique_vehicles_seen}")
    print(f"Saved {out}")
