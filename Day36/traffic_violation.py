"""
Day 36 - Traffic Violation Detection: rules core module.

Builds on tracker.py's YOLO+ByteTrack tracking and per-track MotionHistory
and adds two independent violation rules on top:

1. WRONG-WAY: compare each vehicle's net motion vector (from
   MotionHistory.direction_vector) against a configured "normal" RoadDirection
   vector. A sustained (debounced) large angle between the two means the
   vehicle is heading the wrong way. "Sustained" matters here the same way
   Day32's occupancy smoothing did: a single noisy frame (a car braking, a
   detection jitter) shouldn't flip a vehicle's status - a majority of a
   short rolling window has to agree first. Once confirmed, a vehicle stays
   flagged for the rest of the clip and is recorded exactly once (by track
   ID) - see ViolationState.update().

2. RESTRICTED ZONE: a rectangular region (fraction-of-frame coordinates,
   same resolution-independent convention as Day31's ROI/Day32's spaces).
   A violation is recorded on the OUTSIDE -> INSIDE transition of a
   vehicle's centroid, the same edge-triggered idea Day31 used for line
   crossings - so a vehicle idling inside the zone doesn't spam duplicate
   events, but leaving and re-entering can violate again.

An optional `direction_roi` further restricts which vehicles the wrong-way
rule even looks at. This matters on a real divided road: a highway's two
carriageways carry legitimately opposite traffic, and without some way to
say "only check vehicles in MY lane", the entire oncoming carriageway would
read as a mass wrong-way violation. Scoping the rule to one carriageway
(via the same rectangle type as the restricted zone) fixes that - see
README "Real challenges faced" for which shipped sample videos need it.

analytics.py imports RestrictedZone/ViolationEvent/etc. from here (never the
reverse) to build Variant 2's dashboard on top of the same state - same
dependency direction Day32 used between parking_detection.py and
analytics.py.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    import imageio

from tracker import (DEFAULT_CONF, DEFAULT_IOU, DEFAULT_MODEL, DEFAULT_TRACKER, MAX_SIDE,
                      VEHICLE_CLASS_IDS, MotionHistory, TrackedVehicle, _extract_vehicles,
                      bgr_to_rgb, draw_trail, load_model, resize_max_side, text_color_for,
                      track_color)

DEFAULT_ANGLE_THRESHOLD_DEG = 130.0  # net-motion vs normal-direction angle beyond this counts as "wrong way"
                                     # this frame - well past 90 deg so a vehicle merely changing lanes or
                                     # turning across the road isn't misread as reversing direction outright.
                                     # Measured on bangkok_boulevard/hillside_street that a divided or
                                     # perspective-converging road produces motion vectors within ~30-40 deg
                                     # of "opposite" purely from lane-change/turning noise; 130 (not 90-100)
                                     # keeps that noise from reading as a confirmed violation - see README
                                     # "Real challenges faced".
CONFIRM_WINDOW = 14       # frames of wrong/right history kept per track, for debouncing
CONFIRM_RATIO = 0.8       # >=80% of the window flagged wrong -> confirm the violation (once, ever, per track)

# Compass directions as (dx, dy) unit vectors in IMAGE coordinates (y grows
# downward) - the same "fraction of frame, resolution independent" spirit as
# Day31's CountingLine, just 8-way instead of 2, since a road's normal
# direction in an oblique aerial/elevated shot is rarely purely horizontal
# or vertical.
_S = math.sqrt(0.5)
COMPASS: dict[str, tuple[float, float]] = {
    "up": (0.0, -1.0), "down": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0),
    "up-right": (_S, -_S), "up-left": (-_S, -_S), "down-right": (_S, _S), "down-left": (-_S, _S),
}
COMPASS_ARROW_GLYPH = {  # unicode arrow per direction, for on-frame labels
    "up": "↑", "down": "↓", "left": "←", "right": "→",
    "up-right": "↗", "up-left": "↖", "down-right": "↘", "down-left": "↙",
}


# ---------------------------------------------------------------------------
# Road direction & restricted-zone regions
# ---------------------------------------------------------------------------

@dataclass
class RoadDirection:
    """The road's configured "normal" traffic direction, plus the debounce
    tuning for how confidently a vehicle must disagree with it before it's
    flagged wrong-way."""
    name: str = "down"
    angle_threshold_deg: float = DEFAULT_ANGLE_THRESHOLD_DEG
    confirm_window: int = CONFIRM_WINDOW
    confirm_ratio: float = CONFIRM_RATIO

    @property
    def vector(self) -> tuple[float, float]:
        return COMPASS[self.name]

    @property
    def arrow(self) -> str:
        return COMPASS_ARROW_GLYPH[self.name]

    def angle_to(self, vector: tuple[float, float]) -> float:
        dx, dy = self.vector
        vx, vy = vector
        dot = max(-1.0, min(1.0, dx * vx + dy * vy))
        return math.degrees(math.acos(dot))


@dataclass
class RegionBox:
    """Axis-aligned rectangle in FRACTION-of-frame coordinates (0-1) - same
    resolution-independent convention as Day31's ROI. Used both for the
    restricted zone (a violation region) and, optionally, to scope the
    wrong-way check to one lane/carriageway (see module docstring)."""
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
# Violation events
# ---------------------------------------------------------------------------

WRONG_WAY = "wrong_way"
RESTRICTED_ZONE = "restricted_zone"


@dataclass
class ViolationEvent:
    track_id: int
    class_name: str
    violation_type: str  # WRONG_WAY or RESTRICTED_ZONE
    frame_idx: int
    time_s: float


# ---------------------------------------------------------------------------
# Violation state - advanced one frame at a time
# ---------------------------------------------------------------------------

class ViolationState:
    """Mutable per-video violation state: wrong-way debouncing + sticky
    flagging, restricted-zone edge-triggered entry detection, and the
    running tallies the brief asks for (total vehicles, counts by type,
    counts by class, the event log).
    """

    def __init__(self, direction: RoadDirection, zone: RegionBox | None = None,
                 direction_roi: RegionBox | None = None):
        self.direction = direction
        self.zone = zone
        self.direction_roi = direction_roi

        self._dir_hist: dict[int, deque] = {}
        self.wrong_way_ids: set[int] = set()          # sticky - once flagged, stays flagged
        self._zone_inside_prev: dict[int, bool] = {}
        self.zone_violation_count: dict[int, int] = Counter()  # per-track, how many times it has entered

        self.id_to_class: dict[int, str] = {}          # every vehicle ever seen, by track ID
        self.events: list[ViolationEvent] = []

    # -- derived stats, computed on demand so there's one source of truth --

    @property
    def total_vehicles(self) -> int:
        return len(self.id_to_class)

    @property
    def vehicle_counts_by_class(self) -> dict[str, int]:
        return dict(Counter(self.id_to_class.values()))

    @property
    def total_violations(self) -> int:
        return len(self.events)

    @property
    def wrong_way_count(self) -> int:
        return sum(1 for e in self.events if e.violation_type == WRONG_WAY)

    @property
    def zone_count(self) -> int:
        return sum(1 for e in self.events if e.violation_type == RESTRICTED_ZONE)

    @property
    def violation_counts_by_class(self) -> dict[str, int]:
        return dict(Counter(e.class_name for e in self.events))

    def update(self, vehicles: list[TrackedVehicle], motion: MotionHistory,
               frame_shape: tuple[int, ...], frame_idx: int, fps: float) -> set[int]:
        """Advance state by one frame. Returns the set of track IDs that
        newly violated (either rule) THIS frame - used to flash feedback."""
        just_violated: set[int] = set()

        for v in vehicles:
            self.id_to_class[v.track_id] = v.class_name
            point = v.centroid

            # -- wrong-way (sticky: once confirmed, never re-evaluated) --
            if v.track_id not in self.wrong_way_ids:
                in_roi = self.direction_roi is None or self.direction_roi.contains(point, frame_shape)
                vec = motion.direction_vector(v.track_id) if in_roi else None
                is_wrong_frame = vec is not None and self.direction.angle_to(vec) >= self.direction.angle_threshold_deg

                hist = self._dir_hist.setdefault(v.track_id, deque(maxlen=self.direction.confirm_window))
                hist.append(is_wrong_frame)
                if len(hist) == hist.maxlen and sum(hist) / len(hist) >= self.direction.confirm_ratio:
                    self.wrong_way_ids.add(v.track_id)
                    self.events.append(ViolationEvent(v.track_id, v.class_name, WRONG_WAY,
                                                        frame_idx, frame_idx / fps if fps else 0.0))
                    just_violated.add(v.track_id)

            # -- restricted zone (edge-triggered: only the outside->inside transition counts) --
            if self.zone is not None:
                inside = self.zone.contains(point, frame_shape)
                was_inside = self._zone_inside_prev.get(v.track_id, False)
                if inside and not was_inside:
                    self.zone_violation_count[v.track_id] += 1
                    self.events.append(ViolationEvent(v.track_id, v.class_name, RESTRICTED_ZONE,
                                                        frame_idx, frame_idx / fps if fps else 0.0))
                    just_violated.add(v.track_id)
                self._zone_inside_prev[v.track_id] = inside

        return just_violated


# ---------------------------------------------------------------------------
# Drawing - Variant 1 "Wrong-Way Detection": full-frame movement overlay
# ---------------------------------------------------------------------------

NORMAL_COLOR = (80, 200, 80)      # BGR green - vehicle obeying the normal direction
WRONG_WAY_COLOR = (40, 40, 235)   # BGR red - confirmed wrong-way vehicle
ZONE_COLOR = (0, 140, 255)        # BGR orange - restricted zone
ARROW_LEN = 34


def draw_direction_indicator(frame: np.ndarray, direction: RoadDirection) -> None:
    """Big "NORMAL DIRECTION" arrow, fixed in the top-left corner - the
    frame of reference every vehicle's own arrow is judged against."""
    h, w = frame.shape[:2]
    cx, cy = 70, 90
    dx, dy = direction.vector
    length = 44
    tip = (int(cx + dx * length), int(cy + dy * length))
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), 58, (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    cv2.arrowedLine(frame, (cx, cy), tip, (255, 255, 255), 4, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(frame, "NORMAL", (cx - 42, cy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "DIRECTION", (cx - 52, cy + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_zone(frame: np.ndarray, zone: RegionBox, label: str = "RESTRICTED ZONE") -> None:
    (x1, y1), (x2, y2) = zone.pixel_rect(frame.shape)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), ZONE_COLOR, -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), ZONE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x1 + 6, max(y1 + 18, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                ZONE_COLOR, 2, cv2.LINE_AA)


def draw_vehicle(frame: np.ndarray, v: TrackedVehicle, motion: MotionHistory, is_wrong: bool) -> None:
    x1, y1, x2, y2 = v.box
    color = WRONG_WAY_COLOR if is_wrong else track_color(v.track_id)
    short_side = min(frame.shape[:2])
    thickness = max(2, round(short_side / 300)) + (2 if is_wrong else 0)

    draw_trail(frame, motion.trail(v.track_id), color, thickness)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f"#{v.track_id} {v.class_name}" + (" WRONG WAY" if is_wrong else "")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, short_side / 900)
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
    ty1 = max(y1 - th - baseline - 6, 0)
    ty2 = ty1 + th + baseline + 6
    cv2.rectangle(frame, (x1, ty1), (x1 + tw + 8, ty2), color, -1)
    cv2.putText(frame, label, (x1 + 4, ty2 - baseline - 2), font, font_scale,
                text_color_for(color), 1, cv2.LINE_AA)

    # per-vehicle heading arrow from its own recent motion, when known
    vec = motion.direction_vector(v.track_id)
    if vec is not None:
        cx, cy = v.centroid
        tip = (int(cx + vec[0] * ARROW_LEN), int(cy + vec[1] * ARROW_LEN))
        cv2.arrowedLine(frame, (cx, cy), tip, color, 2, cv2.LINE_AA, tipLength=0.4)


def draw_violation_badge(frame: np.ndarray, state: ViolationState) -> None:
    line1 = f"VEHICLES {state.total_vehicles}   VIOLATIONS {state.total_violations}"
    line2 = f"Wrong-way {state.wrong_way_count}   Zone {state.zone_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(frame.shape[:2]) / 900)
    (tw1, th), baseline = cv2.getTextSize(line1, font, font_scale, 1)
    (tw2, _), _ = cv2.getTextSize(line2, font, font_scale, 1)
    tw = max(tw1, tw2)
    pad = 8
    line_h = th + baseline + 4
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 2 * pad, 2 * line_h + pad), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
    cv2.putText(frame, line1, (pad, line_h - baseline), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (pad, 2 * line_h - baseline), font, font_scale, WRONG_WAY_COLOR, 1, cv2.LINE_AA)


def render_variant1_frame(frame: np.ndarray, vehicles: list[TrackedVehicle], motion: MotionHistory,
                           state: ViolationState, just_violated: set[int]) -> np.ndarray:
    """Variant 1 look: full-frame overlay - zone fill, every vehicle boxed
    with a heading arrow, wrong-way vehicles in red, a fixed normal-direction
    compass arrow, and a live violation counter badge."""
    annotated = frame.copy()
    if state.zone is not None:
        draw_zone(annotated, state.zone)
    for v in vehicles:
        draw_vehicle(annotated, v, motion, is_wrong=v.track_id in state.wrong_way_ids)
        if v.track_id in just_violated:
            x1, y1, _, _ = v.box
            cv2.putText(annotated, "VIOLATION!", (x1, max(y1 - 26, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)
    draw_direction_indicator(annotated, state.direction)
    draw_violation_badge(annotated, state)
    return annotated


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ViolationResult:
    out_path: Path
    n_frames: int
    fps: float
    elapsed_s: float
    total_vehicles: int
    total_violations: int
    wrong_way_count: int
    zone_count: int
    vehicle_counts_by_class: dict[str, int]
    violation_counts_by_class: dict[str, int]
    events: list[ViolationEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config presets (per-sample-video direction/zone, like Day32's *_spaces.json)
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> tuple[RoadDirection, RegionBox | None, RegionBox | None]:
    data = json.loads(Path(path).read_text())
    direction = RoadDirection(**data["direction"])
    zone = RegionBox(**data["zone"]) if data.get("zone") else None
    roi = RegionBox(**data["direction_roi"]) if data.get("direction_roi") else None
    return direction, zone, roi


def save_config(path: str | Path, direction: RoadDirection, zone: RegionBox | None,
                 direction_roi: RegionBox | None) -> None:
    data = {
        "direction": {"name": direction.name, "angle_threshold_deg": direction.angle_threshold_deg,
                      "confirm_window": direction.confirm_window, "confirm_ratio": direction.confirm_ratio},
        "zone": None if zone is None else {"x_min": zone.x_min, "x_max": zone.x_max,
                                            "y_min": zone.y_min, "y_max": zone.y_max},
        "direction_roi": None if direction_roi is None else {"x_min": direction_roi.x_min, "x_max": direction_roi.x_max,
                                                               "y_min": direction_roi.y_min, "y_max": direction_roi.y_max},
    }
    Path(path).write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Inference pipeline - shared by both demo variants
# ---------------------------------------------------------------------------

def _reset_tracker(model) -> None:
    """Clear ByteTrack's internal state (and its track-ID counter) before a
    new video. model.track(..., persist=True) is meant to persist state
    ACROSS FRAMES OF ONE VIDEO; because this project's app and batch script
    reuse one cached model instance for MULTIPLE videos in a row, without
    this reset every video after the first would keep incrementing IDs
    from wherever the previous video left off (e.g. a 12-vehicle clip
    showing "#1340") instead of starting fresh at #1 - purely cosmetic, but
    confusing for a demo. Safe to call even if no tracker has been created
    yet (first video in a session)."""
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    for t in getattr(predictor, "trackers", []):
        if hasattr(t, "reset"):
            t.reset()
        if hasattr(t, "reset_id"):
            t.reset_id()


def process_video(model, in_path: str | Path, out_path: str | Path, direction: RoadDirection,
                   zone: RegionBox | None = None, direction_roi: RegionBox | None = None,
                   variant: str = "wrong_way", tracker: str = DEFAULT_TRACKER, conf: float = DEFAULT_CONF,
                   iou: float = DEFAULT_IOU, max_side: int = MAX_SIDE, max_frames: int | None = None,
                   progress_cb: Callable[[int, int], None] | None = None) -> ViolationResult:
    """Run YOLO tracking frame-by-frame, evaluate both violation rules for
    every tracked vehicle (see ViolationState), draw either the "wrong_way"
    (Variant 1) or "analytics" (Variant 2) look, and write an annotated
    H.264 mp4.

    variant="analytics" does a local import of analytics.py rather than a
    module-level one, so this module - the shared detection/rules pipeline -
    has no dependency on the presentation layer built on top of it;
    analytics.py is the only file that imports the other way around (same
    split Day32 used).
    """
    if variant not in ("wrong_way", "analytics"):
        raise ValueError(f"unknown variant: {variant!r}")

    draw_analytics_frame = history_cls = None
    if variant == "analytics":
        from analytics import ViolationHistory as history_cls_, draw_analytics_frame as draw_
        draw_analytics_frame = draw_
        history_cls = history_cls_

    in_path, out_path = Path(in_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _reset_tracker(model)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    motion = MotionHistory()
    state = ViolationState(direction, zone, direction_roi)
    history = history_cls() if history_cls else None

    writer = None
    start = time.perf_counter()
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_max_side(frame, max_side)
            result = model.track(frame, persist=True, tracker=tracker, conf=conf, iou=iou,
                                  classes=list(VEHICLE_CLASS_IDS), verbose=False)[0]
            vehicles = _extract_vehicles(result, result.names)
            for v in vehicles:
                motion.update(v.track_id, v.centroid)
            just_violated = state.update(vehicles, motion, frame.shape, frame_idx, fps)
            motion.prune({v.track_id for v in vehicles})

            if variant == "wrong_way":
                canvas = render_variant1_frame(frame, vehicles, motion, state, just_violated)
            else:
                history.append(frame_idx / fps, state)
                canvas = draw_analytics_frame(frame, vehicles, state, history, frame_idx, fps)

            if writer is None:
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
    return ViolationResult(out_path, frame_idx, fps, elapsed_s, state.total_vehicles, state.total_violations,
                            state.wrong_way_count, state.zone_count, state.vehicle_counts_by_class,
                            state.violation_counts_by_class, state.events)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect traffic violations (wrong-way + restricted zone) in one video.")
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--config", help="path to a *_config.json (direction/zone/direction_roi) preset")
    parser.add_argument("--direction", default="down", choices=list(COMPASS))
    parser.add_argument("--zone", nargs=4, type=float, default=None, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--variant", choices=["wrong_way", "analytics"], default="wrong_way")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    m = load_model(args.model)
    if args.config:
        road_direction, road_zone, road_roi = load_config(args.config)
    else:
        road_direction = RoadDirection(name=args.direction)
        road_zone = RegionBox(*args.zone) if args.zone else None
        road_roi = None

    out = args.out or (Path(args.video).stem + f"_{args.variant}.mp4")
    res = process_video(m, args.video, out, road_direction, road_zone, road_roi,
                         variant=args.variant, conf=args.conf)
    print(f"{res.n_frames} frames, {res.elapsed_s:.1f}s")
    print(f"Vehicles: {res.total_vehicles}  Violations: {res.total_violations}  "
          f"(wrong-way: {res.wrong_way_count}, zone: {res.zone_count})")
    for cls, n in res.vehicle_counts_by_class.items():
        print(f"  {cls:<12} {n}")
    print(f"Saved {out}")
