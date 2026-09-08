"""
Day 36 - Traffic violation analytics: the layer that sits on top of
traffic_violation.py's per-frame ViolationState.

Two jobs live here, both "derived from" rather than "part of" the rules
engine, same split Day32 used between parking_detection.py and its own
analytics.py:

1. Turning the running ViolationState into a trend - a rolling history and
   a Normal/Caution/High-Violations traffic-status classification - neither
   of which traffic_violation.py needs to know about to do its own job
   (deciding whether one vehicle just violated a rule).
2. Variant 2's "Traffic Violation Dashboard" look: a docked panel appended
   to the right of the frame (totals, a status banner, per-class counts, a
   scrolling violation event log with timestamps, a trend sparkline) plus a
   deliberately minimal treatment of vehicles themselves (small dots, no
   boxes/arrows) - see draw_analytics_frame(). This is meant to look like a
   different *composition*, not just a different colour scheme, from
   Variant 1's full-frame movement overlay in traffic_violation.py.

traffic_violation.process_video() imports draw_analytics_frame and
ViolationHistory from here with a local import when variant="analytics" -
see that function's docstring for why the dependency only goes this
direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from tracker import TrackedVehicle, track_color
from traffic_violation import WRONG_WAY, RegionBox, ViolationState

SPARKLINE_LEN = 60  # samples kept for the in-frame trend line (not the whole video's history)
EVENT_LOG_LEN = 6   # most recent violation events shown in the docked panel

STATUS_LOW_MAX = 2     # total violations at/under this -> "Normal"
STATUS_HIGH_MIN = 6    # total violations at/over this -> "High Violations"; between -> "Caution"

NORMAL_COLOR = (80, 200, 80)     # BGR green
CAUTION_COLOR = (0, 200, 255)    # BGR amber
HIGH_COLOR = (40, 40, 235)       # BGR red
STATUS_COLORS = {"Normal": NORMAL_COLOR, "Caution": CAUTION_COLOR, "High Violations": HIGH_COLOR}


def traffic_status(total_violations: int) -> str:
    if total_violations <= STATUS_LOW_MAX:
        return "Normal"
    if total_violations >= STATUS_HIGH_MIN:
        return "High Violations"
    return "Caution"


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@dataclass
class HistoryPoint:
    time_s: float
    total_violations: int
    wrong_way_count: int
    zone_count: int


@dataclass
class ViolationHistory:
    """Full per-frame record (for the post-run chart in app.py) plus a
    short rolling window (for the in-frame sparkline, which would be
    unreadable if it tried to plot an entire multi-minute clip)."""
    points: list[HistoryPoint] = field(default_factory=list)

    def append(self, time_s: float, state: ViolationState) -> None:
        self.points.append(HistoryPoint(time_s, state.total_violations, state.wrong_way_count, state.zone_count))

    @property
    def recent(self) -> list[HistoryPoint]:
        return self.points[-SPARKLINE_LEN:]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "time_s": [p.time_s for p in self.points],
            "total_violations": [p.total_violations for p in self.points],
        }).set_index("time_s")


# ---------------------------------------------------------------------------
# Drawing - Variant 2 "Traffic Violation Dashboard": docked panel
# ---------------------------------------------------------------------------

PANEL_FRACTION = 0.36
PANEL_BG = (26, 24, 22)
PANEL_TEXT = (235, 235, 235)
PANEL_MUTED = (150, 150, 150)


def _draw_minimal_vehicles(frame: np.ndarray, vehicles: list[TrackedVehicle], state: ViolationState) -> None:
    """Small centroid dot per vehicle - red for a confirmed wrong-way
    vehicle, its own track colour otherwise - deliberately quieter than
    Variant 1's full boxes/arrows/labels, so the dashboard panel (not the
    video) carries the analytics."""
    short_side = min(frame.shape[:2])
    radius = max(4, short_side // 130)
    for v in vehicles:
        color = (40, 40, 235) if v.track_id in state.wrong_way_ids else track_color(v.track_id)
        cv2.circle(frame, v.centroid, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, v.centroid, radius, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_zone_outline(frame: np.ndarray, zone: RegionBox) -> None:
    (x1, y1), (x2, y2) = zone.pixel_rect(frame.shape)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 1, cv2.LINE_AA)


def _draw_sparkline(panel: np.ndarray, history: ViolationHistory, x0: int, y0: int, w: int, h: int) -> None:
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (45, 42, 40), -1)
    recent = history.recent
    if len(recent) >= 2:
        max_v = max((p.total_violations for p in recent), default=1) or 1
        pts = []
        for i, p in enumerate(recent):
            px = x0 + round(i / (len(recent) - 1) * w)
            py = y0 + h - round(p.total_violations / max_v * h)
            pts.append((px, py))
        cv2.polylines(panel, [np.array(pts, dtype=np.int32)], False, (0, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "VIOLATIONS TREND", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_MUTED, 1, cv2.LINE_AA)


def draw_analytics_frame(frame: np.ndarray, vehicles: list[TrackedVehicle], state: ViolationState,
                          history: ViolationHistory, frame_idx: int, fps: float) -> np.ndarray:
    """Variant 2 look: minimal vehicle dots + a thin zone outline on the
    video itself, and a solid dashboard panel docked to the right - totals,
    a traffic-status banner, per-class counts, a scrolling violation event
    log, a trend sparkline, frame/timestamp. Returns a WIDER canvas (frame +
    panel side by side), not an in-place overlay, so the panel never covers
    any part of the video."""
    annotated = frame.copy()
    if state.zone is not None:
        _draw_zone_outline(annotated, state.zone)
    _draw_minimal_vehicles(annotated, vehicles, state)

    h, w = annotated.shape[:2]
    panel_w = round(w * PANEL_FRACTION)
    if (w + panel_w) % 2 != 0:
        panel_w += 1  # keep the canvas width even - libx264/yuv420p rejects odd frame dimensions
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = annotated

    panel = np.full((h, panel_w, 3), PANEL_BG, dtype=np.uint8)
    pad = max(14, panel_w // 20)
    y = pad + 10
    cv2.putText(panel, "TRAFFIC VIOLATION DASHBOARD", (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                PANEL_MUTED, 1, cv2.LINE_AA)

    status = traffic_status(state.total_violations)
    status_color = STATUS_COLORS[status]

    y += 50
    big_scale = max(1.0, panel_w / 230)
    cv2.putText(panel, str(state.total_violations), (pad, y), cv2.FONT_HERSHEY_SIMPLEX, big_scale,
                PANEL_TEXT, 3, cv2.LINE_AA)
    cv2.putText(panel, "TOTAL VIOLATIONS", (pad, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_MUTED, 1, cv2.LINE_AA)

    y += 45
    cv2.rectangle(panel, (pad, y), (panel_w - pad, y + 26), status_color, -1)
    (tw, th), _ = cv2.getTextSize(status.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(panel, status.upper(), (pad + (panel_w - 2 * pad - tw) // 2, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    y += 45
    for text, color in ((f"Total vehicles      {state.total_vehicles}", PANEL_TEXT),
                         (f"Wrong-way            {state.wrong_way_count}", (60, 60, 235)),
                         (f"Restricted zone      {state.zone_count}", (0, 160, 255))):
        cv2.putText(panel, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += 24

    y += 14
    cv2.putText(panel, "VEHICLE TYPES", (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_MUTED, 1, cv2.LINE_AA)
    y += 20
    counts = state.vehicle_counts_by_class
    class_line = "  ".join(f"{cls}={counts.get(cls, 0)}" for cls in ("car", "truck", "bus", "motorcycle"))
    cv2.putText(panel, class_line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANEL_TEXT, 1, cv2.LINE_AA)

    y += 30
    spark_h = 60
    _draw_sparkline(panel, history, pad, y + 18, panel_w - 2 * pad, spark_h)
    y += 18 + spark_h + 26

    cv2.putText(panel, "RECENT VIOLATIONS (by vehicle ID)", (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                PANEL_MUTED, 1, cv2.LINE_AA)
    y += 20
    recent_events = state.events[-EVENT_LOG_LEN:][::-1]
    if not recent_events:
        cv2.putText(panel, "none yet", (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANEL_MUTED, 1, cv2.LINE_AA)
    else:
        for ev in recent_events:
            if y > h - pad - 16:
                break
            kind = "wrong-way" if ev.violation_type == WRONG_WAY else "zone entry"
            line = f"#{ev.track_id} {ev.class_name:<10} {kind:<10} t={ev.time_s:5.1f}s"
            color = (60, 60, 235) if ev.violation_type == WRONG_WAY else (0, 160, 255)
            cv2.putText(panel, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            y += 20

    time_s = frame_idx / fps if fps else 0.0
    stamp = f"Frame {frame_idx}   t={int(time_s // 60):02d}:{int(time_s % 60):02d}"
    cv2.putText(panel, stamp, (pad, h - pad // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_MUTED, 1, cv2.LINE_AA)

    canvas[:, w:] = panel
    return canvas
