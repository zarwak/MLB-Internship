"""
Day 32 - Parking occupancy analytics: the layer that sits on top of
parking_detection.py's per-frame occupancy state.

Two jobs live here, both "derived from" rather than "part of" detection:

1. Turning a stream of (occupied, total) samples into a trend - a rolling
   history, a Low/Medium/High utilization classification, and a small
   in-frame sparkline - none of which parking_detection.py needs to know
   about to do its own job (deciding whether one space is occupied right
   now).
2. Variant 2's "Parking Analytics" look: a docked dashboard panel appended
   to the right of the frame (big occupancy %, a utilization bar, the
   sparkline, a timestamp) plus a deliberately minimal treatment of the
   spaces themselves (thin outline + a centroid dot, no vehicle boxes) -
   see draw_analytics_frame(). This is meant to look like a different
   *composition*, not just a different colour scheme, from Variant 1's
   full-frame monitor overlay in parking_detection.py.

parking_detection.process_video() imports draw_analytics_frame and
OccupancyHistory from here with a local import when variant="analytics" -
see that function's docstring for why the dependency only goes this
direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from parking_detection import FREE_COLOR, OCCUPIED_COLOR, SpaceStatus

SPARKLINE_LEN = 60  # samples kept for the in-frame trend line (not the whole video's history)

UTIL_LOW_MAX = 40    # occupancy% at/under this -> "Low"
UTIL_HIGH_MIN = 75   # occupancy% at/over this -> "High"; between the two -> "Medium"

LEVEL_COLORS = {"Low": FREE_COLOR, "Medium": (0, 200, 255), "High": OCCUPIED_COLOR}  # BGR


def utilization_level(occupancy_pct: float) -> str:
    if occupancy_pct <= UTIL_LOW_MAX:
        return "Low"
    if occupancy_pct >= UTIL_HIGH_MIN:
        return "High"
    return "Medium"


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@dataclass
class HistoryPoint:
    time_s: float
    occupied: int
    total: int

    @property
    def pct(self) -> float:
        return 100.0 * self.occupied / self.total if self.total else 0.0


@dataclass
class OccupancyHistory:
    """Full per-frame record (for the post-run chart in app.py) plus a
    short rolling window (for the in-frame sparkline, which would be
    unreadable if it tried to plot an entire multi-minute clip)."""
    points: list[HistoryPoint] = field(default_factory=list)

    def append(self, time_s: float, occupied: int, total: int) -> None:
        self.points.append(HistoryPoint(time_s, occupied, total))

    @property
    def recent(self) -> list[HistoryPoint]:
        return self.points[-SPARKLINE_LEN:]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "time_s": [p.time_s for p in self.points],
            "occupancy_pct": [p.pct for p in self.points],
        }).set_index("time_s")


# ---------------------------------------------------------------------------
# Drawing - Variant 2 "Parking Analytics": docked dashboard panel
# ---------------------------------------------------------------------------

PANEL_FRACTION = 0.32       # panel width as a fraction of the video frame's width
PANEL_BG = (26, 24, 22)
PANEL_TEXT = (235, 235, 235)
PANEL_MUTED = (150, 150, 150)


def _draw_minimal_spaces(frame: np.ndarray, statuses: list[SpaceStatus]) -> None:
    """Thin outline + a small centroid dot per space - deliberately quieter
    than Variant 1's filled polygons, so the dashboard panel (not the
    video) carries the analytics."""
    short_side = min(frame.shape[:2])
    radius = max(3, short_side // 150)
    for status in statuses:
        color = OCCUPIED_COLOR if status.occupied else FREE_COLOR
        pts = status.space.pixel_polygon(frame.shape).astype(np.int32)
        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)


def _draw_sparkline(panel: np.ndarray, history: OccupancyHistory, x0: int, y0: int, w: int, h: int) -> None:
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (45, 42, 40), -1)
    recent = history.recent
    if len(recent) >= 2:
        pts = []
        for i, p in enumerate(recent):
            px = x0 + round(i / (len(recent) - 1) * w)
            py = y0 + h - round(p.pct / 100 * h)
            pts.append((px, py))
        cv2.polylines(panel, [np.array(pts, dtype=np.int32)], False, (0, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "OCCUPANCY TREND", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_MUTED, 1, cv2.LINE_AA)


def draw_analytics_frame(frame: np.ndarray, statuses: list[SpaceStatus], history: OccupancyHistory,
                          frame_idx: int, fps: float) -> np.ndarray:
    """Variant 2 look: minimal space markers on the video itself, and a
    solid dashboard panel docked to the right - big occupancy %, a
    utilization bar/level, a trend sparkline, frame/timestamp. Returns a
    WIDER canvas (frame + panel side by side), not an in-place overlay, so
    the panel never covers any part of the video."""
    annotated = frame.copy()
    _draw_minimal_spaces(annotated, statuses)

    h, w = annotated.shape[:2]
    panel_w = round(w * PANEL_FRACTION)
    if (w + panel_w) % 2 != 0:
        panel_w += 1  # keep the canvas width even - libx264/yuv420p rejects odd frame dimensions
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = annotated

    panel = np.full((h, panel_w, 3), PANEL_BG, dtype=np.uint8)
    total = len(statuses)
    occupied = sum(1 for s in statuses if s.occupied)
    free = total - occupied
    pct = 100.0 * occupied / total if total else 0.0
    level = utilization_level(pct)
    level_color = LEVEL_COLORS[level]

    pad = max(14, panel_w // 18)
    y = pad + 10
    cv2.putText(panel, "PARKING ANALYTICS", (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PANEL_MUTED, 1, cv2.LINE_AA)

    y += 60
    big_scale = max(1.1, panel_w / 190)
    cv2.putText(panel, f"{pct:.0f}%", (pad, y), cv2.FONT_HERSHEY_SIMPLEX, big_scale, PANEL_TEXT, 3, cv2.LINE_AA)
    cv2.putText(panel, "OCCUPANCY", (pad, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_MUTED, 1, cv2.LINE_AA)

    y += 55
    bar_w = panel_w - 2 * pad
    bar_h = 22
    cv2.rectangle(panel, (pad, y), (pad + bar_w, y + bar_h), (50, 48, 46), -1)
    fill_w = round(bar_w * min(pct, 100) / 100)
    if fill_w > 0:
        cv2.rectangle(panel, (pad, y), (pad + fill_w, y + bar_h), level_color, -1)
    cv2.rectangle(panel, (pad, y), (pad + bar_w, y + bar_h), (80, 78, 76), 1)
    cv2.putText(panel, level.upper(), (pad, y + bar_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, level_color, 2, cv2.LINE_AA)

    y += bar_h + 45
    for text, color in ((f"Occupied   {occupied}", OCCUPIED_COLOR),
                         (f"Free       {free}", FREE_COLOR),
                         (f"Total      {total}", PANEL_TEXT)):
        cv2.putText(panel, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 26

    y += 20
    spark_h = min(90, h - y - 60)
    if spark_h > 20:
        _draw_sparkline(panel, history, pad, y + 20, bar_w, spark_h)
        y += 20 + spark_h

    time_s = frame_idx / fps if fps else 0.0
    stamp = f"Frame {frame_idx}   t={int(time_s // 60):02d}:{int(time_s % 60):02d}"
    cv2.putText(panel, stamp, (pad, h - pad // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_MUTED, 1, cv2.LINE_AA)

    canvas[:, w:] = panel
    return canvas
