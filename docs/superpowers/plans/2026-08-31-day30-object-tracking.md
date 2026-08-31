# Day 30 Object Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Day-30 deliverable: a `tracking.py` core module + Streamlit
"Smart Object Tracking System" app that runs ByteTrack/BoT-SORT multi-object
tracking (via Ultralytics YOLO) on 5 real videos, keeps IDs stable through
occlusion/crossing, and reports unique-object counts.

**Architecture:** Same split Day27/Day29 already use in this repo -
`tracking.py` holds all model/inference/drawing logic (no Streamlit imports),
`app.py` is UI-only and imports from it, `coding_practice/*.py` are one-off
scripts that also import `tracking.py` to fulfill the "Coding Practice"
section of the brief.

**Tech Stack:** Python, `ultralytics` (YOLOv8 + built-in ByteTrack/BoT-SORT
trackers via `model.track(..., persist=True)`), OpenCV, `imageio[ffmpeg]`
for browser-playable mp4 output, Streamlit, pandas.

**Note on testing approach:** this repo has no pytest/unit-test
infrastructure anywhere (Day27, Day29 - verified by inspection). Their
established verification pattern is: run the actual script/CLI against real
sample data and inspect the printed/produced output. This plan follows that
same pattern rather than introducing pytest - "run X, expect Y" steps below
are real, checkable commands and expected outputs, not unit tests.

## Global Constraints

- Python module style: `from __future__ import annotations`, dataclasses,
  type hints on public functions - matches Day29's `detection.py`.
- `ultralytics>=8.3.0`, `opencv-python-headless>=4.9.0`, `streamlit>=1.40.0`,
  `imageio[ffmpeg]>=2.34.0` (exact floors copied from Day29's
  `requirements.txt` - already proven working in this environment).
- Video output must be browser-playable H.264 mp4 via `imageio` - do **not**
  use `cv2.VideoWriter` for final output (this OpenCV build has no licensed
  H.264 encoder; confirmed in Day29's `detection.py` docstring).
- `model.track(..., persist=True)` is required on every frame of a given
  video - without `persist=True` the tracker's internal state resets every
  call and every object gets a new ID every frame.
- The 5 sample videos already exist at `Day30/sample_videos/` (done in a
  prior session): `pedestrians_cctv.mp4`, `pedestrians_crosswalk.mp4`,
  `pedestrians_mall.mp4`, `sports_soccer.mp4`, `sports_basketball.mp4`.
- Trackers to support: `"bytetrack.yaml"` (default) and `"botsort.yaml"` -
  both ship inside the `ultralytics` package, no extra download needed.

---

### Task 1: Pretrained weights + confirm `.track()` works in this environment

**Files:**
- Create: `Day30/yolov8n.pt` (copy of the existing `Day27/yolov8n.pt` - same
  COCO nano checkpoint, no need to re-download)

**Interfaces:**
- Produces: a working `Day30/yolov8n.pt` file that Task 2's `load_model()`
  reads.

- [ ] **Step 1:** Copy the weights file.

```bash
cp "Day27/yolov8n.pt" "Day30/yolov8n.pt"
```

- [ ] **Step 2:** Sanity-check `.track()` runs end-to-end on one of the real
  sample videos, one frame, before writing any project code around it.

```bash
python -c "
from ultralytics import YOLO
import cv2
m = YOLO('Day30/yolov8n.pt')
cap = cv2.VideoCapture('Day30/sample_videos/pedestrians_cctv.mp4')
ok, frame = cap.read()
r = m.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)[0]
print('boxes:', 0 if r.boxes is None else len(r.boxes))
print('has ids:', r.boxes.id is not None)
"
```

Expected: prints `boxes: <some number > 0>` and `has ids: True` (the CCTV
frame has several people in it; `boxes.id` is only populated when tracking,
not plain `.predict()` - this confirms the tracking API path, weights file,
and OpenCV video read all work together before Task 2 is written).

- [ ] **Step 3: Commit**

```bash
git add Day30/yolov8n.pt
git commit -m "Add Day-30 pretrained YOLOv8n weights for tracking"
```

---

### Task 2: `tracking.py` - core tracking module

**Files:**
- Create: `Day30/tracking.py`

**Interfaces:**
- Produces (used by Tasks 3, 4, 5):
  - `load_model(name: str = DEFAULT_MODEL) -> YOLO`
  - `TrackedBox` dataclass: `track_id: int, class_id: int, class_name: str, confidence: float, box: tuple[int,int,int,int]`
  - `VideoTrackResult` dataclass with fields `out_path: Path, tracks_per_frame: list[list[TrackedBox]], n_frames: int, fps: float, elapsed_s: float, tracker: str` and properties `unique_ids: set[int]`, `class_counts: dict[str,int]` (per **unique track ID**, not per frame), `id_to_class: dict[int,str]`, `id_best_conf: dict[int,float]`, `id_frame_count: dict[int,int]`
  - `track_video(model, in_path, out_path, tracker="bytetrack.yaml", conf=0.25, iou=0.45, max_side=960, max_frames=None, progress_cb=None) -> VideoTrackResult`
  - `resize_max_side`, `bgr_to_rgb` (copy verbatim from Day29's `detection.py` - identical, reusable helpers)

This module has **no Streamlit import** - `app.py` and the `coding_practice`
scripts both depend on it being pure/importable standalone.

- [ ] **Step 1: Write the module.**

```python
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
from dataclasses import dataclass, field
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
    text = f"{tracker}   |   {n_active} active   |   {n_unique_so_far} unique so far"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(image.shape[:2]) / 900)
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    pad = 8
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 2 * pad, th + baseline + 2 * pad), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, dst=image)
    cv2.putText(image, text, (pad, th + pad), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


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
        id_class = self.id_to_class
        return dict(Counter(id_class.values()))


def track_video(model: YOLO, in_path: str | Path, out_path: str | Path,
                 tracker: str = DEFAULT_TRACKER, conf: float = DEFAULT_CONF,
                 iou: float = DEFAULT_IOU, max_side: int = MAX_SIDE,
                 max_frames: int | None = None,
                 progress_cb: Callable[[int, int], None] | None = None) -> VideoTrackResult:
    """Run YOLO tracking frame-by-frame on a video, write an annotated H.264
    mp4 (ID-coloured boxes + motion trail + live unique-count badge), and
    return per-frame track records + aggregate stats.

    persist=True is load-bearing: it keeps the tracker's internal state
    (and therefore every object's ID) alive across calls to this frame's
    .track(). Without it, each frame would start a brand-new tracking
    session and every object would get a new ID immediately.
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
```

- [ ] **Step 2: Run it against a real sample video and confirm the output
  makes sense.**

```bash
cd Day30
python tracking.py sample_videos/pedestrians_cctv.mp4 --tracker bytetrack.yaml --out /tmp_check.mp4
```

Expected: prints a frame count matching `pedestrians_cctv.mp4`'s ~150
frames (15s @ 10fps), a unique-object count that's a small plausible number
of people (not in the hundreds - if it's huge, IDs are churning instead of
persisting, which would mean `persist=True` isn't taking effect), a
`person` line in the class breakdown, and `Saved /tmp_check.mp4`. Play the
output file to visually confirm boxes have `#<id>` labels that stay on the
same person as they walk.

- [ ] **Step 3: Commit**

```bash
git add Day30/tracking.py
git commit -m "Add Day-30 tracking.py core module (ByteTrack/BoT-SORT via Ultralytics)"
```

---

### Task 3: `coding_practice/01_track_videos.py` - run tracking on all 5 videos

**Files:**
- Create: `Day30/coding_practice/01_track_videos.py`

**Interfaces:**
- Consumes: `tracking.load_model`, `tracking.track_video`, `tracking.TRACKERS`
  from Task 2.
- Produces: `Day30/sample_outputs/<video_stem>_<tracker>.mp4` (10 files) and
  `Day30/sample_outputs/tracking_results.md` (summary table) - Task 6's
  README links to this table.

- [ ] **Step 1: Write the script.**

```python
"""
Day 30 coding practice - run tracking on every sample video with every
supported tracker, save annotated outputs, and report unique-object counts.

Fulfills the brief's "Coding Practice" checklist: load a model, run
tracking on >=5 videos, show a unique ID per object, count unique objects
per video, save the output videos.

Run: python coding_practice/01_track_videos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracking import TRACKERS, load_model, track_video  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "sample_videos"
OUT_DIR = ROOT / "sample_outputs"
VIDEOS = [
    "pedestrians_cctv.mp4",
    "pedestrians_crosswalk.mp4",
    "pedestrians_mall.mp4",
    "sports_soccer.mp4",
    "sports_basketball.mp4",
]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    model = load_model()
    rows = []

    for video_name in VIDEOS:
        in_path = VIDEO_DIR / video_name
        for tracker in TRACKERS:
            tracker_tag = tracker.split(".")[0]
            out_path = OUT_DIR / f"{Path(video_name).stem}_{tracker_tag}.mp4"
            result = track_video(model, in_path, out_path, tracker=tracker)
            class_summary = ", ".join(f"{k}={v}" for k, v in result.class_counts.items()) or "none"
            print(f"{video_name:<28} {tracker_tag:<10} "
                  f"{len(result.unique_ids):>3} unique  ({class_summary})  "
                  f"{result.elapsed_s:.1f}s")
            rows.append((video_name, tracker_tag, result.n_frames,
                         len(result.unique_ids), class_summary, f"{result.elapsed_s:.1f}s"))

    lines = ["| Video | Tracker | Frames | Unique objects | Per-class | Time |",
             "|---|---|---|---|---|---|"]
    for video_name, tracker_tag, n_frames, n_unique, class_summary, elapsed in rows:
        lines.append(f"| {video_name} | {tracker_tag} | {n_frames} | {n_unique} | "
                     f"{class_summary} | {elapsed} |")
    (OUT_DIR / "tracking_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'tracking_results.md'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real (this is the actual deliverable run, not
  just a smoke test).**

```bash
cd Day30
python coding_practice/01_track_videos.py
```

Expected: 10 lines printed (5 videos x 2 trackers), each with a small
plausible unique-object count and a `person` (pedestrian videos) or
`person`/`sports ball` (sports videos) class breakdown; `sample_outputs/`
ends up with 10 `.mp4` files plus `tracking_results.md`.

- [ ] **Step 3: Read `tracking_results.md` and eyeball it for anything
  wrong** (e.g. a unique-object count in the hundreds for a 15s clip means
  IDs are churning - would need investigating before moving on, not silently
  accepted).

- [ ] **Step 4: Commit**

```bash
git add Day30/coding_practice/01_track_videos.py Day30/sample_outputs/
git commit -m "Add Day-30 coding-practice tracking script + run outputs for all 5 videos"
```

---

### Task 4: `coding_practice/02_id_consistency_check.py` - ID-switch heuristic

**Files:**
- Create: `Day30/coding_practice/02_id_consistency_check.py`

**Interfaces:**
- Consumes: `tracking.load_model`, `tracking.track_video`, `tracking.TRACKERS`.
- Produces: `Day30/sample_outputs/id_consistency.md` - Task 6's README
  "Challenges" section references this to back up real (not invented)
  numbers about ID stability.

This is a heuristic, not a formal MOTA/IDF1 score (no ground-truth ID
annotations exist for these clips) - document it as such in the output.

- [ ] **Step 1: Write the script.**

```python
"""
Day 30 coding practice - heuristic ID-switch counter.

There's no ground-truth ID annotation for these clips, so this isn't a
formal MOTA/IDF1 score. Instead: for each track ID, find the last frame it
appeared in. Count it as a suspected "ID switch" if, within the next
GAP_FRAMES frames, a *new* track ID (one that had never appeared before)
first appears whose class matches and whose box center is within
DIST_THRESHOLD_FRAC of the frame's shorter side from where the old ID was
last seen. That pattern - one ID vanishing right as a nearby new one of the
same class appears - is what a tracker losing an object and relabeling it
looks like.

Run: python coding_practice/02_id_consistency_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracking import TRACKERS, load_model, track_video  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "sample_videos"
OUT_DIR = ROOT / "sample_outputs"
VIDEOS = [
    "pedestrians_cctv.mp4",
    "pedestrians_crosswalk.mp4",
    "pedestrians_mall.mp4",
    "sports_soccer.mp4",
    "sports_basketball.mp4",
]
GAP_FRAMES = 5
DIST_THRESHOLD_FRAC = 0.08


def _centroid(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def count_suspected_switches(tracks_per_frame: list[list], frame_shape_short_side: float) -> int:
    last_seen: dict[int, tuple[int, str, tuple[float, float]]] = {}
    first_seen_frame: dict[int, int] = {}
    threshold = frame_shape_short_side * DIST_THRESHOLD_FRAC
    switches = 0

    for frame_idx, boxes in enumerate(tracks_per_frame):
        for t in boxes:
            if t.track_id not in first_seen_frame:
                first_seen_frame[t.track_id] = frame_idx
                # a brand-new ID: check if it lines up with a recently-lost one
                cx, cy = _centroid(t.box)
                for old_id, (old_frame, old_class, (ox, oy)) in list(last_seen.items()):
                    if old_class != t.class_name:
                        continue
                    if 0 < frame_idx - old_frame <= GAP_FRAMES:
                        if math.hypot(cx - ox, cy - oy) <= threshold:
                            switches += 1
                            del last_seen[old_id]
                            break
            last_seen[t.track_id] = (frame_idx, t.class_name, _centroid(t.box))

    return switches


def main() -> None:
    model = load_model()
    lines = ["| Video | Tracker | Unique IDs | Suspected ID switches |",
             "|---|---|---|---|"]

    for video_name in VIDEOS:
        in_path = VIDEO_DIR / video_name
        for tracker in TRACKERS:
            tracker_tag = tracker.split(".")[0]
            out_path = OUT_DIR / f"_scratch_{Path(video_name).stem}_{tracker_tag}.mp4"
            result = track_video(model, in_path, out_path, tracker=tracker)
            import cv2
            cap = cv2.VideoCapture(str(in_path))
            short_side = min(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            switches = count_suspected_switches(result.tracks_per_frame, short_side)
            print(f"{video_name:<28} {tracker_tag:<10} "
                  f"{len(result.unique_ids):>3} unique  {switches} suspected switch(es)")
            lines.append(f"| {video_name} | {tracker_tag} | {len(result.unique_ids)} | {switches} |")
            out_path.unlink(missing_ok=True)  # this script only needs the stats, not the video

    lines.append("")
    lines.append("Heuristic, not a formal MOTA/IDF1 score - no ground-truth ID "
                 "annotations exist for these clips. A 'suspected switch' is a new "
                 "track ID appearing within "
                 f"{GAP_FRAMES} frames and {DIST_THRESHOLD_FRAC:.0%} of the frame's "
                 "shorter side of where a same-class ID was last seen.")
    (OUT_DIR / "id_consistency.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'id_consistency.md'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it.**

```bash
cd Day30
python coding_practice/02_id_consistency_check.py
```

Expected: a table with a small suspected-switch count per video/tracker
(0-3 is reasonable for these clips; a much larger number suggests a real
tracking problem worth looking at before writing the README's "Challenges"
section in Task 6).

- [ ] **Step 3: Commit**

```bash
git add Day30/coding_practice/02_id_consistency_check.py Day30/sample_outputs/id_consistency.md
git commit -m "Add Day-30 ID-consistency heuristic check across all videos/trackers"
```

---

### Task 5: `app.py` - Smart Object Tracking System (Streamlit)

**Files:**
- Create: `Day30/app.py`
- Create: `Day30/requirements.txt`
- Create: `Day30/runtime.txt`
- Create: `Day30/.streamlit/config.toml`

**Interfaces:**
- Consumes: `tracking.load_model`, `tracking.track_video`, `tracking.TRACKERS`,
  `tracking.DEFAULT_CONF`, `tracking.DEFAULT_IOU`, `tracking.bgr_to_rgb`.

- [ ] **Step 1: Write `requirements.txt`** (same floors as Day29's, proven
  working in this environment; no `roboflow`/`python-dotenv` needed here -
  Day30 doesn't train anything).

```
# Day 30 - Multi-Object Tracking (YOLOv8 + ByteTrack/BoT-SORT)

streamlit>=1.40.0
ultralytics>=8.3.0
opencv-python-headless>=4.9.0
numpy>=1.26.0
pillow>=10.0.0
pandas>=2.0.0

# Video output as browser-playable H.264 mp4 (see tracking.py docstring)
imageio[ffmpeg]>=2.34.0
```

- [ ] **Step 2: Write `runtime.txt`** (copy Day29's exactly - same Python
  version this repo already targets).

```bash
cp Day29/runtime.txt Day30/runtime.txt
```

- [ ] **Step 3: Write `.streamlit/config.toml`** (same theme as Day27/Day29
  for visual consistency across the course deliverables).

```toml
[server]
maxUploadSize = 300

[theme]
base = "light"
primaryColor = "#3a86ff"
```

- [ ] **Step 4: Write `app.py`.**

```python
"""
Day 30 - Smart Object Tracking System (Streamlit).

Upload a video or pick a sample, choose a tracker (ByteTrack or BoT-SORT),
run YOLOv8 multi-object tracking, and see every object's ID + confidence
persist across the clip - including through crossings and occlusion.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from tracking import (DEFAULT_CONF, DEFAULT_IOU, TRACKERS, bgr_to_rgb,
                      load_model, track_video)

ROOT = Path(__file__).resolve().parent
VIDEO_SAMPLES = ROOT / "sample_videos"
WEIGHTS_PATH = ROOT / "yolov8n.pt"
MAX_VIDEO_FRAMES = 300  # cap so an uploaded video can't stall a free CPU host

st.set_page_config(page_title="Smart Object Tracking System", page_icon="\U0001F3AF", layout="wide")


@st.cache_data(show_spinner=False)
def sample_video_paths() -> dict[str, str]:
    return {p.stem: str(p) for p in sorted(VIDEO_SAMPLES.glob("*.mp4"))}


@st.cache_resource(show_spinner="Loading YOLOv8n...")
def get_model():
    return load_model(str(WEIGHTS_PATH))


def tracks_table(result) -> pd.DataFrame:
    ids = sorted(result.unique_ids)
    if not ids:
        return pd.DataFrame(columns=["Track ID", "Class", "Best confidence", "Frames visible"])
    id_class = result.id_to_class
    id_conf = result.id_best_conf
    id_frames = result.id_frame_count
    return pd.DataFrame({
        "Track ID": ids,
        "Class": [id_class[i] for i in ids],
        "Best confidence": [round(id_conf[i], 3) for i in ids],
        "Frames visible": [id_frames[i] for i in ids],
    })


st.title("\U0001F3AF Smart Object Tracking System")

if not WEIGHTS_PATH.exists():
    st.error("yolov8n.pt not found next to app.py.")
    st.stop()

model = get_model()

st.markdown(
    "Tracks people and objects across a video with **Ultralytics YOLOv8** "
    "+ your choice of **ByteTrack** or **BoT-SORT** - each object keeps one "
    "ID (and one box colour) for as long as it's visible, including while "
    "crossing paths with other objects."
)

with st.sidebar:
    st.header("Tracker")
    tracker = st.selectbox("Algorithm", TRACKERS,
                           format_func=lambda t: {"bytetrack.yaml": "ByteTrack",
                                                  "botsort.yaml": "BoT-SORT"}[t])
    conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)
    iou = st.slider("IoU threshold (NMS)", 0.10, 0.90, DEFAULT_IOU, 0.05)

    st.header("Input")
    samples = sample_video_paths()
    source = st.radio("Source", ["Sample", "Upload your own"], label_visibility="collapsed")

    video_path = None
    video_name = "video"
    if source == "Sample":
        if not samples:
            st.error("No sample videos found in sample_videos/.")
        else:
            choice = st.selectbox("Sample video", list(samples))
            video_path = samples[choice]
            video_name = choice
    else:
        upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
        if upload:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix)
            tmp.write(upload.getvalue())
            tmp.close()
            video_path = tmp.name
            video_name = Path(upload.name).stem

if video_path is None:
    st.info("Pick a sample video in the sidebar, or upload your own, to run tracking.")
    st.stop()

st.caption(f"Processing is capped at {MAX_VIDEO_FRAMES} frames on this hosted app "
          "so one upload can't stall it for other users.")

if st.button("Run tracking on this video", type="primary"):
    progress = st.progress(0.0, text="Starting...")

    def on_progress(done: int, total: int) -> None:
        progress.progress(min(done / max(total, 1), 1.0), text=f"Processing frame {done}/{total}")

    out_path = Path(tempfile.gettempdir()) / f"{video_name}_tracked.mp4"
    with st.spinner("Running tracking on every frame..."):
        result = track_video(model, video_path, out_path, tracker=tracker, conf=conf, iou=iou,
                             max_frames=MAX_VIDEO_FRAMES, progress_cb=on_progress)
    progress.empty()

    cols = st.columns(4)
    cols[0].metric("Frames processed", result.n_frames)
    cols[1].metric("Unique objects tracked", len(result.unique_ids))
    cols[2].metric("Classes seen", len(result.class_counts))
    cols[3].metric("Processing time", f"{result.elapsed_s:.1f} s")

    st.video(str(out_path))
    st.download_button("Download annotated video (MP4)", data=out_path.read_bytes(),
                       file_name=f"{video_name}_tracked.mp4", mime="video/mp4")

    tab_table, tab_chart = st.tabs(["Tracked Objects", "Per-Class Counts"])
    with tab_table:
        st.caption("One row per unique track ID - not per frame.")
        st.dataframe(tracks_table(result), use_container_width=True, hide_index=True)
    with tab_chart:
        if result.class_counts:
            st.bar_chart(result.class_counts)
        else:
            st.warning("Nothing tracked at this confidence threshold - try lowering it.")

st.divider()
st.caption("Day 30 - YOLOv8n (COCO) + Ultralytics' built-in ByteTrack/BoT-SORT trackers. "
          "Box colour is keyed off track ID, not class, so one object keeps one colour "
          "for its whole appearance in the clip.")
```

- [ ] **Step 5: Start the app and verify it in the browser** (per this
  project's standing rule: UI changes get verified in a real browser, not
  just "the code looks right").

```bash
cd Day30 && streamlit run app.py --server.headless true --server.port 8501
```

Then, using the browser tooling: open `http://localhost:8501`, select the
"Sample" source, pick `pedestrians_crosswalk` (a dense-crowd clip - the best
visual proof IDs survive crossings), pick "ByteTrack", click "Run tracking
on this video", wait for it to finish, and confirm: the 4 metrics render
with plausible numbers, the annotated video plays and shows `#<id>` labels
on boxes, the "Tracked Objects" table lists multiple track IDs with a
`person` class, the per-class bar chart renders, and the download button
produces a non-empty file. Then switch the tracker dropdown to "BoT-SORT"
and re-run once to confirm that path also works.

- [ ] **Step 6: Commit**

```bash
git add Day30/app.py Day30/requirements.txt Day30/runtime.txt Day30/.streamlit/
git commit -m "Add Day-30 Smart Object Tracking System Streamlit app"
```

---

### Task 6: `README.md` + `HOW_TO_RUN.txt`

**Files:**
- Create: `Day30/README.md`
- Create: `Day30/HOW_TO_RUN.txt`

**Interfaces:**
- Consumes: real numbers from `Day30/sample_outputs/tracking_results.md`
  and `Day30/sample_outputs/id_consistency.md` (Tasks 3-4) - the README's
  "challenges faced" and algorithm-choice sections must cite the actual
  measured results, not invented ones (matches Day29's README style, which
  documents real measured numbers throughout).

- [ ] **Step 1: Read both result files produced in Tasks 3-4** to pull real
  numbers into the README (unique-object counts, processing times per
  tracker, suspected-switch counts, which tracker was faster/more stable on
  which clip).

- [ ] **Step 2: Write `README.md`.** Must answer, explicitly, every question
  the brief asks for:
  - What is object tracking? (assigning a persistent identity to a
    detected object across frames, vs. detecting fresh each frame)
  - Difference between detection and tracking (Day27/29 did per-frame
    detection with no cross-frame identity; tracking adds identity +
    motion continuity - illustrate with the same object's ID surviving an
    occlusion in one of the actual result videos)
  - Which tracking algorithm was used, and why both are offered (ByteTrack
    default - lighter, associates even low-confidence boxes so it recovers
    from brief occlusion without extra appearance modeling; BoT-SORT adds a
    ReID/appearance-based association step - use the actual measured
    processing-time difference from `tracking_results.md` here, not a
    guessed number)
  - Real challenges faced (occlusion-driven ID switches - cite the actual
    `id_consistency.md` numbers per clip; tuning confidence threshold to
    avoid short-lived spurious IDs from flickering low-confidence
    detections; CPU-only video processing speed)
  - Project layout, dataset table (5 videos, categories, sources - copy
    from the design spec at
    `docs/superpowers/specs/2026-08-31-day30-object-tracking-design.md`),
    setup/run instructions
  - A `**Live demo:**` and `**APP LINK:**` placeholder exactly like
    Day29's README (`_(recording + hosted URL added after deployment - see
    HOW_TO_RUN.txt)_` / `_(added after deploying to Streamlit Community
    Cloud)_`) - deployment is explicitly out of scope for this session per
    the design spec.

- [ ] **Step 3: Write `HOW_TO_RUN.txt`** - copy Day29's structure (setup,
  `pip install -r requirements.txt`, `streamlit run app.py`, how to
  re-run the coding_practice scripts) adapted for Day30's actual files -
  no Roboflow/`.env` section needed here since Day30 doesn't train
  anything or need an API key.

- [ ] **Step 4: Self-review against the brief** - re-read the original
  Day-30 task text and confirm every deliverable it lists has a
  corresponding file in `Day30/`: tracking script (Task 3), Gradio/Streamlit
  app (Task 5), `requirements.txt` (Task 5), `README.md` (this task),
  sample input videos (already done), output videos (Task 3), GitHub repo
  link (this is the repo itself), HF Space/Streamlit URL (placeholder, per
  design spec), screen recording (placeholder, per design spec - cannot be
  produced by this session).

- [ ] **Step 5: Commit**

```bash
git add Day30/README.md Day30/HOW_TO_RUN.txt
git commit -m "Add Day-30 README and setup instructions"
```

---

## Self-Review Notes (completed while writing this plan)

- **Spec coverage:** every section of the design spec maps to a task -
  model/weights (Task 1), tracking core + both trackers (Task 2), "run on
  >=5 videos, unique IDs, save outputs" coding practice (Task 3), the
  brief's "Make Sure" ID-consistency requirement (Task 4), the Streamlit
  mini-project (Task 5), README's required questions + deployment/recording
  placeholders (Task 6).
- **Type consistency:** `TrackedBox`, `VideoTrackResult`, `track_video()`,
  `load_model()`, `TRACKERS` are defined once in Task 2 and referenced with
  identical names/signatures in Tasks 3, 4, 5 - checked.
- **No placeholders:** every step above has real, runnable code or an exact
  shell command with a concrete expected output - checked.
