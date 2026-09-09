from __future__ import annotations

import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
MAX_SIDE = 1280
TRAIL_LENGTH = 18
PERSON_CLASS_NAME = "person"


@dataclass
class PersonTrack:
    track_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]
    centroid: tuple[int, int]


@dataclass
class CountingLine:
    orientation: str = "horizontal"  # horizontal or vertical
    position: float = 0.5

    def y_value(self, frame_h: int) -> int:
        return int(frame_h * self.position)

    def x_value(self, frame_w: int) -> int:
        return int(frame_w * self.position)


@dataclass
class ROI:
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    def contains(self, point: tuple[int, int], frame_shape: tuple[int, int, int]) -> bool:
        h, w = frame_shape[:2]
        x, y = point
        x1 = int(self.x_min * w)
        x2 = int(self.x_max * w)
        y1 = int(self.y_min * h)
        y2 = int(self.y_max * h)
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class PeopleCountResult:
    out_path: Path
    n_frames: int
    fps: float
    elapsed_s: float
    people_per_frame: list[int]
    peak_count: int
    current_count: int
    total_people_seen: int
    line_crossings: int
    roi_count: int


_MODEL_CACHE: dict[str, YOLO] = {}


def load_model(model_name: str = DEFAULT_MODEL) -> YOLO:
    if model_name not in _MODEL_CACHE:
        model = YOLO(model_name)
        model.fuse(verbose=False)
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def _track_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 0.6180339887) % 1.0
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)


def _extract_people(result, names: dict[int, str]) -> list[PersonTrack]:
    tracked: list[PersonTrack] = []
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return tracked

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf_val, cls_id, tid in zip(xyxy, confs, cls_ids, ids):
        label = names.get(int(cls_id), "").lower()
        if label != PERSON_CLASS_NAME:
            continue
        px1, py1, px2, py2 = map(int, (x1, y1, x2, y2))
        cx = (px1 + px2) // 2
        cy = (py1 + py2) // 2
        tracked.append(
            PersonTrack(
                track_id=int(tid),
                class_name=PERSON_CLASS_NAME,
                confidence=float(conf_val),
                box=(px1, py1, px2, py2),
                centroid=(cx, cy),
            )
        )

    tracked.sort(key=lambda t: t.track_id)
    return tracked


def _draw_track(image: np.ndarray, person: PersonTrack, trail: deque, show_id: bool = True) -> None:
    x1, y1, x2, y2 = person.box
    color = _track_color(person.track_id)
    thickness = max(2, min(image.shape[:2]) // 180)

    for i in range(1, len(trail)):
        alpha = i / len(trail)
        pt_color = tuple(int(c * alpha) for c in color)
        cv2.line(image, trail[i - 1], trail[i], pt_color, max(1, thickness // 2), cv2.LINE_AA)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    label = f"#{person.track_id} {PERSON_CLASS_NAME} {person.confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(image.shape[:2]) / 900)
    (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)
    ty1 = max(y1 - th - baseline - 6, 0)
    ty2 = ty1 + th + baseline + 6
    cv2.rectangle(image, (x1, ty1), (x1 + tw + 8, ty2), color, -1)
    text_color = _text_color_for(color)
    cv2.putText(image, label, (x1 + 4, ty2 - baseline - 2), font, scale, text_color, 1, cv2.LINE_AA)

    if show_id:
        cv2.putText(image, str(person.track_id), (person.centroid[0], person.centroid[1]), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_overlay(image: np.ndarray, count: int, peak: int, total_seen: int = 0, roi_count: int | None = None, line_crossings: int | None = None) -> None:
    overlay = image.copy()
    text = f"People in frame: {count}   Peak so far: {peak}   Total seen: {total_seen}"
    if roi_count is not None:
        text += f"   ROI people: {roi_count}"
    if line_crossings is not None:
        text += f"   Line crossings: {line_crossings}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, min(image.shape[:2]) / 900)
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    pad = 12
    cv2.rectangle(overlay, (0, 0), (tw + pad * 2, th + baseline + pad * 2), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, dst=image)
    cv2.putText(image, text, (pad, th + pad), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_line(image: np.ndarray, line: CountingLine | None) -> None:
    if line is None:
        return
    h, w = image.shape[:2]
    if line.orientation == "horizontal":
        y = line.y_value(h)
        cv2.line(image, (0, y), (w, y), (0, 255, 255), 2, cv2.LINE_AA)
    else:
        x = line.x_value(w)
        cv2.line(image, (x, 0), (x, h), (0, 255, 255), 2, cv2.LINE_AA)


def _draw_roi(image: np.ndarray, roi: ROI | None) -> None:
    if roi is None:
        return
    h, w = image.shape[:2]
    x1 = int(roi.x_min * w)
    x2 = int(roi.x_max * w)
    y1 = int(roi.y_min * h)
    y2 = int(roi.y_max * h)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "ROI", (x1 + 6, max(y1 + 20, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)


def _compute_line_crossings(people: list[PersonTrack], current_side_map: dict[int, bool], line: CountingLine | None, frame_shape: tuple[int, int, int]) -> tuple[int, dict[int, bool]]:
    if line is None:
        return 0, current_side_map

    h, w = frame_shape[:2]
    crossings = 0
    for person in people:
        cx, cy = person.centroid
        if line.orientation == "horizontal":
            line_value = line.y_value(h)
            side = cy < line_value
        else:
            line_value = line.x_value(w)
            side = cx < line_value

        prev_side = current_side_map.get(person.track_id)
        if prev_side is not None and prev_side != side:
            crossings += 1
        current_side_map[person.track_id] = side

    return crossings, current_side_map


def process_frame_for_people(frame: np.ndarray, model: YOLO, conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU, tracker: str = DEFAULT_TRACKER, line: CountingLine | None = None, roi: ROI | None = None) -> tuple[np.ndarray, list[PersonTrack], int, int, dict[int, bool]]:
    result = model.track(frame, persist=True, tracker=tracker, conf=conf, iou=iou, classes=[0], verbose=False)[0]
    people = _extract_people(result, result.names)
    unique_ids = {person.track_id for person in people}
    people_by_id = {person.track_id: person for person in people}
    frame_h, frame_w = frame.shape[:2]

    side_map: dict[int, bool] = {}
    if line is not None:
        _, side_map = _compute_line_crossings(people, {}, line, frame.shape)

    roi_count = 0
    for person in people:
        if roi is not None and roi.contains(person.centroid, frame.shape):
            roi_count += 1

    annotated = frame.copy()
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    for person in people:
        trails[person.track_id].append(person.centroid)

    for person in people:
        _draw_track(annotated, person, trails[person.track_id])

    if line is not None:
        _draw_line(annotated, line)
    if roi is not None:
        _draw_roi(annotated, roi)

    count = len(people)
    overlay_text = f"People in frame: {count} | Unique IDs visible: {len(unique_ids)}"
    cv2.putText(annotated, overlay_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated, people, count, roi_count, side_map


def process_video(model: YOLO, in_path: str | Path, out_path: str | Path, conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                  tracker: str = DEFAULT_TRACKER, line: CountingLine | None = None, roi: ROI | None = None,
                  progress_cb: Callable[[int, int], None] | None = None, max_frames: int | None = None) -> PeopleCountResult:
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    seen_ids: set[int] = set()
    people_per_frame: list[int] = []
    peak_count = 0
    line_crossings = 0
    current_side_map: dict[int, bool] = {}
    start = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (min(frame.shape[1], MAX_SIDE), min(frame.shape[0], MAX_SIDE)))

            result = model.track(frame, persist=True, tracker=tracker, conf=conf, iou=iou, classes=[0], verbose=False)[0]
            people = _extract_people(result, result.names)
            seen_ids.update(person.track_id for person in people)
            count = len(people)
            people_per_frame.append(count)
            peak_count = max(peak_count, count)

            if line is not None:
                crossings, current_side_map = _compute_line_crossings(people, current_side_map, line, frame.shape)
                line_crossings += crossings

            roi_count = 0
            if roi is not None:
                roi_count = sum(1 for person in people if roi.contains(person.centroid, frame.shape))

            annotated = frame.copy()
            trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
            for person in people:
                trails[person.track_id].append(person.centroid)
            for person in people:
                _draw_track(annotated, person, trails[person.track_id])
            if line is not None:
                _draw_line(annotated, line)
            if roi is not None:
                _draw_roi(annotated, roi)
            _draw_overlay(annotated, count, peak_count, total_seen=len(seen_ids), roi_count=roi_count, line_crossings=line_crossings)

            writer.write(annotated)
            frame_idx += 1
            if progress_cb:
                progress_cb(frame_idx, n_total or frame_idx)
            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    elapsed_s = time.perf_counter() - start
    current_count = people_per_frame[-1] if people_per_frame else 0
    return PeopleCountResult(
        out_path=out_path,
        n_frames=frame_idx,
        fps=fps,
        elapsed_s=elapsed_s,
        people_per_frame=people_per_frame,
        peak_count=peak_count,
        current_count=current_count,
        total_people_seen=len(seen_ids),
        line_crossings=line_crossings,
        roi_count=roi_count if 'roi_count' in locals() else 0,
    )


def process_image(model: YOLO, image_path: str | Path, out_path: str | Path, conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                  tracker: str = DEFAULT_TRACKER, line: CountingLine | None = None, roi: ROI | None = None) -> tuple[np.ndarray, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    result = model(image, conf=conf, iou=iou, classes=[0], verbose=False)[0]
    people = _extract_people(result, result.names)
    annotated = image.copy()
    for person in people:
        x1, y1, x2, y2 = person.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _track_color(person.track_id), 2)
        label = f"#{person.track_id} {PERSON_CLASS_NAME} {person.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if line is not None:
        _draw_line(annotated, line)
    if roi is not None:
        _draw_roi(annotated, roi)
    _draw_overlay(annotated, len(people), len(people))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    return annotated, len(people)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect and count people with YOLO + ByteTrack.")
    parser.add_argument("--video", type=str, help="Input video path")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--out", type=str, default="output/people_counted.mp4", help="Output file path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--tracker", type=str, default=DEFAULT_TRACKER)
    args = parser.parse_args()

    if not args.video and not args.image:
        raise SystemExit("Provide either --video or --image.")

    model = load_model(DEFAULT_MODEL)
    if args.video:
        result = process_video(model, args.video, args.out, conf=args.conf, iou=args.iou, tracker=args.tracker)
        print(f"Processed {result.n_frames} frames; peak count = {result.peak_count}; output = {result.out_path}")
    else:
        annotated, count = process_image(model, args.image, args.out, conf=args.conf, iou=args.iou, tracker=args.tracker)
        print(f"Detected {count} people; saved to {args.out}")
