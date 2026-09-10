from datetime import datetime, timedelta
from pathlib import Path
import csv
import math
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import imageio_ffmpeg


def read_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def point_in_polygon(point, polygon):
    contour = np.asarray(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def validate_rois(rois, width, height):
    clean = []
    for roi in rois:
        name = str(roi["name"])
        pts = roi["points"]
        if len(pts) < 3:
            raise ValueError(f"ROI '{name}' must contain at least 3 points.")
        normalized = []
        for p in pts:
            if len(p) != 2:
                raise ValueError(f"ROI '{name}' contains an invalid point.")
            x, y = int(p[0]), int(p[1])
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            normalized.append([x, y])
        clean.append({"name": name, "points": normalized})
    return clean


def convert_to_browser_mp4(src_avi, dst_mp4, fps):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-i", src_avi,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-r", str(max(1, fps)),
        dst_mp4,
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError("FFmpeg conversion failed: " + completed.stderr[-1000:])


def process_video(video_path, rois, conf=0.35, process_every=1, min_stable_frames=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rois = validate_rois(rois, width, height)

    work_dir = Path(tempfile.mkdtemp(prefix="day38_result_"))
    raw_output = work_dir / "processed.avi"
    final_output = work_dir / "security_monitor_processed.mp4"
    csv_output = work_dir / "security_events.csv"

    writer = cv2.VideoWriter(
        str(raw_output),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Unable to create output video.")

    model = YOLO("yolo11n.pt")

    # State is maintained independently for every tracked person and ROI.
    states = {}
    sessions = {}
    events = []
    first_seen = {}
    unique_ids = set()
    max_active = 0
    entries = 0
    exits = 0
    frame_index = 0

    def now_for_frame(idx):
        return datetime.now() + timedelta(seconds=(idx / fps))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % process_every != 0:
            writer.write(frame)
            frame_index += 1
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=conf,
            verbose=False,
        )

        annotated = frame.copy()

        # Draw ROIs first.
        for roi in rois:
            pts = np.asarray(roi["points"], dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 2)
            anchor = tuple(pts[0])
            cv2.putText(
                annotated, roi["name"], (anchor[0], max(20, anchor[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
            )

        active_ids = set()

        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id
            ids = ids.cpu().numpy().astype(int) if ids is not None else []

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track_id = int(track_id)
                unique_ids.add(track_id)
                first_seen.setdefault(track_id, frame_index)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    annotated, f"Person #{track_id}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA
                )

                for roi in rois:
                    roi_name = roi["name"]
                    inside = point_in_polygon((cx, cy), roi["points"])
                    key = (track_id, roi_name)
                    state = states.setdefault(
                        key,
                        {"stable": False, "inside_count": 0, "outside_count": 0, "last_event_frame": -10**9},
                    )

                    if inside:
                        state["inside_count"] += 1
                        state["outside_count"] = 0
                    else:
                        state["outside_count"] += 1
                        state["inside_count"] = 0

                    # Hysteresis/stability prevents duplicate alerts caused by boundary jitter.
                    if not state["stable"] and state["inside_count"] >= min_stable_frames:
                        state["stable"] = True
                        active_ids.add(key)
                        entries += 1
                        t = now_for_frame(frame_index)
                        entry_time = t.strftime("%Y-%m-%d %H:%M:%S")
                        sessions[key] = {
                            "entry_time": entry_time,
                            "entry_frame": frame_index,
                        }
                        events.append({
                            "timestamp": entry_time,
                            "entry_time": entry_time,
                            "exit_time": "",
                            "frame": frame_index,
                            "track_id": track_id,
                            "roi": roi_name,
                            "event": "ENTRY",
                            "status": "active",
                            "duration_seconds": "",
                        })
                        state["last_event_frame"] = frame_index

                    elif state["stable"] and state["outside_count"] >= min_stable_frames:
                        state["stable"] = False
                        exits += 1
                        t = now_for_frame(frame_index)
                        exit_time = t.strftime("%Y-%m-%d %H:%M:%S")
                        session = sessions.pop(key, None)
                        entry_time = session["entry_time"] if session else ""
                        entry_frame = session["entry_frame"] if session else frame_index
                        duration = max(0, (frame_index - entry_frame) / fps)
                        events.append({
                            "timestamp": exit_time,
                            "entry_time": entry_time,
                            "exit_time": exit_time,
                            "frame": frame_index,
                            "track_id": track_id,
                            "roi": roi_name,
                            "event": "EXIT",
                            "status": "inactive",
                            "duration_seconds": round(duration, 2),
                        })
                        state["last_event_frame"] = frame_index

                    if state["stable"]:
                        active_ids.add(key)

        # Current active count is the number of stable person/ROI states.
        active_count = len(active_ids)
        max_active = max(max_active, active_count)

        # Dashboard overlay.
        cv2.rectangle(annotated, (10, 10), (330, 105), (0, 0, 0), -1)
        cv2.putText(annotated, "SECURITY MONITOR", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Active in ROI: {active_count}", (20, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Entries: {entries}  Exits: {exits}", (20, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated)
        frame_index += 1

    cap.release()
    writer.release()

    # Convert to H.264 MP4 for reliable Streamlit/browser playback.
    convert_to_browser_mp4(str(raw_output), str(final_output), fps)

    # Normalize CSV columns so rows with/without duration still align.
    columns = [
        "timestamp", "entry_time", "exit_time", "frame", "track_id",
        "roi", "event", "status", "duration_seconds"
    ]
    df = pd.DataFrame(events, columns=columns)
    df.to_csv(csv_output, index=False)

    return {
        "video_path": str(final_output),
        "csv_path": str(csv_output),
        "events": df,
        "unique_people": len(unique_ids),
        "entries": entries,
        "exits": exits,
        "max_active": max_active,
    }
