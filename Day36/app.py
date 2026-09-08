"""
Day 36 - Traffic Violation Detection System (Streamlit).

Upload a traffic video or pick a sample, define the road's normal direction
and an optional restricted zone (a preset per sample video, or sliders with
a live preview for your own), then run YOLOv8n + ByteTrack detection and
tracking to find wrong-way and restricted-zone violations - as either the
"Wrong-Way Detection" or "Traffic Violation Analytics" demo variant.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from analytics import traffic_status
from tracker import DEFAULT_CONF, DEFAULT_IOU, load_model
from traffic_violation import COMPASS, RegionBox, RoadDirection, WRONG_WAY, load_config, process_video

ROOT = Path(__file__).resolve().parent
VIDEO_SAMPLES = ROOT / "sample_videos"
WEIGHTS_PATH = ROOT / "yolov8n.pt"
MAX_VIDEO_FRAMES = 300  # cap so an uploaded video can't stall a free CPU host

st.set_page_config(page_title="Traffic Violation Detection System", page_icon="\U0001F6A6", layout="wide")


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def sample_video_paths() -> dict[str, str]:
    return {p.stem: str(p) for p in sorted(VIDEO_SAMPLES.glob("*.mp4"))}


@st.cache_resource(show_spinner="Loading detector...")
def get_model(weights_path: str):
    return load_model(weights_path)


@st.cache_data(show_spinner=False)
def first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def preview_with_overlay(frame_rgb, direction: RoadDirection, zone: RegionBox | None,
                          roi: RegionBox | None):
    import numpy as np
    preview = frame_rgb.copy()
    h, w = preview.shape[:2]
    if roi is not None:
        pt1 = (int(roi.x_min * w), int(roi.y_min * h))
        pt2 = (int(roi.x_max * w), int(roi.y_max * h))
        cv2.rectangle(preview, pt1, pt2, (80, 160, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, "direction check area", (pt1[0] + 4, max(pt1[1] + 18, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 160, 255), 2, cv2.LINE_AA)
    if zone is not None:
        pt1 = (int(zone.x_min * w), int(zone.y_min * h))
        pt2 = (int(zone.x_max * w), int(zone.y_max * h))
        overlay = preview.copy()
        cv2.rectangle(overlay, pt1, pt2, (255, 140, 0), -1)
        preview = cv2.addWeighted(overlay, 0.3, preview, 0.7, 0)
        cv2.rectangle(preview, pt1, pt2, (255, 140, 0), 2, cv2.LINE_AA)
        cv2.putText(preview, "restricted zone", (pt1[0] + 4, max(pt1[1] + 18, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2, cv2.LINE_AA)
    cx, cy = 70, 90
    dx, dy = direction.vector
    tip = (int(cx + dx * 44), int(cy + dy * 44))
    cv2.circle(preview, (cx, cy), 58, (30, 30, 30), -1)
    cv2.arrowedLine(preview, (cx, cy), tip, (255, 255, 255), 4, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(preview, "normal dir.", (cx - 46, cy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def events_table(events) -> pd.DataFrame:
    return pd.DataFrame({
        "Vehicle ID": [f"#{e.track_id}" for e in events],
        "Type": [e.class_name for e in events],
        "Violation": ["Wrong-way" if e.violation_type == WRONG_WAY else "Restricted zone" for e in events],
        "Time (s)": [round(e.time_s, 1) for e in events],
        "Frame": [e.frame_idx for e in events],
    })


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("\U0001F6A6 Traffic Violation Detection System")

st.markdown(
    "Detect and track vehicles with YOLOv8n + **ByteTrack**, then check each one against two "
    "independent rules: is it moving against the road's configured normal direction (**wrong-way**), "
    "and has it entered a marked **restricted zone**. A violation is recorded once per vehicle ID, "
    "not once per frame - see README.md for how duplicates are avoided."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Demo variant")
    variant_label = st.radio("Variant", ["Wrong-Way Detection", "Traffic Violation Analytics"],
                              label_visibility="collapsed")
    variant = "wrong_way" if variant_label == "Wrong-Way Detection" else "analytics"
    st.caption(
        "**Wrong-Way Detection**: full-frame overlay - vehicle boxes, IDs, heading arrows, a fixed "
        "normal-direction compass, red highlighting for confirmed wrong-way vehicles."
        if variant == "wrong_way" else
        "**Traffic Violation Analytics**: minimal vehicle dots + a docked dashboard - totals, a "
        "traffic-status banner, per-class counts, a violation event log, a trend sparkline."
    )

    st.header("Input")
    samples = sample_video_paths()
    source = st.radio("Source", ["Sample", "Upload your own"], label_visibility="collapsed")

    video_path = None
    video_name = "video"
    if source == "Sample":
        if not samples:
            st.error("No sample videos found in sample_videos/.")
        else:
            names = list(samples)
            choice = st.selectbox("Sample video", names,
                                  help="Each sample ships a hand-calibrated normal direction + restricted "
                                       "zone. All 4 were picked to cover different road directions - see "
                                       "README.md.")
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

    road_direction: RoadDirection | None = None
    zone: RegionBox | None = None
    direction_roi: RegionBox | None = None

    if video_path is not None:
        st.header("Road direction & zone")
        preset_path = VIDEO_SAMPLES / f"{video_name}_config.json"
        use_preset = False
        if source == "Sample" and preset_path.exists():
            use_preset = st.checkbox(f"Use preset config ({preset_path.name})", value=True)

        if use_preset:
            road_direction, zone, direction_roi = load_config(preset_path)
            st.caption(f"Normal direction: **{road_direction.name}** {road_direction.arrow}"
                       + ("  |  zone configured" if zone else "  |  no zone")
                       + ("  |  direction check scoped to one lane" if direction_roi else ""))
        else:
            compass_names = list(COMPASS)
            dir_name = st.selectbox("Normal traffic direction", compass_names,
                                    index=compass_names.index("down"),
                                    help="The direction vehicles are expected to travel. A vehicle whose "
                                         "recent movement disagrees with this by a wide, sustained margin "
                                         "is flagged wrong-way.")
            road_direction = RoadDirection(name=dir_name)

            add_zone = st.checkbox("Add a restricted zone", value=True)
            if add_zone:
                zx = st.slider("Zone - x range", 0.0, 1.0, (0.65, 0.85), 0.01)
                zy = st.slider("Zone - y range", 0.0, 1.0, (0.35, 0.95), 0.01)
                zone = RegionBox(zx[0], zx[1], zy[0], zy[1])

            restrict_roi = st.checkbox("Scope wrong-way check to one region", value=False,
                                       help="Useful on a divided road: limits the wrong-way rule to "
                                            "vehicles inside this box, so the opposite (legitimately "
                                            "oncoming) carriageway isn't misread as a mass violation.")
            if restrict_roi:
                rx = st.slider("Direction-check area - x range", 0.0, 1.0, (0.0, 1.0), 0.01)
                ry = st.slider("Direction-check area - y range", 0.0, 1.0, (0.0, 1.0), 0.01)
                direction_roi = RegionBox(rx[0], rx[1], ry[0], ry[1])

        frame_rgb = first_frame(video_path)
        if frame_rgb is not None and road_direction is not None:
            st.image(preview_with_overlay(frame_rgb, road_direction, zone, direction_roi),
                     caption="Direction / zone preview", use_container_width=True)

        st.header("Detection")
        conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)
        iou = st.slider("IoU threshold (NMS)", 0.10, 0.90, DEFAULT_IOU, 0.05)

        st.header("Wrong-way sensitivity")
        angle_threshold = st.slider("Angle threshold (degrees)", 60, 175, int(road_direction.angle_threshold_deg), 5,
                                    help="How far a vehicle's heading must diverge from the normal direction "
                                         "before it counts as wrong-way this frame.")
        confirm_ratio = st.slider("Confirmation ratio", 0.3, 1.0, road_direction.confirm_ratio, 0.05,
                                  help="Fraction of the last 10 frames that must agree before a wrong-way "
                                       "flag is confirmed - avoids flagging a single noisy frame.")
        road_direction.angle_threshold_deg = float(angle_threshold)
        road_direction.confirm_ratio = float(confirm_ratio)

if video_path is None:
    st.info("Pick a sample video in the sidebar, or upload your own, to run violation detection.")
    st.stop()

try:
    model = get_model(str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else "yolov8n.pt")
except Exception as exc:
    st.error(f"Could not load the detector: {exc}")
    st.stop()

st.caption(f"Processing is capped at {MAX_VIDEO_FRAMES} frames on this hosted app "
          "so one upload can't stall it for other users.")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if st.button(f"Run {variant_label} on this video", type="primary"):
    progress = st.progress(0.0, text="Starting...")

    def on_progress(done: int, total: int) -> None:
        progress.progress(min(done / max(total, 1), 1.0), text=f"Processing frame {done}/{total}")

    out_path = Path(tempfile.gettempdir()) / f"{video_name}_{variant}.mp4"
    with st.spinner("Detecting vehicles and checking for violations..."):
        result = process_video(model, video_path, out_path, road_direction, zone, direction_roi,
                               variant=variant, conf=conf, iou=iou, max_frames=MAX_VIDEO_FRAMES,
                               progress_cb=on_progress)
    progress.empty()

    cols = st.columns(5)
    cols[0].metric("Total vehicles", result.total_vehicles)
    cols[1].metric("Total violations", result.total_violations)
    cols[2].metric("Wrong-way", result.wrong_way_count)
    cols[3].metric("Restricted zone", result.zone_count)
    status = traffic_status(result.total_violations)
    cols[4].metric("Traffic status", status)

    st.video(str(out_path))
    st.download_button("Download processed video (MP4)", data=out_path.read_bytes(),
                       file_name=f"{video_name}_{variant}.mp4", mime="video/mp4")

    tab_events, tab_classes, tab_trend = st.tabs(["Violation Events", "Vehicle Types", "Violations Over Time"])
    with tab_events:
        if result.events:
            st.dataframe(events_table(result.events), use_container_width=True, hide_index=True)
        else:
            st.success("No violations recorded on this clip.")
    with tab_classes:
        st.bar_chart(pd.Series(result.vehicle_counts_by_class, name="count"))
    with tab_trend:
        st.caption("Cumulative violation count over the clip - a flat line means a quiet stretch, "
                   "a step means a new violation was just confirmed.")
        counts = [0] * (result.n_frames + 1)
        for e in result.events:
            counts[e.frame_idx] += 1
        import itertools
        running = list(itertools.accumulate(counts))
        df = pd.DataFrame({"time_s": [i / result.fps for i in range(len(running))],
                           "total_violations": running}).set_index("time_s")
        st.line_chart(df)

st.divider()
st.caption("Day 36 - YOLOv8n + ByteTrack (same tracking stack as Day30/31/32). Wrong-way is decided by "
          "comparing each vehicle's net motion vector to a configured normal direction, debounced over a "
          "rolling window; restricted-zone violations trigger on the outside-to-inside edge, never while "
          "idling inside. See README.md for the full explanation and known limitations.")
