"""
Day 32 - Smart Parking Occupancy Detection (Streamlit).

Upload a parking-lot video or pick a sample, define the parking spaces
(a preset layout for the sample clips, or a trapezoid+grid you tune with
sliders for your own video), then run YOLOv8n + ByteTrack detection and
tracking to see which spaces are occupied - as either the "Parking
Monitor" or "Parking Analytics" demo variant.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from analytics import LEVEL_COLORS, utilization_level
from parking_detection import (DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_IOU,
                               DEFAULT_OVERLAP_THRESHOLD, ParkingSpace,
                               generate_grid_layout, load_layout, load_model,
                               process_video)

ROOT = Path(__file__).resolve().parent
VIDEO_SAMPLES = ROOT / "sample_videos"
WEIGHTS_PATH = ROOT / "yolov8n.pt"
AERIAL_WEIGHTS_PATH = ROOT / "vehicle_detection_aerial.pt"
DEFAULT_AERIAL_OVERLAP = 0.15  # the aerial layouts are now calibrated to real per-stall size
                               # (measured car footprint from live detections, not hand-eyeballed
                               # zones) - see README "Real challenges faced" for the recalibration
                               # that made a stable, non-fragile default possible here
MAX_VIDEO_FRAMES = 300  # cap so an uploaded video can't stall a free CPU host

st.set_page_config(page_title="Smart Parking Occupancy Detection", page_icon="\U0001F17F", layout="wide")


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


def preview_with_grid(frame_rgb, layout: list[ParkingSpace]):
    preview = frame_rgb.copy()
    for space in layout:
        pts = space.pixel_polygon(preview.shape).astype(int)
        cv2.polylines(preview, [pts], True, (0, 220, 0), 2, cv2.LINE_AA)
    return preview


def spaces_table(statuses) -> pd.DataFrame:
    return pd.DataFrame({
        "Space": [s.space.id for s in statuses],
        "Status": ["Occupied" if s.occupied else "Free" for s in statuses],
        "Overlap": [round(s.overlap, 2) for s in statuses],
    })


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("\U0001F17F Smart Parking Occupancy Detection")

st.markdown(
    "Define the parking spaces as a grid over the lot, then run YOLOv8n + "
    "**ByteTrack** to detect and track vehicles and test how much of each space "
    "they cover. A space is occupied once a vehicle's overlap crosses a "
    "threshold - not \"is a box fully inside it\" - which is what lets a "
    "partially visible vehicle (cut off by the frame edge, half-hidden behind "
    "another car) still register correctly."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Demo variant")
    variant_label = st.radio("Variant", ["Parking Monitor", "Parking Analytics"], label_visibility="collapsed")
    variant = "monitor" if variant_label == "Parking Monitor" else "analytics"
    st.caption(
        "**Parking Monitor**: full-frame overlay - filled spaces, vehicle boxes, a counts badge."
        if variant == "monitor" else
        "**Parking Analytics**: minimal space markers + a docked dashboard - occupancy %, "
        "utilization level, a trend sparkline."
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
            default_idx = names.index("mall_curbside_lot") if "mall_curbside_lot" in names else 0
            choice = st.selectbox("Sample video", names, index=default_idx,
                                  help="All samples are genuinely static (fixed) cameras - no pan, "
                                       "tilt, or zoom - verified frame-to-frame before shipping. The "
                                       "aerial_* samples use a separate, custom-trained detector - see "
                                       "the note under 'Detection' below.")
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

    layout = None
    if video_path is not None:
        st.header("Parking spaces")
        preset_path = VIDEO_SAMPLES / f"{video_name}_spaces.json"
        use_preset = False
        if source == "Sample" and preset_path.exists():
            use_preset = st.checkbox(f"Use preset layout ({preset_path.name})", value=True)

        if use_preset:
            layout = load_layout(preset_path)
            st.caption(f"{len(layout)} spaces loaded from the preset.")
        else:
            st.caption("Trapezoid: the near (bottom) edge is usually wider than the far (top) "
                       "edge for an elevated camera looking down a row of spaces.")
            top_x = st.slider("Top edge (far) - x range", 0.0, 1.0, (0.15, 0.85), 0.01)
            bottom_x = st.slider("Bottom edge (near) - x range", 0.0, 1.0, (0.0, 1.0), 0.01)
            top_y = st.slider("Top edge (far) - y position", 0.0, 1.0, 0.40, 0.01)
            bottom_y = st.slider("Bottom edge (near) - y position", 0.0, 1.0, 0.80, 0.01)
            rows = st.slider("Rows", 1, 4, 1)
            cols = st.slider("Columns", 1, 16, 8)
            layout = generate_grid_layout(top_x, bottom_x, top_y, bottom_y, rows, cols)
            st.caption(f"{len(layout)} spaces generated.")

        frame_rgb = first_frame(video_path)
        if frame_rgb is not None:
            st.image(preview_with_grid(frame_rgb, layout), caption="Space layout preview", use_container_width=True)

    st.header("Detection")
    if source == "Sample":
        use_aerial_model = video_path is not None and video_name.startswith("aerial_")
        if video_path is not None:
            st.caption(
                "Using the **aerial-trained detector** - plain COCO YOLOv8n was measured to find "
                "~0 vehicles at this steep top-down angle, see README 'Challenges'."
                if use_aerial_model else
                "Using the **general-purpose detector** (YOLOv8n, COCO-pretrained)."
            )
    else:
        use_aerial_model = st.checkbox(
            "Use the aerial/top-down detector",
            help="Switch this on for drone or steep overhead footage. The general-purpose "
                 "COCO detector was measured to find ~0 vehicles at that camera angle - "
                 "see README 'Challenges'.")

    conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05,
                     help="Only detections the model is at least this sure about are tracked.")
    iou = st.slider("IoU threshold (NMS)", 0.10, 0.90, DEFAULT_IOU, 0.05,
                     help="Lower = more aggressive removal of duplicate overlapping boxes.")
    default_overlap = DEFAULT_AERIAL_OVERLAP if use_aerial_model else DEFAULT_OVERLAP_THRESHOLD
    overlap_threshold = st.slider("Occupancy overlap threshold", 0.02, 0.60, default_overlap, 0.01,
                                  key=f"overlap_{'aerial' if use_aerial_model else 'general'}",
                                  help="Fraction of a space's area a vehicle must cover to count as parked "
                                       "in it. Lower this if partially visible vehicles aren't registering, "
                                       "or if you upload your own layout with looser-drawn space boundaries.")
    imgsz = st.select_slider("Inference resolution", options=[640, 960, 1280, 1600, 1920, 2560],
                             value=DEFAULT_IMGSZ,
                             help="YOLO resizes every frame to this before looking at it, regardless of "
                                  "the video's real resolution - raise this for aerial/top-down footage "
                                  "where vehicles are small, at the cost of slower processing.")

if video_path is None:
    st.info("Pick a sample video in the sidebar, or upload your own, to run parking occupancy detection.")
    st.stop()

weights_path = AERIAL_WEIGHTS_PATH if use_aerial_model else WEIGHTS_PATH
try:
    model = get_model(str(weights_path))
except Exception as exc:
    st.error(f"Could not load {weights_path.name}: {exc}")
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
    with st.spinner("Detecting vehicles and checking space occupancy..."):
        result = process_video(model, video_path, out_path, layout, variant=variant, conf=conf, iou=iou,
                               overlap_threshold=overlap_threshold, imgsz=imgsz, max_frames=MAX_VIDEO_FRAMES,
                               progress_cb=on_progress)
    progress.empty()

    cols = st.columns(5)
    cols[0].metric("Total spaces", result.total_spaces)
    cols[1].metric("Occupied", result.final_occupied)
    cols[2].metric("Free", result.final_free)
    cols[3].metric("Occupancy", f"{result.final_occupancy_pct:.0f}%")
    cols[4].metric("Unique vehicles seen", result.unique_vehicles_seen)

    level = utilization_level(result.final_occupancy_pct)
    b, g, r = LEVEL_COLORS[level]
    st.markdown(f"**Utilization: <span style='color:rgb({r},{g},{b})'>{level}</span>**", unsafe_allow_html=True)

    st.video(str(out_path))
    st.download_button("Download annotated video (MP4)", data=out_path.read_bytes(),
                       file_name=f"{video_name}_{variant}.mp4", mime="video/mp4")

    tab_spaces, tab_trend = st.tabs(["Space Status", "Occupancy Over Time"])
    with tab_spaces:
        st.caption("Final status of every defined space at the end of the clip.")
        st.dataframe(spaces_table(result.final_statuses), use_container_width=True, hide_index=True)
    with tab_trend:
        history_df = pd.DataFrame({
            "time_s": [s.time_s for s in result.history],
            "occupancy_pct": [s.occupancy_pct for s in result.history],
        }).set_index("time_s")
        if len(history_df) > 1:
            st.line_chart(history_df)
        else:
            st.warning("Clip too short to plot a trend.")

st.divider()
st.caption("Day 32 - YOLOv8n + ByteTrack (COCO-pretrained for ground-level footage, a custom "
          "aerial-trained checkpoint for top-down footage), same tracking stack as Day30/31. "
          "Occupancy is decided by polygon-overlap ratio (cv2.intersectConvexConvex), not box "
          "containment, and each space debounces over a few frames before flipping state - see "
          "README.md for the full explanation.")
