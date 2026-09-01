"""
Day 31 - Smart Vehicle Counting System (Streamlit).

Upload a traffic video or pick a sample, position a counting line (and an
optional region of interest), run YOLOv8n + ByteTrack detection/tracking,
and get back an annotated video with a live running count - split by
vehicle class (car/motorcycle/bus/truck) and by crossing direction.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from counting import (DEFAULT_CONF, DEFAULT_IOU, ROI, CountingLine,
                      count_video, load_model)

ROOT = Path(__file__).resolve().parent
VIDEO_SAMPLES = ROOT / "sample_videos"
WEIGHTS_PATH = ROOT / "yolov8n.pt"
MAX_VIDEO_FRAMES = 300  # cap so an uploaded video can't stall a free CPU host

st.set_page_config(page_title="Smart Vehicle Counting System", page_icon="\U0001F697", layout="wide")


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def sample_video_paths() -> dict[str, str]:
    return {p.stem: str(p) for p in sorted(VIDEO_SAMPLES.glob("*.mp4"))}


@st.cache_resource(show_spinner="Loading YOLOv8n (first run downloads the ~6MB checkpoint)...")
def get_model():
    return load_model(str(WEIGHTS_PATH))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("\U0001F697 Smart Vehicle Counting System")

try:
    model = get_model()
except Exception as exc:
    st.error(f"Could not load yolov8n.pt: {exc}")
    st.stop()

st.markdown(
    "Detects and tracks vehicles with **YOLOv8n** + **ByteTrack**, then counts "
    "each one **exactly once** - the instant its track crosses a line you "
    "position - broken down by class (car / motorcycle / bus / truck) and by "
    "crossing direction. Counting by track ID, not by per-frame detection, is "
    "what keeps one vehicle visible for 100 frames from being counted 100 times."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Counting line")
    orientation = st.radio("Orientation", ["horizontal", "vertical"], horizontal=True,
                           help="Horizontal: line runs left-right; counts up/down crossings. "
                                "Vertical: line runs top-bottom; counts left/right crossings.")
    position = st.slider("Line position", 0.0, 1.0, 0.5, 0.05,
                         help="Where the line sits - a fraction of frame height (horizontal) "
                              "or width (vertical).")

    st.header("Region of interest")
    use_roi = st.checkbox("Restrict counting to a region", value=False,
                          help="Vehicles outside this box are still detected and drawn, "
                               "just never counted - useful for ignoring a parking lane, "
                               "sidewalk, or opposite carriageway visible in frame.")
    roi = None
    if use_roi:
        x_min, x_max = st.slider("Horizontal range", 0.0, 1.0, (0.0, 1.0), 0.05)
        y_min, y_max = st.slider("Vertical range", 0.0, 1.0, (0.0, 1.0), 0.05)
        roi = ROI(x_min, x_max, y_min, y_max)

    st.header("Detection")
    conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05,
                     help="Only detections the model is at least this sure about are tracked.")
    iou = st.slider("IoU threshold (NMS)", 0.10, 0.90, DEFAULT_IOU, 0.05,
                     help="Lower = more aggressive removal of duplicate overlapping boxes.")

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
    st.info("Pick a sample video in the sidebar, or upload your own, to run vehicle counting.")
    st.stop()

st.caption(f"Processing is capped at {MAX_VIDEO_FRAMES} frames on this hosted app "
          "so one upload can't stall it for other users.")

# ---------------------------------------------------------------------------
# Run counting
# ---------------------------------------------------------------------------

if st.button("Run vehicle counting on this video", type="primary"):
    progress = st.progress(0.0, text="Starting...")

    def on_progress(done: int, total: int) -> None:
        progress.progress(min(done / max(total, 1), 1.0), text=f"Processing frame {done}/{total}")

    line = CountingLine(orientation=orientation, position=position)
    out_path = Path(tempfile.gettempdir()) / f"{video_name}_counted.mp4"
    with st.spinner("Detecting, tracking, and counting vehicles..."):
        result = count_video(model, video_path, out_path, line=line, roi=roi, conf=conf, iou=iou,
                             max_frames=MAX_VIDEO_FRAMES, progress_cb=on_progress)
    progress.empty()

    cols = st.columns(4)
    cols[0].metric("Frames processed", result.n_frames)
    cols[1].metric("Total vehicles counted", result.total_count)
    cols[2].metric("Vehicle classes seen", len(result.counts_by_class))
    cols[3].metric("Processing time", f"{result.elapsed_s:.1f} s")

    st.video(str(out_path))
    st.download_button("Download annotated video (MP4)", data=out_path.read_bytes(),
                       file_name=f"{video_name}_counted.mp4", mime="video/mp4")

    tab_class, tab_dir = st.tabs(["Per-Class Counts", "Per-Direction Counts"])
    with tab_class:
        if result.counts_by_class:
            st.bar_chart(result.counts_by_class)
        else:
            st.warning("No vehicles crossed the line yet - try a lower confidence threshold, "
                       "a different line position, or a busier clip.")
    with tab_dir:
        if result.counts_by_direction:
            st.bar_chart(result.counts_by_direction)
        else:
            st.warning("No crossings recorded yet.")

st.divider()
st.caption("Day 31 - YOLOv8n (COCO) + ByteTrack. Each vehicle is counted at most once, the "
          "moment it crosses the line, keyed by its persistent track ID - the same ID "
          "persistence Day30 introduced is what makes duplicate-free counting possible.")
