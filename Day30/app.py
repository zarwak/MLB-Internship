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

from tracking import (DEFAULT_CONF, DEFAULT_IOU, TRACKERS, load_model,
                      track_video)

ROOT = Path(__file__).resolve().parent
VIDEO_SAMPLES = ROOT / "sample_videos"
WEIGHTS_PATH = ROOT / "yolov8n.pt"
MAX_VIDEO_FRAMES = 300  # cap so an uploaded video can't stall a free CPU host

TRACKER_LABELS = {"bytetrack.yaml": "ByteTrack", "botsort.yaml": "BoT-SORT"}

st.set_page_config(page_title="Smart Object Tracking System", page_icon="\U0001F3AF", layout="wide")


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Tracker")
    tracker = st.selectbox("Algorithm", TRACKERS, format_func=lambda t: TRACKER_LABELS[t])
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
    st.info("Pick a sample video in the sidebar, or upload your own, to run tracking.")
    st.stop()

st.caption(f"Processing is capped at {MAX_VIDEO_FRAMES} frames on this hosted app "
          "so one upload can't stall it for other users.")

# ---------------------------------------------------------------------------
# Run tracking
# ---------------------------------------------------------------------------

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
st.caption(f"Day 30 - YOLOv8n (COCO) + Ultralytics' built-in {TRACKER_LABELS[tracker]} tracker. "
          "Box colour is keyed off track ID, not class, so one object keeps one colour "
          "for its whole appearance in the clip.")
