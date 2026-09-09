from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from people_counter import (DEFAULT_CONF, DEFAULT_IOU, CountingLine, ROI, load_model,
                           process_image, process_video)

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_videos"
MODEL_PATH = ROOT / "yolov8n.pt"

st.set_page_config(page_title="Smart People Counting System", page_icon="👥", layout="wide")
st.title("👥 Smart People Counting System")


@st.cache_resource(show_spinner="Loading YOLOv8n for people detection...")
def get_model():
    return load_model(str(MODEL_PATH))


try:
    model = get_model()
except Exception as exc:
    st.error(f"Could not load YOLO model: {exc}")
    st.stop()

st.markdown(
    "Detect people with YOLO, keep a stable track ID for each person, count people visible in each frame, "
    "and optionally count crossings across a line or inside a selected ROI."
)

with st.sidebar:
    st.header("Detection settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.90, DEFAULT_IOU, 0.05)

    st.header("Counting line")
    use_line = st.checkbox("Use counting line", value=False)
    line = None
    if use_line:
        orientation = st.radio("Orientation", ["horizontal", "vertical"], horizontal=True)
        position = st.slider("Line position", 0.05, 0.95, 0.5, 0.05)
        line = CountingLine(orientation=orientation, position=position)

    st.header("ROI")
    use_roi = st.checkbox("Count only inside ROI", value=False)
    roi = None
    if use_roi:
        x_min, x_max = st.slider("Horizontal range", 0.0, 1.0, (0.0, 1.0), 0.05)
        y_min, y_max = st.slider("Vertical range", 0.0, 1.0, (0.0, 1.0), 0.05)
        roi = ROI(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    st.header("Input source")
    source = st.radio("Source", ["Upload", "Sample"], horizontal=True)
    uploaded_file = None
    sample_path = None
    if source == "Sample" and SAMPLE_DIR.exists():
        sample_files = sorted(SAMPLE_DIR.glob("*.*"))
        if sample_files:
            chosen = st.selectbox("Sample video", [p.name for p in sample_files])
            sample_path = str(SAMPLE_DIR / chosen)
    if source == "Upload":
        uploaded_file = st.file_uploader("Upload an image or video", type=["png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv"])

    st.caption("Tip: use a short, clear video with visible people for best results.")

if source == "Sample" and sample_path is None:
    st.info("No sample videos were found in the sample_videos folder yet. Upload your own video or add sample clips to that folder.")
    st.stop()

if source == "Upload" and uploaded_file is None:
    st.info("Upload an image or video to analyze people in the frame.")
    st.stop()

if source == "Sample":
    input_path = sample_path
    input_name = Path(sample_path).stem
    input_kind = "video" if Path(sample_path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} else "image"
else:
    file_bytes = uploaded_file.read()
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        input_path = tmp.name
        input_name = Path(uploaded_file.name).stem
    input_kind = "video" if suffix in {".mp4", ".avi", ".mov", ".mkv"} else "image"

if st.button("Run analysis", type="primary"):
    output_path = ROOT / "outputs" / f"{input_name}_people_counted{Path(input_path).suffix or '.png'}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    progress = st.progress(0.0, text="Starting analysis...")

    def on_progress(done: int, total: int):
        value = 0.0 if total <= 0 else done / total
        progress.progress(min(value, 1.0), text=f"Processing frame {done}/{total}")

    if input_kind == "video":
        result = process_video(
            model=model,
            in_path=input_path,
            out_path=output_path,
            conf=conf,
            iou=iou,
            line=line,
            roi=roi,
            progress_cb=on_progress,
        )
        progress.empty()

        st.success(f"Processed {result.n_frames} frames and saved output video.")
        cols = st.columns(4)
        cols[0].metric("Total people seen", result.total_people_seen)
        cols[1].metric("Peak people", result.peak_count)
        cols[2].metric("Current frame", result.current_count)
        cols[3].metric("Line crossings", result.line_crossings)

        st.caption(f"Processing time: {result.elapsed_s:.2f}s")

        st.video(str(result.out_path))
        with open(result.out_path, "rb") as f:
            st.download_button("Download processed video", f.read(), file_name=result.out_path.name, mime="video/mp4")

        st.line_chart(result.people_per_frame)
    else:
        annotated, count = process_image(
            model=model,
            image_path=input_path,
            out_path=output_path,
            conf=conf,
            iou=iou,
            line=line,
            roi=roi,
        )
        progress.empty()
        st.success(f"Detected {count} people in the image.")
        st.image(annotated, channels="BGR", caption="Annotated image")
        with open(output_path, "rb") as f:
            st.download_button("Download processed image", f.read(), file_name=output_path.name, mime="image/png")

st.caption("Day 37 - People Counting & Crowd Analysis using YOLOv8n + ByteTrack.")
