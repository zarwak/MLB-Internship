"""
Day 27 - Smart Object Detection Application (Streamlit).

Upload an image or a short video, run a pretrained YOLO model, see every
detection's bounding box (a distinct colour per class), class name and
confidence score, and download the annotated result.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detection import (DEFAULT_CONF, DEFAULT_IOU, bgr_to_rgb, detect_image,
                       detect_video, encode_png, load_image, load_model,
                       resize_max_side)

ROOT = Path(__file__).resolve().parent
IMAGE_SAMPLES = ROOT / "sample_images"
VIDEO_SAMPLES = ROOT / "sample_videos"
MAX_SIDE = 960
MAX_VIDEO_FRAMES = 300  # cap so an uploaded video can't stall a free CPU host

st.set_page_config(page_title="Smart Object Detection (YOLO)", page_icon="\U0001F3AF", layout="wide")

MODEL_CHOICES = {
    "YOLO11n - fastest, recommended": "yolo11n.pt",
    "YOLOv8n - previous generation": "yolov8n.pt",
    "YOLO11s - small, more accurate & slower": "yolo11s.pt",
}


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def sample_image_paths() -> dict[str, str]:
    return {p.stem: str(p) for p in sorted(IMAGE_SAMPLES.glob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}


@st.cache_data(show_spinner=False)
def sample_video_paths() -> dict[str, str]:
    return {p.stem: str(p) for p in sorted(VIDEO_SAMPLES.glob("*"))
            if p.suffix.lower() in {".avi", ".mp4", ".mov"}}


@st.cache_resource(show_spinner="Loading YOLO model (first run downloads the weights)...")
def get_model(name: str):
    return load_model(name)


def decode_upload_image(upload) -> np.ndarray:
    data = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {upload.name} as an image")
    return resize_max_side(image, MAX_SIDE)


def detections_table(detections) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame(columns=["Class", "Confidence", "Box (x1,y1,x2,y2)"])
    return pd.DataFrame({
        "Class": [d.class_name for d in detections],
        "Confidence": [round(d.confidence, 3) for d in detections],
        "Box (x1,y1,x2,y2)": [str(d.box) for d in detections],
    })


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.title("\U0001F3AF Smart Object Detection Application")
st.markdown(
    "Detects everyday objects - people, vehicles, animals, furniture and more "
    "(80 [COCO](https://cocodataset.org/#explore) classes) - in an image or "
    "video using a **pretrained YOLO** model. This is pure inference: no "
    "training happens here, the model already knows these classes."
)

with st.sidebar:
    st.header("Model")
    model_label = st.selectbox("YOLO variant", list(MODEL_CHOICES), index=0)
    model_name = MODEL_CHOICES[model_label]

    conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05,
                     help="Only detections the model is at least this sure about are kept and drawn.")
    iou = st.slider("IoU threshold (NMS)", 0.10, 0.90, DEFAULT_IOU, 0.05,
                     help="Lower = more aggressive removal of duplicate overlapping boxes for the same object.")

    st.header("Input")
    mode = st.radio("Detect in", ["Image", "Video"], horizontal=True)
    source = st.radio("Source", ["Sample", "Upload your own"], label_visibility="collapsed")

model = get_model(model_name)

# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

if mode == "Image":
    samples = sample_image_paths()
    image = None
    image_name = "image"

    with st.sidebar:
        if source == "Sample":
            if not samples:
                st.error("No sample images found. Run `python download_samples.py` first.")
            else:
                choice = st.selectbox("Sample image", list(samples))
                image = resize_max_side(load_image(samples[choice]), MAX_SIDE)
                image_name = choice
        else:
            upload = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if upload:
                image = decode_upload_image(upload)
                image_name = Path(upload.name).stem

    if image is None:
        st.info("Pick a sample image in the sidebar, or upload your own, to run detection.")
        st.stop()

    with st.spinner("Running YOLO..."):
        result = detect_image(model, image, conf=conf, iou=iou)

    cols = st.columns(4)
    cols[0].metric("Image size", f"{image.shape[1]} x {image.shape[0]}")
    cols[1].metric("Objects detected", len(result.detections))
    cols[2].metric("Unique classes", len(result.class_counts))
    cols[3].metric("Inference time", f"{result.elapsed_ms:.0f} ms")

    tab_result, tab_table, tab_gallery = st.tabs(["Detection Result", "Detections Table", "Try All Samples"])

    with tab_result:
        col_a, col_b = st.columns(2)
        col_a.image(bgr_to_rgb(image), use_container_width=True, caption="Original")
        col_b.image(bgr_to_rgb(result.annotated), use_container_width=True,
                    caption=f"Detected ({model_label.split(' - ')[0]}, conf >= {conf:.2f})")
        st.download_button("Download annotated image (PNG)", data=encode_png(result.annotated),
                           file_name=f"{image_name}_detected.png", mime="image/png")

    with tab_table:
        st.caption("Every kept detection: class, confidence score, and pixel bounding box.")
        st.dataframe(detections_table(result.detections), use_container_width=True, hide_index=True)
        if result.class_counts:
            st.bar_chart(result.class_counts)
        else:
            st.warning("Nothing detected at this confidence threshold - try lowering it in the sidebar.")

    with tab_gallery:
        st.caption("Runs the same model over every bundled sample image, so you can see class/colour "
                  "consistency across many photos at once (this is what the coding-practice script does headlessly).")
        if st.button("Run detection on all sample images"):
            grid_cols = st.columns(3)
            for i, (name, path) in enumerate(samples.items()):
                img = resize_max_side(load_image(path), 500)
                r = detect_image(model, img, conf=conf, iou=iou)
                caption = f"{name}: " + (", ".join(f"{k}({v})" for k, v in r.class_counts.items()) or "no detections")
                grid_cols[i % 3].image(bgr_to_rgb(r.annotated), use_container_width=True, caption=caption)

# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------

else:
    samples = sample_video_paths()
    video_path = None
    video_name = "video"

    with st.sidebar:
        if source == "Sample":
            if not samples:
                st.error("No sample videos found. Run `python download_samples.py` first.")
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
        st.info("Pick a sample video in the sidebar, or upload your own, to run detection.")
        st.stop()

    st.caption(f"Processing is capped at {MAX_VIDEO_FRAMES} frames on this hosted app "
              "so one upload can't stall it for other users - run coding_practice/03_detect_videos.py "
              "locally for full-length, uncapped output.")

    if st.button("Run detection on this video", type="primary"):
        progress = st.progress(0.0, text="Starting...")

        def on_progress(done: int, total: int) -> None:
            frac = min(done / max(total, 1), 1.0)
            progress.progress(frac, text=f"Processing frame {done}/{total}")

        out_path = Path(tempfile.gettempdir()) / f"{video_name}_detected.mp4"
        with st.spinner("Running YOLO on every frame..."):
            result = detect_video(model, video_path, out_path, conf=conf, iou=iou,
                                  max_frames=MAX_VIDEO_FRAMES, progress_cb=on_progress)
        progress.empty()

        cols = st.columns(4)
        cols[0].metric("Frames processed", result.n_frames)
        cols[1].metric("Frames with a detection", result.frames_with_detection)
        cols[2].metric("Unique classes", len(result.class_counts))
        cols[3].metric("Processing time", f"{result.elapsed_s:.1f} s")

        st.video(str(out_path))
        st.download_button("Download annotated video (MP4)", data=out_path.read_bytes(),
                           file_name=f"{video_name}_detected.mp4", mime="video/mp4")

        st.caption("Class counts across all processed frames (a fast-moving object is counted once per "
                  "frame it appears in, not once per unique object):")
        if result.class_counts:
            st.bar_chart(result.class_counts)
        else:
            st.warning("Nothing detected at this confidence threshold - try lowering it in the sidebar.")

st.divider()
st.caption("Day 27 - YOLO object detection (Ultralytics). Bounding-box colour is deterministic per "
          "class (same class = same colour every run). The banner burned into each result shows the "
          "confidence threshold used, object count, and inference time.")
