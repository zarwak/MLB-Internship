import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from monitoring import process_video, read_first_frame
from segmentation import segment_image

st.set_page_config(page_title="Day 38 AI Security & Segmentation", page_icon="🛡️", layout="wide")

st.title("🛡️ Day 38 — Intelligent Security Monitoring + Image Segmentation")
st.caption("Two complete tasks: YOLO person tracking/ROI events and classical image segmentation.")

tab1, tab2 = st.tabs(["🎥 Coding Practice — Security Monitoring", "🖼️ Deployment — Segmentation"])

with tab1:
    st.header("Intelligent Security Monitoring System")
    st.write("Upload a video, define one or more polygon ROIs, then track people and log entry/exit events.")

    col1, col2 = st.columns([1, 1])
    with col1:
        video = st.file_uploader("Upload security video", type=["mp4", "avi", "mov", "mkv"], key="video")
        conf = st.slider("YOLO confidence", 0.10, 0.90, 0.35, 0.05)
        process_every = st.slider("Process every Nth frame", 1, 5, 1)
    with col2:
        st.info("ROI format: JSON list of objects. Coordinates are pixels in the original video frame.")
        roi_default = '[{"name":"Main Entrance","points":[[80,80],[1180,80],[1180,620],[80,620]]}]'
        roi_text = st.text_area("ROI configuration", value=roi_default, height=150)
        min_stable = st.slider("Stable frames before event", 1, 10, 3)

    if video:
        tmp_dir = Path(tempfile.mkdtemp(prefix="day38_"))
        input_path = tmp_dir / video.name
        input_path.write_bytes(video.getbuffer())

        first = read_first_frame(str(input_path))
        if first is not None:
            h, w = first.shape[:2]
            st.image(cv2.cvtColor(first, cv2.COLOR_BGR2RGB), caption=f"First frame — width={w}, height={h}", use_container_width=True)
            st.caption("Use these width/height values when defining ROI points. A polygon is [x,y] points in clockwise/counter-clockwise order.")

        if st.button("▶️ Run Security Monitoring", type="primary", use_container_width=True):
            try:
                rois = json.loads(roi_text)
                if not isinstance(rois, list) or not rois:
                    raise ValueError("ROI configuration must be a non-empty JSON list.")
                for roi in rois:
                    if "name" not in roi or "points" not in roi or len(roi["points"]) < 3:
                        raise ValueError("Each ROI needs a name and at least 3 points.")
                with st.spinner("Running YOLO person tracking and event analytics..."):
                    result = process_video(
                        str(input_path),
                        rois,
                        conf=conf,
                        process_every=process_every,
                        min_stable_frames=min_stable,
                    )
                st.session_state["monitor_result"] = result
            except Exception as e:
                st.error(f"Could not process the video: {e}")

    if "monitor_result" in st.session_state:
        result = st.session_state["monitor_result"]
        st.success("Processing completed.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unique people", result["unique_people"])
        m2.metric("Entries", result["entries"])
        m3.metric("Exits", result["exits"])
        m4.metric("Max active", result["max_active"])

        st.subheader("Processed video")
        st.video(result["video_path"])

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download processed video",
                data=Path(result["video_path"]).read_bytes(),
                file_name="security_monitor_processed.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "⬇️ Download event log CSV",
                data=Path(result["csv_path"]).read_bytes(),
                file_name="security_events.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.subheader("Event log")
        st.dataframe(result["events"], use_container_width=True, hide_index=True)

with tab2:
    st.header("Image Segmentation")
    st.write("Upload an image, choose Binary, Adaptive, or Otsu thresholding, preview the result, and download it.")

    img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="image")
    method = st.selectbox("Segmentation method", ["Binary", "Adaptive", "Otsu"])
    threshold = st.slider("Binary threshold", 0, 255, 127)
    block_size = st.slider("Adaptive block size", 3, 51, 11, step=2)
    c_value = st.slider("Adaptive C", -20, 20, 2)

    if img_file:
        raw = np.frombuffer(img_file.getvalue(), np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Invalid image.")
        else:
            output = segment_image(
                image,
                method,
                threshold=threshold,
                block_size=block_size,
                c_value=c_value,
            )
            left, right = st.columns(2)
            with left:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
            with right:
                st.image(output, caption=f"{method} segmentation", use_container_width=True)

            ok, encoded = cv2.imencode(".png", output)
            if ok:
                st.download_button(
                    "⬇️ Download segmented image",
                    data=encoded.tobytes(),
                    file_name=f"segmented_{method.lower()}.png",
                    mime="image/png",
                    use_container_width=True,
                )
