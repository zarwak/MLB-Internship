"""
Real-Time Video Processing Tool - Streamlit app.
Upload a video and this runs the same grayscale -> blur -> Canny pipeline
used by process_video.py, showing a live-updating preview while it works
and a download button for the processed result when it's done.

A live cv2.imshow() window (like the local scripts use) only works when a
script runs on YOUR OWN machine with a display - it can't run on a cloud
server. So instead of a real video window, this app updates one image in
place, every so often, while it processes - and for the webcam, it uses
Streamlit's built-in camera widget (one snapshot at a time) instead of a
continuous live feed, since a browser can't hand a website a raw persistent
camera stream the way cv2.VideoCapture(0) can on your own machine.
"""
import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from processor import process_frame, print_video_properties, open_writer

st.set_page_config(page_title="Real-Time Video Processing Tool", layout="wide")

st.title("Real-Time Video Processing Tool")
st.write(
    "Upload a video and watch it get converted to grayscale, blurred, and "
    "run through Canny edge detection - frame by frame - with a download "
    "button for the processed result."
)

st.sidebar.header("Tune the Pipeline")
blur_k = st.sidebar.slider(
    "**Smoothing strength** (blur kernel, odd values only)",
    min_value=1, max_value=31, value=5, step=2,
    help="Turns this up to wipe out more grainy noise before edges get "
         "detected - but crank it too far and real edges blur away too. "
         "Has to land on an odd number since the kernel is centered on one pixel.",
)
canny_low = st.sidebar.slider(
    "**Edge detector - faint cutoff**", min_value=0, max_value=300, value=50,
    help="Anything fainter than this brightness jump never counts as an edge.",
)
canny_high = st.sidebar.slider(
    "**Edge detector - strong cutoff**", min_value=0, max_value=300, value=150,
    help="Anything past this brightness jump always counts as an edge. "
         "Values sitting between the two cutoffs only survive if they link up "
         "with a strong edge - that's what keeps the result clean, not speckled.",
)
resize_width = st.sidebar.slider(
    "**Working resolution** (frame width in pixels)",
    min_value=160, max_value=1280, value=640, step=80,
    help="Every frame gets shrunk to this width before any processing runs. "
         "Lower it for a speed boost; raise it to keep finer detail.",
)

tab_video, tab_webcam = st.tabs(["Process a video", "Webcam snapshot"])

with tab_video:
    uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # OpenCV's VideoCapture needs an actual file path, not the in-memory
        # bytes Streamlit hands us - so the upload is written to a temp file first
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(uploaded_file.read())
            in_path = tmp_in.name

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            st.error("Couldn't open that video - please upload a valid mp4, avi, mov, or mkv file.")
        else:
            props = print_video_properties(cap, label=uploaded_file.name)
            st.caption(
                f"FPS: {props['fps']:.1f} | "
                f"{props['width']}x{props['height']} | "
                f"{props['frame_count']} frames"
            )

            # the writer's declared size must exactly match the frames it
            # receives - so if the resize slider shrinks frames below the
            # source's width, the writer needs to be opened at THAT smaller
            # size, computed the same way processor.resize_for_processing does
            if resize_width < props["width"]:
                out_w = resize_width
                out_h = int(props["height"] * (resize_width / props["width"]))
            else:
                out_w, out_h = props["width"], props["height"]

            out_path = os.path.join(tempfile.gettempdir(), f"processed_{uploaded_file.name}.mp4")
            writer = open_writer(out_path, props["fps"], out_w, out_h)

            col1, col2 = st.columns(2)
            col1.caption("Original")
            col2.caption("Processed (grayscale -> blur -> Canny)")
            original_slot = col1.empty()
            processed_slot = col2.empty()
            progress_bar = st.progress(0)

            total = max(1, props["frame_count"])
            # redrawing the preview on EVERY frame makes the app crawl - each
            # update is a full round-trip to the browser, unlike a local
            # cv2.imshow window which just redraws in place - so only refresh
            # roughly 40 times over the whole video
            preview_every = max(1, total // 40)

            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1

                result = process_frame(
                    frame,
                    blur_ksize=(blur_k, blur_k),
                    canny_low=canny_low,
                    canny_high=canny_high,
                    resize_width=resize_width,
                )
                writer.write(result["edges_bgr"])

                if frame_num % preview_every == 0 or frame_num == total:
                    original_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    processed_slot.image(result["edges_bgr"], channels="BGR", use_container_width=True)
                    progress_bar.progress(min(1.0, frame_num / total))

            cap.release()
            writer.release()
            progress_bar.progress(1.0)
            st.success(f"Processed {frame_num} frame(s).")

            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button(
                    "Download processed video",
                    data=f.read(),
                    file_name=f"processed_{uploaded_file.name}",
                    mime="video/mp4",
                )

with tab_webcam:
    st.write(
        "Browsers can't hand a website a continuous live camera stream the "
        "way a local script can - so this takes one snapshot at a time "
        "through your camera and runs it through the same pipeline."
    )
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        file_bytes = np.frombuffer(camera_image.read(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = process_frame(
            frame,
            blur_ksize=(blur_k, blur_k),
            canny_low=canny_low,
            canny_high=canny_high,
            resize_width=resize_width,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Original")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.caption("Processed (grayscale -> blur -> Canny)")
            st.image(result["edges_bgr"], channels="BGR", use_container_width=True)
