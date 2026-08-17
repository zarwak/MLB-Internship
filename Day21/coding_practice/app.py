"""
Day 21 coding practice: a Streamlit interface over the 7 image operations
in image_ops.py. Deliberately minimal - upload, pick an operation, process,
display, download. No extra parameters/sliders here on purpose; that
polish lives in the mini project (../cv_image_studio).
"""

import io
import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from image_ops import OPERATIONS

st.set_page_config(page_title="CV Coding Practice", page_icon=":camera:")
st.title("Computer Vision - Coding Practice")
st.write("Upload an image, pick an operation, and see the result.")

# Anchored to this file's own folder, not the terminal's current directory -
# otherwise "streamlit run" launched from a different working directory
# would save files there instead of here.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

operation_name = st.selectbox("Choose an operation", list(OPERATIONS.keys()))

if uploaded_file is not None:
    # Streamlit gives us raw bytes; PIL decodes them, then we flip RGB->BGR
    # because every function in image_ops.py expects OpenCV's channel order.
    pil_image = Image.open(uploaded_file).convert("RGB")
    original_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    if st.button("Process Image"):
        operation_fn = OPERATIONS[operation_name]
        result = operation_fn(original_bgr)

        # Result may be single-channel (grayscale/edges) or 3-channel (BGR).
        # st.image and PIL both want RGB, so convert only when needed.
        if result.ndim == 2:
            display_result = result
            pil_result = Image.fromarray(result)
        else:
            display_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            pil_result = Image.fromarray(display_result)

        # Stash everything the display/download/save section needs in
        # session_state. st.button() only returns True on the exact rerun
        # its own click triggered - clicking Download or Save below causes
        # a NEW rerun where this "Process Image" button is False again, so
        # anything needed afterward has to survive outside this if-block.
        st.session_state["result"] = {
            "pil_image": pil_image,
            "operation_name": operation_name,
            "display_result": display_result,
            "pil_result": pil_result,
        }

    if "result" in st.session_state:
        r = st.session_state["result"]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(r["pil_image"], use_container_width=True)
        with col2:
            st.subheader(r["operation_name"])
            st.image(r["display_result"], use_container_width=True)

        # Download button needs bytes, not a PIL Image - encode to PNG in memory.
        buffer = io.BytesIO()
        r["pil_result"].save(buffer, format="PNG")

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            st.download_button(
                label="Download processed image",
                data=buffer.getvalue(),
                file_name=f"{r['operation_name'].lower().replace(' ', '_')}_result.png",
                mime="image/png",
                use_container_width=True,
            )
        with button_col2:
            if st.button("Save processed image to disk", use_container_width=True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_op = r["operation_name"].lower().replace(" ", "_")
                out_path = os.path.join(OUTPUT_DIR, f"{safe_op}_{timestamp}.png")
                r["pil_result"].save(out_path)
                st.success(f"Saved to {out_path}")
else:
    st.info("Upload an image to get started. Try the sample_images/ folder if you don't have one handy.")
