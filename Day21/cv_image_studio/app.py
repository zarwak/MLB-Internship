"""
Challenge Task: everything from the mini project, plus 3 operations not
covered in class (Brightness & Contrast, Flip, Threshold), plus a Pipeline
mode that chains multiple operations together in sequence - the "make it
more useful" ask from the assignment.
"""

import io
import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from image_ops_extended import OPERATIONS, PARAM_SPECS

st.set_page_config(page_title="CV Studio - Challenge", page_icon=":sparkles:", layout="wide")

# Anchored to this file's own folder, not the terminal's current directory -
# otherwise "streamlit run" launched from a different working directory
# (e.g. still sitting in a sibling project folder) would save/read files
# there instead of here.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_input_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CUSTOM_OPS = {"Brightness & Contrast", "Flip", "Threshold"}


def load_bgr(image_bytes: bytes) -> np.ndarray:
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def to_displayable(result: np.ndarray) -> np.ndarray:
    if result.ndim == 2:
        return result
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def build_controls(operation_name: str, key_prefix: str = "") -> dict:
    """Reads PARAM_SPECS for one operation and renders the right widget for
    each parameter - a slider for numeric ranges, a selectbox for named
    choices - keyed uniquely so the same operation can appear twice on a
    page (e.g. once per pipeline step) without Streamlit key collisions."""
    params = {}
    for param_name, spec in PARAM_SPECS[operation_name].items():
        widget_key = f"{key_prefix}_{operation_name}_{param_name}"
        if spec.get("type") == "choice":
            params[param_name] = st.selectbox(
                spec["label"], spec["options"],
                index=spec["options"].index(spec["default"]),
                key=widget_key,
            )
        else:
            is_float = isinstance(spec["default"], float)
            params[param_name] = st.slider(
                spec["label"],
                min_value=float(spec["min"]) if is_float else int(spec["min"]),
                max_value=float(spec["max"]) if is_float else int(spec["max"]),
                value=float(spec["default"]) if is_float else int(spec["default"]),
                step=float(spec["step"]) if is_float else int(spec["step"]),
                key=widget_key,
            )
    return params


def download_button_for(pil_image: Image.Image, label: str, filename: str, key: str) -> None:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    st.download_button(label, data=buffer.getvalue(), file_name=filename, mime="image/png", key=key)


st.title("CV Studio - Challenge Task")
st.caption(
    "Beyond the 7 core operations: Brightness & Contrast, Flip, and Threshold "
    "are new. Pipeline mode lets you chain several operations in sequence, "
    "which is the part that actually makes this more useful than a single "
    "dropdown - real editing is rarely just one filter."
)

with st.sidebar:
    st.header("Image source")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    sample_names = sorted(os.listdir(SAMPLE_DIR)) if os.path.isdir(SAMPLE_DIR) else []
    chosen_sample = st.selectbox("Or pick a sample", ["(none)"] + sample_names) if uploaded_file is None else "(none)"

    st.divider()
    mode = st.radio("Mode", ["Single Operation", "Chain Multiple Filters (Pipeline)"])

image_bytes = None
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
elif chosen_sample != "(none)":
    with open(os.path.join(SAMPLE_DIR, chosen_sample), "rb") as f:
        image_bytes = f.read()

if image_bytes is None:
    st.info("Upload an image or pick a sample from the sidebar to get started.")
    st.stop()

original_bgr = load_bgr(image_bytes)
pil_original = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))

if mode == "Single Operation":
    with st.sidebar:
        st.divider()
        operation_name = st.selectbox("Operation", list(OPERATIONS.keys()))
        if operation_name in CUSTOM_OPS:
            st.caption("Custom feature - not covered in class")
        params = build_controls(operation_name, key_prefix="single")

    result = OPERATIONS[operation_name](original_bgr, **params)
    display_result = to_displayable(result)
    pil_result = Image.fromarray(display_result)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil_original, use_container_width=True)
    with col2:
        st.subheader(f"Processed - {operation_name}")
        st.image(display_result, use_container_width=True)

    button_col1, button_col2 = st.columns(2)
    with button_col1:
        download_button_for(pil_result, "Download processed image",
                             f"{operation_name.lower().replace(' ', '_')}_result.png", key="single_download")
    with button_col2:
        if st.button("Save processed image to disk", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_op = operation_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and")
            out_path = os.path.join(OUTPUT_DIR, f"{safe_op}_{timestamp}.png")
            pil_result.save(out_path)
            st.success(f"Saved to {out_path}")

else:
    with st.sidebar:
        st.divider()
        pipeline_ops = st.multiselect(
            "Build your pipeline (applied in the order you pick them)",
            list(OPERATIONS.keys()),
            default=["Brightness & Contrast", "Blur", "Edge Detection"],
        )

    st.subheader("Original")
    st.image(pil_original, width=300)

    if not pipeline_ops:
        st.info("Pick at least one operation in the sidebar to build a pipeline.")
        st.stop()

    st.subheader("Pipeline steps")
    current = original_bgr
    step_cols = st.columns(len(pipeline_ops))
    for i, op_name in enumerate(pipeline_ops):
        with st.sidebar.expander(f"Step {i + 1}: {op_name} settings"):
            params = build_controls(op_name, key_prefix=f"step{i}")
        current = OPERATIONS[op_name](current, **params)
        # Every function can hand back grayscale (2D); the next step in the
        # chain may expect 3 channels (e.g. Brightness & Contrast, Flip),
        # so convert back to BGR between steps whenever that happens.
        if current.ndim == 2:
            current = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
        with step_cols[i]:
            st.caption(f"Step {i + 1}: {op_name}")
            st.image(to_displayable(current), use_container_width=True)

    st.subheader("Final result")
    final_display = to_displayable(current)
    pil_final = Image.fromarray(final_display)
    st.image(final_display, width=400)

    button_col1, button_col2 = st.columns(2)
    with button_col1:
        download_button_for(pil_final, "Download final result", "pipeline_result.png", key="pipeline_download")
    with button_col2:
        if st.button("Save final result to disk", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(OUTPUT_DIR, f"pipeline_{timestamp}.png")
            pil_final.save(out_path)
            st.success(f"Saved to {out_path}")
