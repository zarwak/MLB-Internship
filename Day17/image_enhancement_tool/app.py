"""
Document Image Enhancement Tool - Streamlit app.
Upload a document photo, choose which enhancement steps to apply, compare the
result against the original, and download the enhanced image.
"""
import cv2
import numpy as np
import streamlit as st

from enhancer import run_pipeline

st.set_page_config(
    page_title="Document Image Enhancement Tool",
    layout="wide",
)

st.title("Document Image Enhancement Tool")
st.write(
    "Upload a photo of a document to straighten it, remove noise, correct "
    "brightness and contrast, and sharpen the text. Use the controls on the "
    "left to choose which steps to apply."
)

# ---------- Sidebar: processing controls ----------
st.sidebar.header("Processing Options")

correct_tilt = st.sidebar.checkbox(
    "Correct perspective (straighten a tilted page)", value=True
)
convert_gray = st.sidebar.checkbox("Convert to grayscale", value=True)
denoise = st.sidebar.checkbox("Reduce noise", value=True)

fix_contrast = st.sidebar.checkbox("Enhance brightness and contrast", value=True)
clip_limit = st.sidebar.slider(
    "Contrast strength", min_value=1.0, max_value=5.0, value=2.0, step=0.5,
    disabled=not fix_contrast,
)

brightness_beta = st.sidebar.slider(
    "Extra brightness adjustment", min_value=-100, max_value=100, value=0, step=5
)

sharpen_image = st.sidebar.checkbox("Sharpen text edges", value=True)

st.sidebar.markdown("---")
show_steps = st.sidebar.checkbox("Show each processing step", value=False)


# ---------- Helper functions ----------
def load_image(uploaded_file):
    """Reads an uploaded file into an OpenCV (BGR) image."""
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def to_display(img):
    """OpenCV stores color as BGR, Streamlit expects RGB for display."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_png(img):
    """Encodes an image to PNG bytes for the download button."""
    success, buffer = cv2.imencode(".png", img)
    return buffer.tobytes() if success else None


# ---------- Main area ----------
uploaded_file = st.file_uploader(
    "Upload a document image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
else:
    original = load_image(uploaded_file)

    if original is None:
        st.error("This file could not be read. Please upload a valid image.")
    else:
        with st.spinner("Processing image..."):
            steps = run_pipeline(
                original,
                correct_tilt=correct_tilt,
                convert_gray=convert_gray,
                denoise=denoise,
                fix_contrast=fix_contrast,
                clip_limit=clip_limit,
                brightness_beta=brightness_beta,
                sharpen_image=sharpen_image,
            )

        final_image = steps["Final"]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(to_display(original), use_container_width=True)
        with col2:
            st.subheader("Enhanced")
            st.image(to_display(final_image), use_container_width=True)

        download_bytes = encode_png(final_image)
        if download_bytes is not None:
            st.download_button(
                label="Download enhanced image",
                data=download_bytes,
                file_name="enhanced_document.png",
                mime="image/png",
            )

        if show_steps:
            st.markdown("---")
            st.subheader("Processing Steps")
            step_names = [n for n in steps.keys() if n not in ("Original", "Final")]

            if not step_names:
                st.write("No processing steps were applied. Enable options in the sidebar.")
            else:
                cols = st.columns(len(step_names))
                for col, name in zip(cols, step_names):
                    with col:
                        st.caption(name)
                        st.image(to_display(steps[name]), use_container_width=True)

st.markdown("---")
st.caption(
    "Pipeline: perspective correction, grayscale conversion, noise reduction "
    "(bilateral filter), contrast enhancement (CLAHE), brightness adjustment, "
    "and sharpening. Built with OpenCV and Streamlit."
)
