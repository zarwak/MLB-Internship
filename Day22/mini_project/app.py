"""
Day 22 Mini Project - Simple OCR Document Reader.

Upload an image -> EasyOCR extracts the text -> see original image and
extracted text side by side -> download the result as .txt.

WHY st.cache_resource for the EasyOCR reader specifically (not
st.cache_data): Streamlit reruns this whole script top-to-bottom on every
interaction (see Day21's README for more on that rerun model). Without
caching, moving so much as a slider would reload two neural networks from
disk every single time - several seconds of dead time per click.
cache_resource is Streamlit's cache for things that should be built once
and *shared* across reruns/users (models, DB connections) as opposed to
cache_data, which is for cacheable *data* that gets copied per call.
"""

import io
import os
from datetime import datetime

import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="DoxProX OCR Reader", page_icon=":page_facing_up:", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "extracted_text_saved")
os.makedirs(SAVED_DIR, exist_ok=True)


@st.cache_resource(show_spinner="Loading EasyOCR model (first run only)...")
def get_reader() -> easyocr.Reader:
    return easyocr.Reader(["en"], gpu=False)


def load_image(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def preprocess(pil_image: Image.Image) -> np.ndarray:
    """Grayscale + CLAHE contrast + Otsu threshold - the same pipeline
    that helped most on the noisy/low-light sample images in
    ocr_practice/run_easyocr_batch.py. Optional here because it can
    actually hurt already-clean, high-contrast images (over-thresholding
    can erase thin strokes) - that trade-off is the whole reason this is
    a checkbox and not something applied unconditionally."""
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _t, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def run_ocr(reader: easyocr.Reader, image_for_ocr) -> tuple[str, float, list]:
    results = reader.readtext(image_for_ocr)
    if not results:
        return "", 0.0, []
    lines = [text for (_box, text, _conf) in results]
    confidences = [conf for (_box, _text, conf) in results]
    return "\n".join(lines), sum(confidences) / len(confidences), results


st.title("DoxProX OCR Reader")
st.caption(
    "Upload a document, receipt, signboard, book page, or handwritten note. "
    "EasyOCR extracts the text; grayscale + contrast preprocessing is "
    "optional and helps most on noisy or low-light images."
)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
apply_preprocessing = st.checkbox(
    "Apply preprocessing (grayscale + CLAHE contrast + Otsu threshold)",
    value=False,
)

if uploaded_file is not None:
    pil_image = load_image(uploaded_file)
    ocr_input = preprocess(pil_image) if apply_preprocessing else np.array(pil_image)

    with st.spinner("Extracting text..."):
        reader = get_reader()
        text, avg_confidence, results = run_ocr(reader, ocr_input)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(pil_image, use_container_width=True)
        if apply_preprocessing:
            st.caption("Preprocessed version sent to OCR:")
            st.image(ocr_input, use_container_width=True)

    with col2:
        st.subheader("Extracted Text")
        if text:
            st.metric("Average confidence", f"{avg_confidence:.0%}")
            st.text_area("Text", text, height=350)
        else:
            st.warning("No text detected. Try toggling preprocessing.")

    if text:
        st.download_button(
            "Download as .txt",
            data=text.encode("utf-8"),
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_extracted.txt",
            mime="text/plain",
        )

        if st.button("Save to extracted_text_saved/ on disk"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"{os.path.splitext(uploaded_file.name)[0]}_{stamp}.txt"
            out_path = os.path.join(SAVED_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            st.success(f"Saved to {out_path}")

    with st.expander("Show detected text regions (boxes + per-line confidence)"):
        for box, line_text, conf in results:
            st.write(f"`{conf:.2f}`  {line_text}")
else:
    st.info("Upload an image to get started. Sample images are in ../images/.")
