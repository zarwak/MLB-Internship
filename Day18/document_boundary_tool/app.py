"""
Document Boundary Detection Tool - Streamlit app.
Upload a document photo, the pipeline runs with fixed sensible defaults, and
you see EVERY step it went through - grayscale, blur, Canny edges,
morphological cleanup, and the final boundary - plus a download button.
"""
import cv2
import numpy as np
import streamlit as st

from detector import detect_boundary

st.set_page_config(page_title="Document Boundary Detection Tool", layout="wide")

st.title("Document Boundary Detection Tool")
st.write(
    "Upload a photo of a document and this finds its edges, cleans them up, "
    "and draws a box around the page it detected."
)


def load_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def to_display(img):
    if img.ndim == 2:  # grayscale/binary steps have no color channels to convert
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


uploaded_file = st.file_uploader("Choose a document photo", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    original = load_image(uploaded_file)

    if original is None:
        st.error("Couldn't read that file - please upload a valid jpg, jpeg, png, or bmp image.")
    else:
        with st.spinner("Detecting boundary..."):
            result = detect_boundary(original)

        if result["found_four_corners"]:
            st.success("Found a clean 4-corner page boundary.")
        elif result["contour_points"] is not None:
            st.warning(
                "Couldn't find a clean 4-corner page outline, so this is an "
                "approximate boundary (orange) based on the largest shape detected."
            )
        else:
            st.error("Couldn't detect any document boundary in this photo.")

        st.subheader("Every step of the pipeline")
        steps = [
            ("1. Original", original),
            ("2. Grayscale", result["gray"]),
            ("3. Gaussian Blur", result["blurred"]),
            ("4. Canny Edges", result["edges"]),
            ("5. Morphological Cleanup", result["cleaned_edges"]),
            ("6. Boundary Detected", result["annotated"]),
        ]
        row1, row2 = st.columns(3), st.columns(3)
        for col, (label, step_img) in zip(row1 + row2, steps):
            with col:
                st.caption(label)
                st.image(to_display(step_img), use_container_width=True)

        success, encoded = cv2.imencode(".jpg", result["annotated"])
        if success:
            st.download_button(
                "Download final boundary image",
                data=encoded.tobytes(),
                file_name=f"boundary_{uploaded_file.name}",
                mime="image/jpeg",
            )
