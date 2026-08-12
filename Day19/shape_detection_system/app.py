"""
Shape Detection System - Streamlit app.
Upload a photo, the pipeline runs with fixed sensible defaults, and you see
every step it went through - grayscale, blur, threshold, all contours, and
the final labeled shapes - plus a table of measurements and a download button.
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detector import detect_shapes

st.set_page_config(page_title="Shape Detection System", layout="wide")

st.title("Shape Detection System")
st.write(
    "Upload a photo containing shapes and this finds each one, labels it "
    "(circle, square, rectangle, triangle, or polygon), and reports its "
    "area and perimeter."
)


def load_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def to_display(img):
    if img.ndim == 2:  # grayscale/binary steps have no color channels to convert
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    original = load_image(uploaded_file)

    if original is None:
        st.error("Couldn't read that file - please upload a valid jpg, jpeg, png, or bmp image.")
    else:
        with st.spinner("Detecting shapes..."):
            result = detect_shapes(original)

        shapes = result["shapes"]
        if shapes:
            st.success(f"Found {len(shapes)} shape(s).")
        else:
            st.warning("No shapes detected - try an image with clearer contrast between shapes and background.")

        st.subheader("Every step of the pipeline")
        steps = [
            ("1. Original", original),
            ("2. Grayscale", result["gray"]),
            ("3. Gaussian Blur", result["blurred"]),
            ("4. Threshold", result["thresh"]),
            ("5. All Contours", result["contours_drawn"]),
            ("6. Labeled Shapes", result["annotated"]),
        ]
        row1, row2 = st.columns(3), st.columns(3)
        for col, (label, step_img) in zip(row1 + row2, steps):
            with col:
                st.caption(label)
                st.image(to_display(step_img), use_container_width=True)

        if shapes:
            st.subheader("Measurements")
            table = pd.DataFrame([
                {
                    "Shape": s["label"],
                    "Area (px)": round(s["area"], 1),
                    "Perimeter (px)": round(s["perimeter"], 1),
                    "Bounding Box (x, y, w, h)": s["bbox"],
                }
                for s in shapes
            ])
            st.dataframe(table, use_container_width=True)

        success, encoded = cv2.imencode(".jpg", result["annotated"])
        if success:
            st.download_button(
                "Download labeled shapes image",
                data=encoded.tobytes(),
                file_name=f"shapes_{uploaded_file.name}",
                mime="image/jpeg",
            )
