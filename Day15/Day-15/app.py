"""
Streamlit App for Object Detection using YOLOv11
=================================================
A simple web app where users can upload an image and see
object detection results with bounding boxes, class labels,
and confidence scores.

Usage:
  streamlit run app.py

Then open the link shown in the terminal in your browser.
"""

import io
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st

# Load the pre-trained YOLOv11 model
MODEL_NAME = "yolo11n.pt"
st.sidebar.title("YOLOv11 Object Detection")
st.sidebar.write(f"**Model:** {MODEL_NAME}")
st.sidebar.write("Pre-trained on COCO dataset (80 classes)")

# Load model (cached for performance)
@st.cache_resource
def load_model():
    return YOLO(MODEL_NAME)

with st.spinner("Loading YOLOv11n model..."):
    model = load_model()

st.sidebar.success("Model loaded!")

# Main app
st.title("🍎 Object Detection using YOLOv11")
st.markdown(
    "Upload an image to detect objects. The model will draw bounding boxes, "
    "show class labels, and confidence scores."
)

# Image upload
uploaded_file = st.file_uploader(
    "Choose an image file", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load the uploaded image using PIL
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Display the original image
    st.subheader("Original Image")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run YOLO inference
    with st.spinner("Running object detection..."):
        results = model(img_array, conf=0.25)
        result = results[0]

    # Get the annotated image (with bounding boxes drawn)
    # result.plot() returns a numpy array (BGR format from OpenCV)
    annotated_img = result.plot()

    # Convert BGR to RGB using numpy slicing (avoids cv2 dependency in app code)
    annotated_img_rgb = annotated_img[:, :, ::-1]

    # Display the detection results
    st.subheader("Detection Results")
    st.image(annotated_img_rgb, caption="Detected Objects", use_container_width=True)

    # Display detection details
    st.subheader("Detection Details")
    if len(result.boxes) == 0:
        st.info("No objects detected.")
    else:
        st.success(f"Detected {len(result.boxes)} object(s):")
        detection_data = []
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            detection_data.append({
                "Object": class_name,
                "Confidence": f"{confidence:.2f}",
                "BBox (x1, y1, x2, y2)": f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]",
            })
        st.table(detection_data)

    # Download button for the result image
    result_pil = Image.fromarray(annotated_img_rgb)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label="📥 Download Detection Result",
        data=buf,
        file_name="detection_result.png",
        mime="image/png",
    )

else:
    st.info("👈 Upload an image using the uploader above to get started!")
    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("1. Upload an image using the uploader above.")
    st.markdown("2. The model runs YOLOv11 inference automatically.")
    st.markdown("3. View the results with bounding boxes and details.")
    st.markdown("4. Download the detection result image.")
    st.markdown("")
    st.markdown("**Model:** YOLOv11n (pre-trained on COCO dataset, 80 classes)")
    st.markdown("**Classes include:** person, car, dog, cat, apple, banana, orange, tomato, grape, book, chair, and many more.")
