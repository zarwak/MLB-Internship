"""
Gradio App for Object Detection using YOLOv8
=============================================
A simple web app where users can upload an image and see
object detection results with bounding boxes, class labels,
and confidence scores.

Usage:
  python app.py

Then open the link shown in the terminal in your browser.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr

# Load the pre-trained YOLOv8 model
MODEL_NAME = "yolov8n.pt"
print(f"Loading model: {MODEL_NAME} ...")
model = YOLO(MODEL_NAME)
print("Model loaded! Starting Gradio app...")


def detect_objects(image):
    """
    Run YOLOv8 object detection on the uploaded image.

    Args:
        image: A numpy array (H, W, 3) from Gradio's image input.

    Returns:
        tuple: (annotated_image, detection_text)
    """
    # Convert RGB (Gradio) to BGR (OpenCV) if needed
    if image is not None:
        # Gradio provides images as numpy arrays in RGB format
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Run YOLO inference
        results = model(img_bgr, conf=0.25)
        result = results[0]

        # Get the annotated image (with bounding boxes drawn)
        annotated_img = result.plot()

        # Convert back to RGB for Gradio display
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Build detection text
        lines = []
        if len(result.boxes) == 0:
            lines.append("No objects detected.")
        else:
            lines.append(f"Detected {len(result.boxes)} object(s):\n")
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                lines.append(
                    f"{i+1}. {class_name} "
                    f"(confidence: {confidence:.2f}) "
                    f"bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                )

        detection_text = "\n".join(lines)
        return annotated_img_rgb, detection_text

    return image, "No image provided."


# Create the Gradio interface
with gr.Blocks(title="YOLOv8 Object Detection") as demo:
    gr.Markdown("# 🍎 Object Detection using YOLOv8")
    gr.Markdown(
        "Upload an image to detect objects. The model will draw bounding boxes, "
        "show class labels, and confidence scores."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Image")
            detect_btn = gr.Button("🔍 Detect Objects", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Detection Results")
            output_text = gr.Textbox(label="Detection Details")

    detect_btn.click(
        fn=detect_objects,
        inputs=input_image,
        outputs=[output_image, output_text],
    )

    gr.Markdown("---")
    gr.Markdown(
        "### How it works:\n"
        "1. Upload an image using the uploader above.\n"
        "2. Click 'Detect Objects' to run YOLOv8 inference.\n"
        "3. View the results with bounding boxes and details.\n"
        "\n"
        "**Model:** YOLOv8n (pre-trained on COCO dataset, 80 classes)\n"
        "**Classes include:** person, car, dog, cat, apple, banana, orange, "
        "tomato, grape, book, chair, and many more."
    )

if __name__ == "__main__":
    demo.launch()
