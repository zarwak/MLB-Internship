"""
Create a screen recording explaining the YOLO Object Detection implementation.
This script generates a video that walks through the project.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configuration
OUTPUT_VIDEO = "screen_recording.mp4"
FPS = 1  # 1 frame per second (simple slideshow)
DURATION_PER_SLIDE = 3  # seconds per slide

# Slide content: (title, text_lines, image_path)
slides = [
    {
        "title": "Day-15: Object Detection using YOLOv8",
        "text": [
            "What is Object Detection?",
            "Object detection identifies WHAT objects are in an image",
            "AND WHERE they are located (bounding boxes).",
            "",
            "Unlike image classification (single label for whole image),",
            "object detection finds MULTIPLE objects with locations.",
        ],
        "image": None,
    },
    {
        "title": "Image Classification vs Object Detection",
        "text": [
            "Image Classification:",
            "  - One label for the entire image",
            "  - 'What is this?'",
            "",
            "Object Detection:",
            "  - Multiple labels + bounding boxes",
            "  - 'What are these and where are they?'",
        ],
        "image": None,
    },
    {
        "title": "What is YOLO?",
        "text": [
            "YOLO = You Only Look Once",
            "",
            "A real-time object detection system that:",
            "  - Uses a SINGLE neural network",
            "  - Predicts bounding boxes + class labels in ONE pass",
            "  - Is fast (45+ FPS) and accurate",
            "",
            "YOLOv8n = nano version (smallest, fastest)",
        ],
        "image": None,
    },
    {
        "title": "Dataset Used",
        "text": [
            "Fruit Disease Detection Dataset",
            "",
            "Images downloaded from public sources:",
            "  - apple.jpg",
            "  - banana.jpg",
            "  - orange.jpg",
            "  - tomato_fruit_Image_1.jpg",
            "  - tomato_fruit_Image_2.jpg",
            "  - grapes_fruit_Image_1.jpg",
            "  - grapes_fruit_Image_2.jpg",
            "  - strawberry_fruit_Image_1.jpg",
            "  - strawberry_fruit_Image_2.jpg",
        ],
        "image": "dataset/images/apple.jpg",
    },
    {
        "title": "Detection Results - Apple",
        "text": [
            "Detected: apple",
            "Confidence: 0.79",
            "Bounding Box: [254, 246, 2074, 2043]",
            "",
            "The model correctly identified the apple!",
        ],
        "image": "output_images/detection_apple.jpg",
    },
    {
        "title": "Detection Results - Banana",
        "text": [
            "Detected: banana",
            "Confidence: 0.84",
            "Bounding Box: [103, 109, 2917, 2537]",
            "",
            "The model correctly identified the banana!",
        ],
        "image": "output_images/detection_banana.jpg",
    },
    {
        "title": "Observations",
        "text": [
            "1. Apple & Banana: Detected correctly (in COCO dataset)",
            "2. Orange: NOT detected (not in COCO classes)",
            "3. Grapes: NOT detected (not in COCO classes)",
            "4. Strawberry: NOT detected (not in COCO classes)",
            "5. Tomato: NOT detected (not in COCO classes)",
            "",
            "Key Insight: Pre-trained YOLOv8n uses COCO (80 classes).",
            "To detect custom objects, train a custom YOLO model!",
        ],
        "image": None,
    },
    {
        "title": "Project Files",
        "text": [
            "Day-15/",
            "  yolo_practice.py     - YOLO practice script",
            "  object_detection.py  - Object detection script",
            "  app.py               - Gradio web app",
            "  requirements.txt   - Python dependencies",
            "  README.md            - Documentation",
            "  sample_images/       - Sample input images",
            "  dataset/images/      - Fruit dataset images",
            "  output_images/       - Detection results",
            "  screen_recording.mp4 - This recording",
        ],
        "image": None,
    },
    {
        "title": "Gradio App",
        "text": [
            "A web app where users can:",
            "  1. Upload an image",
            "  2. Click 'Detect Objects'",
            "  3. See bounding boxes + labels + confidence",
            "",
            "Run with: python app.py",
            "Then open the link in your browser.",
        ],
        "image": None,
    },
    {
        "title": "Thank You!",
        "text": [
            "Questions?",
            "",
            "GitHub: https://github.com/zarwak/MLB-Internship",
            "Day-15: Object Detection using YOLOv8",
        ],
        "image": None,
    },
]


def create_slide(title, text_lines, image_path=None, width=1280, height=720):
    """Create a single slide as a numpy array (BGR for OpenCV)."""
    # Create a white background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Try to use a TrueType font
    try:
        font = ImageFont.truetype("arial.ttf", 36)
        small_font = ImageFont.truetype("arial.ttf", 24)
        title_font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
        small_font = font
        title_font = font

    # Convert to PIL for text rendering
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Draw title
    draw.text((50, 30), title, fill=(0, 0, 0), font=title_font)

    # Draw text lines
    y_offset = 120
    for line in text_lines:
        if line == "":
            y_offset += 30
            continue
        draw.text((50, y_offset), line, fill=(50, 50, 50), font=font)
        y_offset += 40

    # Draw image if provided
    if image_path and os.path.exists(image_path):
        try:
            img_pil = Image.open(image_path)
            # Resize to fit (max 500x400)
            img_pil.thumbnail((500, 400))
            # Paste on the right side
            pil_img.paste(img_pil, (width - img_pil.width - 50, 150))
        except Exception as e:
            draw.text((50, y_offset), f"[Image error: {e}]", fill=(255, 0, 0), font=small_font)

    # Convert back to numpy (BGR for OpenCV)
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_bgr


def main():
    print("Creating screen recording...")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (1280, 720))

    for i, slide in enumerate(slides):
        print(f"  Creating slide {i+1}/{len(slides)}: {slide['title']}")
        frame = create_slide(slide["title"], slide["text"], slide.get("image"))

        # Add each slide for DURATION_PER_SLIDE seconds
        for _ in range(DURATION_PER_SLIDE):
            video.write(frame)

    video.release()
    print(f"\nScreen recording saved to: {OUTPUT_VIDEO}")
    print(f"Total slides: {len(slides)}")
    print(f"Duration: ~{len(slides) * DURATION_PER_SLIDE} seconds")


if __name__ == "__main__":
    main()
