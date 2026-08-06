"""
YOLO Practice Script
====================
This script demonstrates the basic usage of the Ultralytics YOLO package.

Practice 1:
  - Install the Ultralytics YOLO package
  - Load a pre-trained YOLO model
  - Perform object detection on a single image
  - Perform object detection on multiple images
  - Save the prediction results

Practice 2:
  - Test the model on your own images
  - Observe detected objects, confidence scores, and bounding boxes

Requirements:
  pip install ultralytics opencv-python

Usage:
  python yolo_practice.py
"""

import os
import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------
# Step 1: Load a pre-trained YOLO model
# ---------------------------------------------------------------
# YOLOv8n is the smallest and fastest version — great for practice.
# Other options: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
# YOLO11 is also available: yolo11n.pt, yolo11s.pt, etc.
print("=" * 60)
print("Step 1: Loading pre-trained YOLOv8n model...")
print("=" * 60)

model = YOLO("yolov8n.pt")  # 'n' = nano (smallest, fastest)
print(f"Model loaded successfully: {model}")
print(f"Model classes: {len(model.names)} classes")
print(f"Sample classes: {list(model.names.values())[:10]} ...")

# ---------------------------------------------------------------
# Step 2: Perform object detection on a SINGLE image
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Object detection on a single image")
print("=" * 60)

# Create a sample image directory if it doesn't exist
sample_dir = "sample_images"
os.makedirs(sample_dir, exist_ok=True)

# If no sample image exists, download one from the internet
sample_image_path = os.path.join(sample_dir, "sample.jpg")
if not os.path.exists(sample_image_path):
    print("No sample image found. Downloading a sample image...")
    import urllib.request
    url = "https://ultralytics.com/images/bus.jpg"
    urllib.request.urlretrieve(url, sample_image_path)
    print(f"Downloaded sample image to: {sample_image_path}")

# Run inference on the single image
results = model(sample_image_path)

# results is a list with one element (one image)
result = results[0]

# Print detection details
print(f"\nImage: {sample_image_path}")
print(f"Number of objects detected: {len(result.boxes)}")

for i, box in enumerate(result.boxes):
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]
    confidence = float(box.conf[0])
    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
    print(f"  Object {i+1}: {class_name} "
          f"(confidence: {confidence:.2f}) "
          f"bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

# Save the result image (with bounding boxes drawn)
output_path = os.path.join("output_images", "practice_single_result.jpg")
os.makedirs("output_images", exist_ok=True)
result.save(output_path)
print(f"\nResult saved to: {output_path}")

# ---------------------------------------------------------------
# Step 3: Perform object detection on MULTIPLE images
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Object detection on multiple images")
print("=" * 60)

# Download a few more sample images
multi_image_paths = []
image_urls = [
    ("https://ultralytics.com/images/zidane.jpg", "zidane.jpg"),
    ("https://ultralytics.com/images/street.jpg", "street.jpg"),
    ("https://ultralytics.com/images/dogs.jpg", "dogs.jpg"),
]

for url, filename in image_urls:
    path = os.path.join(sample_dir, filename)
    if not os.path.exists(path):
        print(f"Downloading: {filename}")
        import urllib.request
        urllib.request.urlretrieve(url, path)
    multi_image_paths.append(path)

# Run inference on all images at once (batch mode)
print(f"\nRunning inference on {len(multi_image_paths)} images...")
results = model(multi_image_paths)

# Process and save each result
for i, (result, img_path) in enumerate(zip(results, multi_image_paths)):
    img_name = os.path.basename(img_path)
    print(f"\n  Image {i+1}: {img_name}")
    print(f"  Objects detected: {len(result.boxes)}")

    for j, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        print(f"    - {class_name}: {confidence:.2f}")

    # Save the result
    output_path = os.path.join("output_images", f"practice_multi_{img_name}")
    result.save(output_path)
    print(f"  Saved to: {output_path}")

# ---------------------------------------------------------------
# Step 4: Practice 2 — Test on your own images
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Practice 2 — Test on your own images")
print("=" * 60)

# This section shows how to test on any image you provide.
# Simply place your image in the sample_images/ folder and
# update the path below.

own_image_path = os.path.join(sample_dir, "my_image.jpg")
if os.path.exists(own_image_path):
    print(f"Found your image: {own_image_path}")
    results = model(own_image_path)
    result = results[0]

    print(f"Objects detected: {len(result.boxes)}")
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        print(f"  Object {i+1}: {class_name} "
              f"(confidence: {confidence:.2f}) "
              f"bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

    output_path = os.path.join("output_images", "practice_own_result.jpg")
    result.save(output_path)
    print(f"Result saved to: {output_path}")
else:
    print(f"No custom image found at: {own_image_path}")
    print("To test on your own image:")
    print(f"  1. Place an image at: {own_image_path}")
    print("  2. Re-run this script")
    print("  3. Or modify 'own_image_path' in this script to point to your image")

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Practice complete!")
print("=" * 60)
print(f"Sample images: {sample_dir}/")
print(f"Output images: output_images/")
print("\nKey takeaways:")
print("  - YOLO loads pre-trained models with a single line: YOLO('yolov8n.pt')")
print("  - Inference is simple: model('image.jpg') returns results")
print("  - Each result has .boxes with .cls (class), .conf (confidence), .xyxy (bbox)")
print("  - Use result.save('path.jpg') to save the image with bounding boxes")
print("  - You can run inference on multiple images at once (batch mode)")
