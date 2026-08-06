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
# YOLOv11n is the smallest and fastest version — great for practice.
# Other options: yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
print("=" * 60)
print("Step 1: Loading pre-trained YOLOv11n model...")
print("=" * 60)

model = YOLO("yolo11n.pt")  # 'n' = nano (smallest, fastest)
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
sample_dir = "sample_images/sample"
os.makedirs(sample_dir, exist_ok=True)

# If no sample image exists, download one from the internet
sample_image_path = os.path.join(sample_dir, "sample_img.jpg")
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

multi_image_paths = [os.path.join("sample_images", "multiple_images", f) for f in os.listdir("sample_images/multiple_images") if f.endswith((".jpg", ".png"))]
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

# This section shows how to test on your own images.
# Place your images in the sample_images/custom/ folder.

custom_dir = "sample_images/custom"
os.makedirs(custom_dir, exist_ok=True)

# Find all images in the custom directory
custom_image_paths = [
    os.path.join(custom_dir, f)
    for f in sorted(os.listdir(custom_dir))
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
] if os.path.exists(custom_dir) else []

if custom_image_paths:
    print(f"Found {len(custom_image_paths)} custom image(s) in: {custom_dir}/")
    results = model(custom_image_paths)

    for i, (result, img_path) in enumerate(zip(results, custom_image_paths)):
        img_name = os.path.basename(img_path)
        print(f"\n  Custom Image {i+1}: {img_name}")
        print(f"  Objects detected: {len(result.boxes)}")

        for j, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            print(f"    Object {j+1}: {class_name} "
                  f"(confidence: {confidence:.2f}) "
                  f"bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

        # Save the result
        output_path = os.path.join("output_images", f"practice_custom_{img_name}")
        result.save(output_path)
        print(f"  Result saved to: {output_path}")
else:
    print(f"No custom images found in: {custom_dir}/")
    print("To test on your own images:")
    print(f"  1. Place your images in: {custom_dir}/")
    print("  2. Re-run this script")

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Practice complete!")
print("=" * 60)
print(f"Sample images: sample_images/sample/")
print(f"Multiple images: sample_images/multiple_images/")
print(f"Custom images: sample_images/custom/")
print(f"Output images: output_images/")
print("\nKey takeaways:")
print("  - YOLO loads pre-trained models with a single line: YOLO('yolo11n.pt')")
print("  - Inference is simple: model('image.jpg') returns results")
print("  - Each result has .boxes with .cls (class), .conf (confidence), .xyxy (bbox)")
print("  - Use result.save('path.jpg') to save the image with bounding boxes")
print("  - You can run inference on multiple images at once (batch mode)")
