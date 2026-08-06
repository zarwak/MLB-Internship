"""
Object Detection using YOLOv11
==============================
This script performs object detection on a Fruit Disease Detection dataset
using a pre-trained YOLOv11 model.

Dataset: Fruit Disease Detection (from Roboflow Universe / public sources)
  - Images: Various fruit disease images in Dataset/images/
  - The pre-trained YOLOv11n model (trained on COCO) is used for inference.
  - COCO classes include: apple, banana, orange, tomato, grape, etc.

What this script does:
  1. Loads a pre-trained YOLOv11n model
  2. Runs inference on all images in the Dataset/images/ folder
  3. Prints detected objects, confidence scores, and bounding boxes
  4. Saves output images with bounding boxes drawn on them
  5. Creates a summary report of all detections

Requirements:
  pip install ultralytics pillow

Usage:
  python object_detection.py
"""

import os
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
MODEL_NAME = "yolo11n.pt"          # Pre-trained YOLOv11 nano model
DATASET_DIR = "Dataset"              # Folder containing the dataset
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
OUTPUT_DIR = "output_images"         # Folder to save detection results
OUTPUT_SUBDIR = "fruit disease detection"  # Subfolder for detection results
CONFIDENCE_THRESHOLD = 0.25          # Minimum confidence to show a detection

# Create output directories if they don't exist
output_subdir_path = os.path.join(OUTPUT_DIR, OUTPUT_SUBDIR)
os.makedirs(output_subdir_path, exist_ok=True)

# ---------------------------------------------------------------
# Step 1: Load the pre-trained YOLOv11 model
# ---------------------------------------------------------------
print("=" * 70)
print("Object Detection using YOLOv11")
print("=" * 70)
print(f"\nLoading pre-trained model: {MODEL_NAME}")
print("This model was trained on the COCO dataset (80 common object classes).")
print("It can detect fruits like: apple, banana, orange, tomato, grape, etc.")

model = YOLO(MODEL_NAME)
print(f"Model loaded successfully!\n")

# ---------------------------------------------------------------
# Step 2: Get list of images to process
# ---------------------------------------------------------------
image_files = []
for f in sorted(os.listdir(IMAGES_DIR)):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        image_files.append(os.path.join(IMAGES_DIR, f))

print(f"Found {len(image_files)} images to process in: {IMAGES_DIR}/")
print("-" * 70)

# ---------------------------------------------------------------
# Step 3: Run inference on each image
# ---------------------------------------------------------------
all_detections = []  # Store all detection results for summary

for img_path in image_files:
    img_name = os.path.basename(img_path)
    print(f"\nProcessing: {img_name}")

    # Run YOLO inference on the image
    results = model(img_path, conf=CONFIDENCE_THRESHOLD)
    result = results[0]  # Single image result

    # Get image dimensions for reference
    try:
        img_pil = Image.open(img_path)
        w, h = img_pil.size
        print(f"  Image size: {w}x{h} pixels")
    except Exception:
        pass

    # Print detected objects
    num_detections = len(result.boxes)
    print(f"  Objects detected: {num_detections}")

    detections = []
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        detection_info = {
            "image": img_name,
            "class": class_name,
            "confidence": confidence,
            "bbox": bbox,
        }
        detections.append(detection_info)
        all_detections.append(detection_info)

        print(f"    {i+1}. {class_name} "
              f"(confidence: {confidence:.2f}) "
              f"bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

    # Save the output image with bounding boxes drawn
    output_path = os.path.join(output_subdir_path, f"detection_{img_name}")
    result.save(output_path)
    print(f"  Output saved to: {output_path}")

# ---------------------------------------------------------------
# Step 4: Summary Report
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY REPORT")
print("=" * 70)

# Count detections by class
class_counts = {}
for det in all_detections:
    cls = det["class"]
    if cls not in class_counts:
        class_counts[cls] = 0
    class_counts[cls] += 1

print(f"\nTotal images processed: {len(image_files)}")
print(f"Total objects detected: {len(all_detections)}")
print(f"\nDetections by class:")
for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {count}")

# Save summary to a text file
summary_path = os.path.join(output_subdir_path, "detection_summary.txt")
with open(summary_path, "w") as f:
    f.write("Object Detection Summary Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Total images processed: {len(image_files)}\n")
    f.write(f"Total objects detected: {len(all_detections)}\n\n")
    f.write("Detections by class:\n")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        f.write(f"  {cls}: {count}\n")
    f.write("\nDetailed detections:\n")
    f.write("-" * 50 + "\n")
    for det in all_detections:
        f.write(f"  Image: {det['image']}\n")
        f.write(f"  Object: {det['class']}\n")
        f.write(f"  Confidence: {det['confidence']:.2f}\n")
        f.write(f"  Bounding Box: [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
                f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]\n\n")

print(f"\nSummary report saved to: {summary_path}")
print("\n" + "=" * 70)
print("Object detection complete!")
print("=" * 70)
