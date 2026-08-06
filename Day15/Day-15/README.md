# Day-15: Object Detection using YOLOv8

## What is Object Detection?

Object detection is a computer vision technique that **identifies what objects are present in an image and where they are located**. Unlike image classification (which predicts a single label for the entire image), object detection draws **bounding boxes** around each object, assigns a **class label** (e.g., "apple", "person"), and provides a **confidence score** (how sure the model is).

### Key Concepts:
- **Bounding Box**: A rectangle `[x1, y1, x2, y2]` that outlines an object in the image.
- **Class Label**: The category of the detected object (e.g., "apple", "banana").
- **Confidence Score**: A value between 0 and 1 indicating how confident the model is about the detection.

### Real-world Applications:
- Autonomous vehicles (detecting pedestrians, cars, traffic signs)
- Medical imaging (detecting tumors, abnormalities)
- Security and surveillance (detecting intruders)
- Retail (shelf monitoring, checkout-free stores)
- Agriculture (detecting crops, pests, diseases)

---

## How is Object Detection Different from Image Classification?

| Feature | Image Classification | Object Detection |
|---|---|---|
| **Output** | Single label for the whole image | Multiple labels + bounding boxes |
| **Scope** | "What is in this image?" | "What is in this image AND where is it?" |
| **Objects** | One main object | Multiple objects |
| **Location** | No location info | Precise bounding box coordinates |

---

## What is YOLO?

**YOLO (You Only Look Once)** is a state-of-the-art, real-time object detection system. Unlike traditional two-stage detectors (like R-CNN), YOLO treats detection as a **single neural network** that predicts bounding boxes and class probabilities in one pass — making it extremely fast.

### Why YOLO is Popular:
- **Fast**: Can run in real-time (45+ FPS)
- **Accurate**: High accuracy on standard benchmarks
- **Simple**: Single neural network, end-to-end
- **Practical**: Easy to use with the Ultralytics YOLO library

### YOLO Workflow:
1. **Load a pre-trained model** (e.g., `yolov8n.pt`)
2. **Run inference** on an image: `model("image.jpg")`
3. **Get results**: bounding boxes, class labels, confidence scores
4. **Visualize**: Draw boxes on the image and save

### YOLO Annotation Format:
Each image has a corresponding `.txt` file with one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized (0 to 1).

### Evaluation Metrics:
- **mAP (mean Average Precision)**: Average precision across all classes
- **Precision**: Of all detected objects, how many were correct
- **Recall**: Of all actual objects, how many were detected

---

## Dataset Used

**Fruit Disease Detection Dataset** — sourced from public image repositories (Roboflow Universe / Wikimedia Commons).

- **Images**: 9 fruit images (apple, banana, orange, tomato, grapes, strawberry)
- **Model**: YOLOv8n (pre-trained on COCO dataset, 80 classes)
- **Note**: We used a pre-trained model for inference only (no training). The COCO dataset includes classes for apple, banana, orange, tomato, and grape.

---

## What Objects Were Detected?

| Image | Detected Objects | Confidence |
|---|---|---|
| apple.jpg | apple | 0.79 |
| banana.jpg | banana | 0.84 |
| grapes_fruit_Image_1.jpg | (none) | — |
| grapes_fruit_Image_2.jpg | person (x4) | 0.25–0.39 |
| orange.jpg | (none) | — |
| strawberry_fruit_Image_1.jpg | person (x2), car (x2) | 0.37–0.88 |
| strawberry_fruit_Image_2.jpg | person (x2), sports ball | 0.28–0.48 |
| tomato_fruit_Image_1.jpg | person, train | 0.25–0.82 |
| tomato_fruit_Image_2.jpg | person (x3) | 0.28–0.78 |

---

## Observations

1. **Apple and Banana**: Detected with high confidence (0.79 and 0.84). The pre-trained YOLOv8n model correctly identified these fruits.

2. **Orange**: Not detected. The COCO dataset does not include "orange" as a class, so the model could not identify it.

3. **Grapes**: Not detected as grapes. The COCO dataset does not have a "grape" class. The model detected people in the background instead.

4. **Strawberries**: Not detected as strawberries. The COCO dataset does not include "strawberry" as a class. The model detected people and cars in the background.

5. **Tomatoes**: Not detected as tomatoes. The COCO dataset does not include "tomato" as a class. The model detected people and a train instead.

6. **Key Insight**: The pre-trained YOLOv8n model is trained on the COCO dataset (80 classes). It can detect apples and bananas (which are in COCO), but cannot detect oranges, grapes, strawberries, or tomatoes (which are not in COCO). To detect these, you would need to **train a custom YOLO model** on a fruit-specific dataset.

7. **Confidence Threshold**: Some detections had low confidence (0.25–0.39), which may indicate false positives. Adjusting the confidence threshold can help filter these out.

---

## Project Structure

```
Day-15/
├── yolo_practice.py          # YOLO Practice Script (Practice 1 & 2)
├── object_detection.py       # Object Detection Script
├── app.py                    # Gradio web app for image upload/detection
├── create_recording.py       # Script to generate screen recording
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── sample_images/            # Sample input images for practice
│   ├── sample/               # Single image for Step 2
│   │   └── sample_img.jpg
│   ├── multiple_images/      # Multiple images for Step 3
│   │   ├── apple.jpg
│   │   ├── banana.jpg
│   │   └── people.jpg
│   └── custom/               # Custom images for Practice 2
│       ├── people.jpg
│       ├── people (2).jpg
│       ├── people (3).jpg
│       └── people (4).jpg
├── dataset/                  # Fruit disease detection dataset
│   ├── images/               # Input images (apple, banana, orange, etc.)
│   └── labels/               # YOLO format labels (empty - no training)
├── output_images/            # Output images with bounding boxes
│   ├── detection_*.jpg       # Object detection results
│   ├── practice_*.jpg        # YOLO practice results
│   └── detection_summary.txt # Summary report of all detections
├── yolov8n.pt                # Pre-trained YOLOv8 nano model
└── screen_recording.mp4      # Screen recording explaining the implementation
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the YOLO Practice Script
```bash
python yolo_practice.py
```
This demonstrates basic YOLO usage: loading a model, detecting objects on single and multiple images, and saving results.

### 3. Run the Object Detection Script
```bash
python object_detection.py
```
This runs object detection on the fruit dataset and saves results with bounding boxes.

### 4. Launch the Gradio App
```bash
python app.py
```
Open the link shown in the terminal. Upload an image to see real-time object detection results.

---

## Gradio App

The Gradio app (`app.py`) provides a web interface where users can:
- Upload an image
- Click "Detect Objects" to run YOLOv8 inference
- View the results with bounding boxes, class labels, and confidence scores

**Live Demo**: [Gradio App Link](https://your-username.gradio.app) (replace with actual link when deployed)

---

## Requirements

```
ultralytics>=8.0
opencv-python>=4.8
numpy>=1.24
pillow>=10.0
gradio>=4.0
```

---

## Key Takeaways

- Object detection identifies **what** objects are present and **where** they are located.
- YOLO is a fast, accurate, and simple object detection system.
- The pre-trained YOLOv8n model can detect 80 common object classes from the COCO dataset.
- For custom objects (like specific fruit diseases), you need to **train a custom YOLO model** on a labeled dataset.
- Bounding boxes, class labels, and confidence scores are the three key outputs of object detection.
