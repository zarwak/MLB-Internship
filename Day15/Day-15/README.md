# Day-15: Object Detection using YOLOv11

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
1. **Load a pre-trained model** (e.g., `yolo11n.pt`)
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

**Fruit Disease Detection Dataset** — sourced from Roboflow Universe.

- **Images**: 172 fruit disease images in `Dataset/images/`
- **Labels**: YOLO format annotations in `Dataset/labels/`
- **Model**: YOLOv11n (pre-trained on COCO dataset, 80 classes)
- **Note**: We used a pre-trained model for inference only (no training). The COCO dataset includes classes for apple, banana, orange, tomato, grape, etc.

---

## What Objects Were Detected?

The YOLOv11n model was run on all 172 images in the Fruit Disease Detection dataset. Here are the detection results:

| Class | Count |
|---|---|
| apple | 36 |
| orange | 26 |
| vase | 25 |
| person | 17 |
| banana | 14 |
| cake | 12 |
| broccoli | 7 |
| pizza | 7 |
| bird | 5 |
| donut | 3 |
| cup | 2 |
| bear | 2 |
| carrot | 2 |
| keyboard | 1 |
| wine glass | 1 |
| bottle | 1 |
| cat | 1 |
| bowl | 1 |
| fork | 1 |
| dog | 1 |

**Total images processed**: 172
**Total objects detected**: 165

### Sample Detections:
- **apple**: Detected with confidence 0.35–0.91 (36 detections)
- **orange**: Detected with confidence 0.81 (26 detections)
- **banana**: Detected with confidence 0.25–0.85 (14 detections)
- **vase**: Detected with confidence 0.37–0.85 (25 detections)

---

## Observations

1. **Apple**: The most detected fruit (36 detections). The pre-trained YOLOv11n model correctly identified apples in many images, with confidence scores ranging from 0.35 to 0.91.

2. **Orange**: Detected in 26 images with high confidence (0.81). The COCO dataset includes "orange" as a class, so the model can identify it.

3. **Banana**: Detected in 14 images with varying confidence (0.25–0.85). Some detections had lower confidence, possibly due to image quality or occlusion.

4. **Vase**: Detected in 25 images. This is likely a false positive — the model is detecting vase-like shapes in fruit disease images.

5. **Person**: Detected in 17 images. These are likely people in the background of some fruit images.

6. **Other detections**: cake, broccoli, pizza, bird, donut, etc. — these are likely false positives or background objects in the images.

7. **Key Insight**: The pre-trained YOLOv11n model is trained on the COCO dataset (80 classes). It can detect apples, bananas, and oranges (which are in COCO), but cannot detect specific fruit diseases. To detect fruit diseases, you would need to **train a custom YOLO model** on a fruit disease-specific dataset.

8. **Confidence Threshold**: Some detections had low confidence (0.25–0.39), which may indicate false positives. Adjusting the confidence threshold can help filter these out.

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
├── Dataset/                  # Fruit disease detection dataset
│   ├── images/               # 172 input images (fruit disease photos)
│   ├── labels/               # YOLO format labels
│   ├── README.dataset.txt    # Dataset info
│   └── README.roboflow.txt   # Roboflow info
├── output_images/            # Output images with bounding boxes
│   ├── fruit disease detection/  # Detection results for Dataset images
│   │   ├── detection_*.jpg       # Detection results for each image
│   │   └── detection_summary.txt # Summary report of all detections
│   ├── practice_*.jpg        # YOLO practice results
│   └── practice_single_result.jpg
├── yolo11n.pt                # Pre-trained YOLOv11 nano model
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
This runs object detection on the Fruit Disease Detection dataset (172 images) and saves results with bounding boxes in `output_images/fruit disease detection/`.

### 4. Launch the Gradio App
```bash
python app.py
```
Open the link shown in the terminal. Upload an image to see real-time object detection results.

---

## Gradio App

The Gradio app (`app.py`) provides a web interface where users can:
- Upload an image
- Click "Detect Objects" to run YOLOv11 inference
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
torch>=2.0
```

---

## Key Takeaways

- Object detection identifies **what** objects are present and **where** they are located.
- YOLO is a fast, accurate, and simple object detection system.
- The pre-trained YOLOv11n model can detect 80 common object classes from the COCO dataset.
- For custom objects (like specific fruit diseases), you need to **train a custom YOLO model** on a labeled dataset.
- Bounding boxes, class labels, and confidence scores are the three key outputs of object detection.
