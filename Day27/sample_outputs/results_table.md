# Day 27 - Detection results summary

Model: YOLO11n, confidence threshold: 0.25

## Per-image results

| Image | Objects found | Classes | Inference time |
|---|---|---|---|
| coco_cats_couch.jpg | 4 | cat, couch, remote | 8849 ms |
| coco_scene_632.jpg | 5 | bed, bottle, chair, potted plant | 287 ms |
| coco_scene_724.jpg | 2 | stop sign, truck | 270 ms |
| coco_scene_776.jpg | 3 | teddy bear | 257 ms |
| coco_scene_802.jpg | 2 | oven, refrigerator | 213 ms |
| coco_scene_872.jpg | 2 | person | 313 ms |
| coco_scene_885.jpg | 4 | person, tennis racket | 269 ms |
| coco_skier.jpg | 2 | person, skis | 168 ms |
| opencv_basketball.png | 2 | person | 178 ms |
| opencv_messi.jpg | 7 | person, sports ball | 226 ms |
| ultralytics_bus.jpg | 5 | bus, person | 192 ms |
| ultralytics_zidane.jpg | 3 | person, tie | 235 ms |

## Per-class summary (across all sample images)

| Class | Times detected | Min conf | Mean conf | Max conf |
|---|---|---|---|---|
| person | 20 | 0.26 | 0.70 | 0.92 |
| teddy bear | 3 | 0.39 | 0.64 | 0.91 |
| cat | 2 | 0.92 | 0.92 | 0.93 |
| potted plant | 2 | 0.83 | 0.85 | 0.88 |
| remote | 1 | 0.67 | 0.67 | 0.67 |
| couch | 1 | 0.50 | 0.50 | 0.50 |
| bed | 1 | 0.92 | 0.92 | 0.92 |
| bottle | 1 | 0.53 | 0.53 | 0.53 |
| chair | 1 | 0.36 | 0.36 | 0.36 |
| stop sign | 1 | 0.95 | 0.95 | 0.95 |
| truck | 1 | 0.34 | 0.34 | 0.34 |
| refrigerator | 1 | 0.91 | 0.91 | 0.91 |
| oven | 1 | 0.87 | 0.87 | 0.87 |
| tennis racket | 1 | 0.82 | 0.82 | 0.82 |
| skis | 1 | 0.66 | 0.66 | 0.66 |
| sports ball | 1 | 0.91 | 0.91 | 0.91 |
| bus | 1 | 0.94 | 0.94 | 0.94 |
| tie | 1 | 0.45 | 0.45 | 0.45 |

**41 total detections across 12 images, 18 unique classes.**
