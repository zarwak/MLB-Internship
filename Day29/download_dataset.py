"""
Day 29 - Download the Road Damage Detection dataset from Roboflow Universe.

Source: https://universe.roboflow.com/road-damage-detection-ds22n/road-damage-dataset-8jvz5
License: Public Domain
Classes (8): pothole, Alligator, Edge Cracking, Lateral-Crack,
             Longitudinal-Crack, Ravelling, Rutting, Striping

Requires a free Roboflow API key in a .env file next to this script:
    ROBOFLOW_API_KEY=your_key_here
"""

import os
import sys

from dotenv import load_dotenv

DATASET_DIR = "dataset"

load_dotenv()
api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    sys.exit(
        "ROBOFLOW_API_KEY not found. Create a .env file next to this script "
        "with:\n    ROBOFLOW_API_KEY=your_key_here\n"
        "(get a free key at https://app.roboflow.com -> Settings -> API Keys)"
    )

from roboflow import Roboflow  # noqa: E402  (import after the key check)

rf = Roboflow(api_key=api_key)
project = rf.workspace("road-damage-detection-ds22n").project("road-damage-dataset-8jvz5")
version = project.version(2)

print(f"Downloading '{project.name}' v2 (YOLOv8 format) into ./{DATASET_DIR} ...")
dataset = version.download("yolov8", location=DATASET_DIR, overwrite=True)
print(f"Done. Dataset location: {dataset.location}")
