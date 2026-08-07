"""
Challenge Task
==============
Applies ALL operations from the toolkit on 5 different images:
1. Landscape
2. Person
3. Vehicle
4. Document
5. Object

For each image, the following operations are applied:
- Image info (dimensions, channels, file size)
- Grayscale conversion
- Resize (50%, 25%, 300x300, 800x600, 200%)
- Crop (4 quadrants + center)
- Rotate (90 CW, 90 CCW, 180, 270)
- Flip (horizontal, vertical, both)
- Draw shapes (rectangle, circle, line, polygon, text)

All results are saved to ../output_images/ with descriptive filenames.

Usage:
    python coding_practice/challenge_task.py
"""

import cv2
import os
import sys
import numpy as np
from datetime import datetime
from utils import get_output_dir

# Add the coding_practice directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_output_dir


def get_image_info(img, image_path):
    """Print image information."""
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    file_size = os.path.getsize(image_path)
    print(f"  Dimensions: {width}x{height} | Channels: {channels} | "
          f"File size: {file_size / 1024:.2f} KB")


def apply_all_operations(image_path, output_dir):
    """Apply all toolkit operations to a single image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ERROR: Could not read {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    h, w = img.shape[:2]

    print(f"\n  Processing: {os.path.basename(image_path)}")
    get_image_info(img, image_path)

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_challenge_grayscale.jpg"), gray)
    print(f"    ✓ Grayscale conversion")

    # 2. Resize
    for name, size in [("50p", (w // 2, h // 2)), ("25p", (w // 4, h // 4)),
                        ("300x300", (300, 300)), ("800x600", (800, 600))]:
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_challenge_resized_{name}.jpg"), resized)
    print(f"    ✓ Resize (4 resolutions)")

    # 3. Crop
    crops = [
        ("top_left", img[0:h//2, 0:w//2]),
        ("top_right", img[0:h//2, w//2:w]),
        ("bottom_left", img[h//2:h, 0:w//2]),
        ("bottom_right", img[h//2:h, w//2:w]),
        ("center", img[h//4:3*h//4, w//4:3*w//4]),
    ]
    for name, cropped in crops:
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_challenge_cropped_{name}.jpg"), cropped)
    print(f"    ✓ Crop (5 regions)")

    # 4. Rotate
    rotations = [
        ("90_cw", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ("90_ccw", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("180", cv2.rotate(img, cv2.ROTATE_180)),
        ("270", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),  # 270 CW = 90 CCW
    ]
    for name, rotated in rotations:
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_challenge_rotated_{name}.jpg"), rotated)
    print(f"    ✓ Rotate (90, 180, 270)")

    # 5. Flip
    flips = [
        ("horizontal", cv2.flip(img, 1)),
        ("vertical", cv2.flip(img, 0)),
        ("both", cv2.flip(img, -1)),
    ]
    for name, flipped in flips:
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_challenge_flipped_{name}.jpg"), flipped)
    print(f"    ✓ Flip (horizontal, vertical, both)")

    # 6. Draw shapes
    canvas = img.copy()
    # Rectangle
    cv2.rectangle(canvas, (20, 20), (150, 120), (0, 255, 0), 3)
    # Circle
    cv2.circle(canvas, (w - 100, 80), 50, (0, 0, 255), 3)
    # Line
    cv2.line(canvas, (20, 20), (w - 20, h - 20), (255, 0, 0), 2)
    # Triangle (polygon)
    pts = np.array([[w // 2, 100], [w // 2 - 80, 200], [w // 2 + 80, 200]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], True, (0, 255, 255), 3)
    # Text
    text = f"zarwa | {datetime.now().strftime('%B %d, %Y')}"
    cv2.putText(canvas, text, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_challenge_annotated.jpg"), canvas)
    print(f"    ✓ Draw shapes (rectangle, circle, line, polygon, text)")

    # 7. Bonus: Brightness & Contrast
    adjusted = img.copy().astype(np.float32)
    adjusted = adjusted * 1.3 + 30  # 30% more contrast, +30 brightness
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_challenge_brightness.jpg"), adjusted)
    print(f"    ✓ Brightness & contrast adjustment")

    # 8. Bonus: BGR to RGB comparison
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_challenge_rgb.jpg"), rgb)
    print(f"    ✓ BGR to RGB conversion")

    # 9. Bonus: Side-by-side collage
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    collage_top = np.hstack((cv2.resize(img, (300, 300)), cv2.resize(gray_bgr, (300, 300))))
    collage_bottom = np.hstack((cv2.resize(rotations[2][1], (300, 300)),
                                cv2.resize(flips[0][1], (300, 300))))
    collage = np.vstack((collage_top, collage_bottom))
    cv2.putText(collage, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(collage, "Grayscale", (310, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(collage, "Rotated 180", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(collage, "Flipped H", (310, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_challenge_collage.jpg"), collage)
    print(f"    ✓ Side-by-side collage")


def main():
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_images")
    output_dir = get_output_dir()

    images = sorted([f for f in os.listdir(sample_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])

    print("=" * 60)
    print("CHALLENGE TASK: Applying all operations on 5 images")
    print("=" * 60)

    for img_name in images:
        img_path = os.path.join(sample_dir, img_name)
        apply_all_operations(img_path, output_dir)

    print("\n" + "=" * 60)
    print("Challenge task complete! All results saved to output_images/")
    print("=" * 60)


if __name__ == "__main__":
    main()
