"""
04 - Crop Image
===============
Crops different regions of an image:
- Top-left quadrant
- Top-right quadrant
- Bottom-left quadrant
- Bottom-right quadrant
- Center region

Saves all cropped images to ../output_images/.

Usage:
    python 04_crop_image.py [image_path]
"""

import cv2
import os
import sys
import numpy as np
from utils import get_image_path, get_output_dir, safe_imshow


def crop_image(image_path, output_dir):
    """Crop different regions of an image and save them."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    height, width = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"Original image size: {width}x{height}")
    print()

    # Define crop regions as (name, y_start, y_end, x_start, x_end)
    crops = [
        ("top_left", 0, height // 2, 0, width // 2),
        ("top_right", 0, height // 2, width // 2, width),
        ("bottom_left", height // 2, height, 0, width // 2),
        ("bottom_right", height // 2, height, width // 2, width),
        ("center", height // 4, 3 * height // 4, width // 4, 3 * width // 4),
    ]

    results = []
    for name, y1, y2, x1, x2 in crops:
        cropped = img[y1:y2, x1:x2]
        h, w = cropped.shape[:2]
        output_path = os.path.join(output_dir, f"{base_name}_cropped_{name}.jpg")
        cv2.imwrite(output_path, cropped)
        print(f"  {name:15s} -> {w}x{h}  saved as {os.path.basename(output_path)}")
        results.append((name, cropped))

    # Create a 2x2 collage of the four quadrants
    top_row = np.hstack((results[0][1], results[1][1]))
    bottom_row = np.hstack((results[2][1], results[3][1]))
    collage = np.vstack((top_row, bottom_row))

    collage_path = os.path.join(output_dir, f"{base_name}_crop_collage.jpg")
    cv2.imwrite(collage_path, collage)
    print(f"\n  Collage saved as {os.path.basename(collage_path)}")

    return results


def main():
    image_path = get_image_path()
    if image_path is None:
        return

    output_dir = get_output_dir()

    print(f"\nCropping image: {image_path}\n")
    results = crop_image(image_path, output_dir)

    if results is not None:
        # Display the center crop
        safe_imshow("Center Crop - Press any key to close", results[4][1])


if __name__ == "__main__":
    main()
