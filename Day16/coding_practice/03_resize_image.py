"""
03 - Resize Image
=================
Resizes an image to different resolutions:
- 50% of original size
- 25% of original size
- Fixed size: 300x300
- Fixed size: 800x600
- Upscaled: 200% of original size

Uses different interpolation methods:
- INTER_AREA for shrinking
- INTER_LINEAR for upscaling
- INTER_CUBIC for high-quality upscaling

Saves all resized images to ../output_images/.

Usage:
    python 03_resize_image.py [image_path]
"""

import cv2
import os
import sys
import numpy as np
from utils import get_image_path, get_output_dir, safe_imshow


def resize_image(image_path, output_dir):
    """Resize an image to multiple resolutions and save them."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    height, width = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"Original image size: {width}x{height}")
    print()

    # Define resize operations
    operations = [
        ("50_percent", cv2.resize(img, (width // 2, height // 2), interpolation=cv2.INTER_AREA)),
        ("25_percent", cv2.resize(img, (width // 4, height // 4), interpolation=cv2.INTER_AREA)),
        ("300x300", cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)),
        ("800x600", cv2.resize(img, (800, 600), interpolation=cv2.INTER_LINEAR)),
        ("200_percent", cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)),
    ]

    results = []
    for name, resized in operations:
        h, w = resized.shape[:2]
        output_path = os.path.join(output_dir, f"{base_name}_resized_{name}.jpg")
        cv2.imwrite(output_path, resized)
        print(f"  {name:15s} -> {w}x{h}  saved as {os.path.basename(output_path)}")
        results.append((name, resized))

    # Create a collage of all resized images (uniform 300x300 for display)
    collage_top = np.hstack((cv2.resize(img, (300, 300)), cv2.resize(operations[0][1], (300, 300))))
    collage_bottom = np.hstack((cv2.resize(operations[1][1], (300, 300)), cv2.resize(operations[3][1], (300, 300))))
    collage = np.vstack((collage_top, collage_bottom))

    collage_path = os.path.join(output_dir, f"{base_name}_resize_collage.jpg")
    cv2.imwrite(collage_path, collage)
    print(f"\n  Collage saved as {os.path.basename(collage_path)}")

    return results


def main():
    image_path = get_image_path()
    if image_path is None:
        return

    output_dir = get_output_dir()

    print(f"\nResizing image: {image_path}\n")
    results = resize_image(image_path, output_dir)

    if results is not None:
        # Display the 50% resized image
        safe_imshow("50% Resized - Press any key to close", results[0][1])


if __name__ == "__main__":
    main()
