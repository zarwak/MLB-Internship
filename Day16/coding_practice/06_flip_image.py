"""
06 - Flip Image
===============
Flips an image:
- Horizontally (flip around y-axis)
- Vertically (flip around x-axis)
- Both horizontally and vertically (180 degree rotation)

Uses cv2.flip() with flip codes:
- 0: Vertical flip
- 1: Horizontal flip
- -1: Both horizontal and vertical flip

Saves all flipped images to ../output_images/.

Usage:
    python 06_flip_image.py [image_path]
"""

import cv2
import os
import sys
import numpy as np
from utils import get_image_path, get_output_dir, safe_imshow


def flip_image(image_path, output_dir):
    """Flip an image horizontally, vertically, and both, then save them."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    height, width = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"Original image size: {width}x{height}")
    print()

    # Flip operations
    flip_horizontal = cv2.flip(img, 1)   # Horizontal flip (y-axis)
    flip_vertical = cv2.flip(img, 0)    # Vertical flip (x-axis)
    flip_both = cv2.flip(img, -1)       # Both flips

    operations = [
        ("horizontal", flip_horizontal),
        ("vertical", flip_vertical),
        ("both", flip_both),
    ]

    results = []
    for name, flipped in operations:
        h, w = flipped.shape[:2]
        output_path = os.path.join(output_dir, f"{base_name}_flipped_{name}.jpg")
        cv2.imwrite(output_path, flipped)
        print(f"  {name:12s} -> {w}x{h}  saved as {os.path.basename(output_path)}")
        results.append((name, flipped))

    # Create a 2x2 collage: original, horizontal, vertical, both
    top_row = np.hstack((img, results[0][1]))
    bottom_row = np.hstack((results[1][1], results[2][1]))
    collage = np.vstack((top_row, bottom_row))

    # Add labels
    cv2.putText(collage, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(collage, "Horizontal", (width + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(collage, "Vertical", (10, height + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(collage, "Both", (width + 10, height + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    collage_path = os.path.join(output_dir, f"{base_name}_flip_collage.jpg")
    cv2.imwrite(collage_path, collage)
    print(f"\n  Collage saved as {os.path.basename(collage_path)}")

    return results


def main():
    image_path = get_image_path()
    if image_path is None:
        return

    output_dir = get_output_dir()

    print(f"\nFlipping image: {image_path}\n")
    results = flip_image(image_path, output_dir)

    if results is not None:
        # Display the horizontal flip
        safe_imshow("Horizontal Flip - Press any key to close", results[0][1])


if __name__ == "__main__":
    main()
