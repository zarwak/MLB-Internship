"""
05 - Rotate Image
=================
Rotates an image by:
- 90 degrees clockwise
- 90 degrees counter-clockwise
- 180 degrees
- 270 degrees (equivalent to 90 CCW)

Uses cv2.rotate() for 90-degree multiples and cv2.warpAffine() for
arbitrary angles.

Saves all rotated images to ../output_images/.

Usage:
    python 05_rotate_image.py [image_path]
"""

import cv2
import os
import sys
import numpy as np
from utils import get_image_path, get_output_dir, safe_imshow


def rotate_image(image_path, output_dir):
    """Rotate an image by 90, 180, and 270 degrees and save them."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    height, width = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"Original image size: {width}x{height}")
    print()

    # Rotate using cv2.rotate()
    # 90 degrees clockwise
    rot_90_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # 90 degrees counter-clockwise (same as 270 clockwise)
    rot_90_ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # 180 degrees
    rot_180 = cv2.rotate(img, cv2.ROTATE_180)

    # Also rotate by 270 using warpAffine for demonstration
    # 270 clockwise = 90 counter-clockwise
    center = (width // 2, height // 2)
    matrix_270 = cv2.getRotationMatrix2D(center, 270, 1.0)
    rot_270 = cv2.warpAffine(img, matrix_270, (height, width))

    operations = [
        ("90_cw", rot_90_cw),
        ("90_ccw", rot_90_ccw),
        ("180", rot_180),
        ("270", rot_270),
    ]

    results = []
    for name, rotated in operations:
        h, w = rotated.shape[:2]
        output_path = os.path.join(output_dir, f"{base_name}_rotated_{name}.jpg")
        cv2.imwrite(output_path, rotated)
        print(f"  {name:10s} -> {w}x{h}  saved as {os.path.basename(output_path)}")
        results.append((name, rotated))

    # Create a 2x2 collage of all rotations (uniform 300x300 for display)
    top_row = np.hstack((cv2.resize(results[0][1], (300, 300)), cv2.resize(results[1][1], (300, 300))))
    bottom_row = np.hstack((cv2.resize(results[2][1], (300, 300)), cv2.resize(results[3][1], (300, 300))))
    collage = np.vstack((top_row, bottom_row))

    collage_path = os.path.join(output_dir, f"{base_name}_rotation_collage.jpg")
    cv2.imwrite(collage_path, collage)
    print(f"\n  Collage saved as {os.path.basename(collage_path)}")

    return results


def main():
    image_path = get_image_path()
    if image_path is None:
        return

    output_dir = get_output_dir()

    print(f"\nRotating image: {image_path}\n")
    results = rotate_image(image_path, output_dir)

    if results is not None:
        # Display the 180-degree rotation
        safe_imshow("180 Degree Rotation - Press any key to close", results[2][1])


if __name__ == "__main__":
    main()
