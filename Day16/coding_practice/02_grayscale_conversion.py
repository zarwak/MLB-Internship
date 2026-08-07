"""
02 - Grayscale Conversion
=========================
Converts a color image to grayscale using two methods:
1. cv2.cvtColor() with COLOR_BGR2GRAY
2. cv2.imread() with cv2.IMREAD_GRAYSCALE flag

Displays both the original and grayscale images side by side.
Saves the grayscale image to ../output_images/.

Usage:
    python 02_grayscale_conversion.py [image_path]
"""

import cv2
import os
import sys
import numpy as np
from utils import get_image_path, get_output_dir, safe_imshow


def convert_to_grayscale(image_path, output_dir):
    """Convert a color image to grayscale and save it."""
    # Read the color image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Method 1: Using cvtColor
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Method 2: Using imread with IMREAD_GRAYSCALE
    gray2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Verify both methods produce the same result
    print(f"Method 1 (cvtColor) shape: {gray1.shape}")
    print(f"Method 2 (imread) shape:   {gray2.shape}")
    print(f"Both methods produce same result: {np.array_equal(gray1, gray2)}")

    # Save the grayscale image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_grayscale.jpg")
    cv2.imwrite(output_path, gray1)
    print(f"Grayscale image saved to: {output_path}")

    # Create a side-by-side comparison
    # Convert grayscale to 3-channel for display alongside color
    gray_bgr = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack((img, gray_bgr))

    # Add labels
    cv2.putText(comparison, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Grayscale", (img.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    comparison_path = os.path.join(output_dir, f"{base_name}_grayscale_comparison.jpg")
    cv2.imwrite(comparison_path, comparison)
    print(f"Comparison image saved to: {comparison_path}")

    return gray1, comparison


def main():
    image_path = get_image_path()
    if image_path is None:
        return

    output_dir = get_output_dir()

    print(f"\nConverting to grayscale: {image_path}\n")
    result = convert_to_grayscale(image_path, output_dir)

    if result is not None:
        gray, comparison = result
        safe_imshow("Original vs Grayscale", comparison)


if __name__ == "__main__":
    main()
