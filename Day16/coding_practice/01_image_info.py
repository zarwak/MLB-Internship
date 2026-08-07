"""
01 - Image Info
===============
Reads an image and displays its:
- Dimensions (Height, Width)
- Number of channels
- File size (in bytes, KB, MB)
- Data type
- Pixel value range

Usage:
    python 01_image_info.py [image_path]

If no image path is provided, the first image in ../sample_images/ is used.
"""

import cv2
import os
import sys
from utils import get_image_path, safe_imshow


def get_image_info(image_path):
    """Read an image and print its properties."""
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Get file size
    file_size_bytes = os.path.getsize(image_path)

    # Get image properties
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    dtype = img.dtype

    # Pixel value range
    min_val = img.min()
    max_val = img.max()

    # Print information
    print("=" * 50)
    print(f"Image: {os.path.basename(image_path)}")
    print("=" * 50)
    print(f"  Height:       {height} pixels")
    print(f"  Width:        {width} pixels")
    print(f"  Channels:     {channels}")
    print(f"  Data type:    {dtype}")
    print(f"  Shape:        {img.shape}")
    print(f"  File size:    {file_size_bytes} bytes")
    print(f"                 {file_size_bytes / 1024:.2f} KB")
    print(f"                 {file_size_bytes / (1024 * 1024):.4f} MB")
    print(f"  Pixel range:  {min_val} to {max_val}")
    print("=" * 50)

    return img


def main():
    image_path = get_image_path()
    if image_path is None:
        return

    print(f"\nReading image: {image_path}\n")
    img = get_image_info(image_path)

    if img is not None:
        safe_imshow("Image Info - Press any key to close", img)


if __name__ == "__main__":
    main()
