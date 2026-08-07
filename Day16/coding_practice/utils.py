"""
Utility functions for OpenCV practice programs.
Provides a safe image display function that handles headless environments.
"""

import cv2
import os
import sys


def get_sample_image(sample_dir=None, index=0):
    """Get the path to a sample image.

    Args:
        sample_dir: Path to the sample images directory.
                    If None, uses ../sample_images relative to caller.
        index: Index of the image to use (sorted alphabetically).

    Returns:
        Path to the image file, or None if no images found.
    """
    if sample_dir is None:
        sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_images")

    images = sorted([f for f in os.listdir(sample_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    if not images:
        print("No sample images found in ../sample_images/")
        return None

    return os.path.join(sample_dir, images[index])


def get_image_path():
    """Get image path from command line argument or default sample image."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return get_sample_image()


def get_output_dir():
    """Get the output directory path, creating it if needed."""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output_images")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def safe_imshow(window_name, image):
    """Display an image, handling headless environments gracefully.

    In environments without a display (e.g., servers, CI), cv2.imshow()
    will raise an error. This function catches that error and prints
    a message instead.
    """
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print(f"  [Note: Image display not available in this environment. "
              f"Output saved to files instead.]")
