"""
07 - Draw Shapes
================
Draws various shapes on an image:
- Rectangle
- Circle
- Line
- Polygon (filled and outline)
- Custom text (name and date)

Uses OpenCV drawing functions:
- cv2.rectangle()
- cv2.circle()
- cv2.line()
- cv2.polylines() / cv2.fillPoly()
- cv2.putText()

Saves the annotated image to ../output_images/.

Usage:
    python 07_draw_shapes.py [image_path]
"""

import cv2
import os
import sys
import numpy as np
from datetime import datetime
from utils import get_image_path, get_output_dir, safe_imshow


def draw_shapes(image_path, output_dir):
    """Draw various shapes and text on an image and save it."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Work on a copy to preserve the original
    canvas = img.copy()
    height, width = canvas.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"Original image size: {width}x{height}")
    print()

    # 1. Draw a Rectangle (top-left corner)
    cv2.rectangle(canvas, (20, 20), (150, 120), (0, 255, 0), 3)
    print("  Drawn: Rectangle at (20,20) to (150,120) - Green")

    # 2. Draw a Circle (top-right area)
    cv2.circle(canvas, (width - 100, 80), 50, (0, 0, 255), 3)
    print(f"  Drawn: Circle at center ({width - 100}, 80) radius 50 - Red")

    # 3. Draw a Line (diagonal from top-left to bottom-right)
    cv2.line(canvas, (20, 20), (width - 20, height - 20), (255, 0, 0), 2)
    print(f"  Drawn: Line from (20,20) to ({width - 20}, {height - 20}) - Blue")

    # 4. Draw a Polygon (triangle in the center)
    pts = np.array([[width // 2, 100],
                    [width // 2 - 80, 200],
                    [width // 2 + 80, 200]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
    print("  Drawn: Triangle (polygon) in center - Yellow")

    # 5. Draw a filled polygon (pentagon)
    pentagon_pts = np.array([[width // 2, 280],
                             [width // 2 - 50, 330],
                             [width // 2 - 30, 390],
                             [width // 2 + 30, 390],
                             [width // 2 + 50, 330]], np.int32)
    pentagon_pts = pentagon_pts.reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [pentagon_pts], color=(255, 0, 255))
    print("  Drawn: Filled pentagon below triangle - Purple")

    # 6. Add custom text (name and date)
    name = "zarwa"
    date_str = datetime.now().strftime("%B %d, %Y")
    text = f"{name} | {date_str}"

    # Put text at the bottom of the image
    cv2.putText(canvas, text, (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    print(f"  Drawn: Text '{text}' at bottom")

    # Save the annotated image
    output_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    cv2.imwrite(output_path, canvas)
    print(f"\n  Annotated image saved to: {output_path}")

    return canvas


def main():
    image_path = get_image_path()
    if image_path is None:
        return

    output_dir = get_output_dir()

    print(f"\nDrawing shapes on image: {image_path}\n")
    result = draw_shapes(image_path, output_dir)

    if result is not None:
        safe_imshow("Drawn Shapes - Press any key to close", result)


if __name__ == "__main__":
    main()
