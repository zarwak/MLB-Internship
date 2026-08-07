"""
Runner script that executes all coding practice programs sequentially.
"""

import subprocess
import sys
import os

SCRIPTS = [
    "01_image_info.py",
    "02_grayscale_conversion.py",
    "03_resize_image.py",
    "04_crop_image.py",
    "05_rotate_image.py",
    "06_flip_image.py",
    "07_draw_shapes.py",
]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

for script in SCRIPTS:
    print("\n" + "=" * 60)
    print(f"Running: {script}")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr[:500]}")
    print(f"Exit code: {result.returncode}")
