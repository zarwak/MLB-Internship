"""
Runs the Document Image Enhancement Tool on every image inside input_images/.
Usage:  python process.py
Enhanced results are saved into output_images/ with the same filename.
"""
import os
import cv2
from enhancer import enhance_document

# folder this file lives in, so input/output always live HERE
# no matter which folder you were in when you ran "python process.py"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXT)]
    if not files:
        print(f"No images found in {INPUT_DIR}/")
        print("Add some document photos there (or run generate_test_image.py) and try again.")
        return

    for filename in files:
        in_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(in_path)
        if img is None:
            print(f"Skipped (could not read): {filename}")
            continue

        print(f"Processing: {filename}")
        result = enhance_document(img)  # run the full pipeline: tilt -> gray -> denoise -> contrast -> sharpen

        out_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(out_path, result)
        print(f"  -> Saved: {out_path}")


if __name__ == "__main__":
    main()
