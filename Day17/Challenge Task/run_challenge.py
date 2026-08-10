"""
Challenge Task: process 5 tilted document images through the real enhancement
pipeline (reused from the "Image Enhancement Tool" folder, not copy-pasted).

For EACH image, saves 3 versions into results/:
  <name>_1_original.jpg
  <name>_2_perspective_corrected.jpg
  <name>_3_final_enhanced.jpg
"""
import os
import sys
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "tilted_inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# the actual pipeline lives in the other folder - point Python to it instead
# of duplicating the code here
TOOL_DIR = os.path.join(BASE_DIR, "..", "image_enhancement_tool")
sys.path.insert(0, TOOL_DIR)
from enhancer import correct_perspective, enhance_document  # noqa: E402


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXT))
    if not files:
        print(f"No images found in {INPUT_DIR}")
        print("Run generate_test_documents.py first, or add your own 5 tilted photos there.")
        return

    if len(files) < 5:
        print(f"Note: only found {len(files)} image(s) - the challenge task needs 5.")

    for filename in files:
        name = os.path.splitext(filename)[0]
        in_path = os.path.join(INPUT_DIR, filename)
        original = cv2.imread(in_path)
        if original is None:
            print(f"Skipped (could not read): {filename}")
            continue

        print(f"Processing: {filename}")
        perspective_corrected = correct_perspective(original)   # step 1 only
        final_enhanced = enhance_document(original)              # full pipeline

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_1_original.jpg"), original)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_2_perspective_corrected.jpg"), perspective_corrected)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_3_final_enhanced.jpg"), final_enhanced)

    print(f"\nDone. All results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
