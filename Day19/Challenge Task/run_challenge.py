"""
Challenge Task: process 10 shape images through the real shape detection
pipeline (reused from the "shape_detection_system" folder, not copy-pasted).

For EACH image, saves 3 versions into results/:
  <name>_1_original.jpg
  <name>_2_contour_detection.jpg    (all contours found, before classification)
  <name>_3_final_shapes.jpg         (final image with labeled shapes)

Also writes results_summary.md - a table listing the shapes found (and
their area/perimeter) for each image, which you can edit by hand afterward.
"""
import os
import sys
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "shape_detection_system", "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")
NUM_IMAGES = 10

# the actual pipeline lives in the other folder - point Python to it instead
# of duplicating the code here
TOOL_DIR = os.path.join(BASE_DIR, "..", "shape_detection_system")
sys.path.insert(0, TOOL_DIR)
from detector import detect_shapes  # noqa: E402


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXT))
    if not files:
        print(f"No images found in {INPUT_DIR}")
        print("Add your shape photos to shape_detection_system/input_images/ first.")
        return

    if len(files) < NUM_IMAGES:
        print(f"Note: only found {len(files)} image(s) - the challenge task wants {NUM_IMAGES}.")
    files = files[:NUM_IMAGES]

    summary_rows = []
    for filename in files:
        name = os.path.splitext(filename)[0]
        in_path = os.path.join(INPUT_DIR, filename)
        original = cv2.imread(in_path)
        if original is None:
            print(f"Skipped (could not read): {filename}")
            continue

        print(f"Processing: {filename}")
        result = detect_shapes(original)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_1_original.jpg"), original)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_2_contour_detection.jpg"), result["contours_drawn"])
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_3_final_shapes.jpg"), result["annotated"])

        if result["shapes"]:
            details = "; ".join(
                f"{s['label']} (A={s['area']:.0f}, P={s['perimeter']:.0f})" for s in result["shapes"]
            )
        else:
            details = "No shapes detected"
        summary_rows.append((filename, len(result["shapes"]), details))

    summary_path = os.path.join(OUTPUT_DIR, "results_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Challenge Task Results\n\n")
        f.write("Auto-filled by run_challenge.py. Feel free to edit by hand afterward.\n\n")
        f.write("| Image | Shapes Found | Details |\n")
        f.write("|-------|--------------|---------|\n")
        for filename, count, details in summary_rows:
            f.write(f"| {filename} | {count} | {details} |\n")

    print(f"\nDone. All results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
