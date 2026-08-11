"""
Challenge Task: process 10 document images through the real boundary
detection pipeline (reused from the "document_boundary_tool" folder, not
copy-pasted).

For EACH image, saves 4 versions into results/:
  <name>_1_original.jpg
  <name>_2_edge_detection.jpg       (raw Canny edges, before cleanup)
  <name>_3_morphological.jpg        (edges after morphological closing)
  <name>_4_boundary_detected.jpg    (final image with drawn boundary)

Also writes results_summary.md - a table noting whether a clean 4-corner
boundary was found for each image, which you can edit by hand afterward.
"""
import os
import sys
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "document_boundary_tool", "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")
NUM_IMAGES = 10

# the actual pipeline lives in the other folder - point Python to it instead
# of duplicating the code here
TOOL_DIR = os.path.join(BASE_DIR, "..", "document_boundary_tool")
sys.path.insert(0, TOOL_DIR)
from detector import detect_boundary  # noqa: E402


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXT))
    if not files:
        print(f"No images found in {INPUT_DIR}")
        print("Add your document photos to document_boundary_tool/input_images/ first.")
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
        result = detect_boundary(original)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_1_original.jpg"), original)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_2_edge_detection.jpg"), result["edges"])
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_3_morphological.jpg"), result["cleaned_edges"])
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_4_boundary_detected.jpg"), result["annotated"])

        if result["found_four_corners"]:
            outcome = "Clean 4-corner boundary detected"
        elif result["contour_points"] is not None:
            outcome = "Approximate boundary only (fallback bounding box)"
        else:
            outcome = "No boundary detected"
        summary_rows.append((filename, outcome))

    summary_path = os.path.join(OUTPUT_DIR, "results_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Challenge Task Results\n\n")
        f.write("Auto-filled by run_challenge.py. Feel free to edit the Notes column by hand.\n\n")
        f.write("| Image | Expected Outcome / Notes |\n")
        f.write("|-------|---------------------------|\n")
        for filename, outcome in summary_rows:
            f.write(f"| {filename} | {outcome} |\n")

    print(f"\nDone. All results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
