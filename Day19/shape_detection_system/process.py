"""
Runs the shape detector on every image in input_images/ and saves EVERY step
of the pipeline into output_images/ (one subfolder per image), not just the
final result:
  1_original.jpg
  2_grayscale.jpg
  3_blurred.jpg
  4_threshold.jpg
  5_all_contours.jpg
  6_labeled_shapes.jpg
"""
import os
import cv2

from detector import detect_shapes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXT))
    if not files:
        print(f"No images found in {INPUT_DIR}")
        print("Add your shape photos there (jpg, jpeg, png, or bmp) and run again.")
        return

    for filename in files:
        in_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(in_path)
        if img is None:
            print(f"Skipped (could not read): {filename}")
            continue

        result = detect_shapes(img)
        names = ", ".join(s["label"] for s in result["shapes"]) or "none"
        print(f"{filename}: {len(result['shapes'])} shape(s) - {names}")

        name = os.path.splitext(filename)[0]
        step_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(step_dir, exist_ok=True)

        cv2.imwrite(os.path.join(step_dir, "1_original.jpg"), img)
        cv2.imwrite(os.path.join(step_dir, "2_grayscale.jpg"), result["gray"])
        cv2.imwrite(os.path.join(step_dir, "3_blurred.jpg"), result["blurred"])
        cv2.imwrite(os.path.join(step_dir, "4_threshold.jpg"), result["thresh"])
        cv2.imwrite(os.path.join(step_dir, "5_all_contours.jpg"), result["contours_drawn"])
        cv2.imwrite(os.path.join(step_dir, "6_labeled_shapes.jpg"), result["annotated"])

    print(f"\nDone. Each image's steps saved in its own subfolder under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
