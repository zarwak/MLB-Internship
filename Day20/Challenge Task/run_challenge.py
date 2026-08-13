"""
Challenge Task: process 3 different videos through the real pipeline
(reused from the "video_processing_tool" folder, not copy-pasted).

For EACH video, saves 2 versions into results/:
  <name>_original.mp4
  <name>_processed.mp4      (grayscale -> blur -> Canny)

Also writes results_summary.md - one row per video with its properties,
plus an Observations column you fill in by hand after watching both
versions. That comparison is genuinely qualitative - the script can report
FPS/size/frame count, but only a person watching both clips can describe
what actually changed.
"""
import os
import sys
import shutil
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.join(BASE_DIR, "..", "video_processing_tool")
INPUT_DIR = os.path.join(TOOL_DIR, "input_videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
VALID_EXT = (".mp4", ".avi", ".mov", ".mkv")
NUM_VIDEOS = 3

# the actual pipeline lives in the other folder - point Python to it instead
# of duplicating the code here
sys.path.insert(0, TOOL_DIR)
from processor import process_frame, print_video_properties, open_writer  # noqa: E402


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXT))
    if not files:
        print(f"No videos found in {INPUT_DIR}")
        print("Add your video files to video_processing_tool/input_videos/ first.")
        return

    if len(files) < NUM_VIDEOS:
        print(f"Note: only found {len(files)} video(s) - the challenge task wants {NUM_VIDEOS}.")
    files = files[:NUM_VIDEOS]

    summary_rows = []
    for filename in files:
        name = os.path.splitext(filename)[0]
        in_path = os.path.join(INPUT_DIR, filename)
        print(f"Processing: {filename}")

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            print(f"  Skipped (could not open): {filename}")
            continue
        props = print_video_properties(cap, label=filename)

        original_out = os.path.join(OUTPUT_DIR, f"{name}_original.mp4")
        processed_out = os.path.join(OUTPUT_DIR, f"{name}_processed.mp4")
        shutil.copyfile(in_path, original_out)
        writer = open_writer(processed_out, props["fps"], props["width"], props["height"])

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            result = process_frame(frame)
            writer.write(result["edges_bgr"])

        cap.release()
        writer.release()
        print(f"  Processed {frame_num} frame(s)")

        summary_rows.append({
            "filename": filename,
            "fps": props["fps"],
            "size": f"{props['width']}x{props['height']}",
            "frames": frame_num,
        })

    summary_path = os.path.join(OUTPUT_DIR, "results_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Challenge Task Results\n\n")
        f.write("Properties auto-filled by run_challenge.py. Fill in the Observations\n")
        f.write("column by hand after watching each `_original.mp4` / `_processed.mp4` pair.\n\n")
        f.write("| Video | FPS | Size | Frames | Observations |\n")
        f.write("|-------|-----|------|--------|---------------|\n")
        for row in summary_rows:
            f.write(f"| {row['filename']} | {row['fps']:.1f} | {row['size']} | {row['frames']} | _TODO_ |\n")

    print(f"\nDone. All results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
