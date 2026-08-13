"""
Runs the video processing pipeline (from processor.py, not copy-pasted) on
EVERY video in input_videos/. For each one: shows a live "Original |
Processed" preview window while it works, and saves the final Canny edge
result as a new video into outputs/.
"""
import os
import cv2

from processor import process_frame, print_video_properties, open_writer, side_by_side

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VALID_EXT = (".mp4", ".avi", ".mov", ".mkv")


def process_one(path, out_path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  Could not open: {path}")
        return

    props = print_video_properties(cap, label=os.path.basename(path))
    writer = open_writer(out_path, props["fps"], props["width"], props["height"])

    # waitKey's delay is in milliseconds - deriving it from the source's own
    # FPS makes the live preview play back at roughly its real speed instead
    # of racing through frames or crawling
    delay = max(1, int(1000 / props["fps"])) if props["fps"] > 0 else 30

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:  # ret=False means the video ran out of frames
            break
        frame_num += 1

        result = process_frame(frame)
        writer.write(result["edges_bgr"])

        preview = side_by_side(result["original"], result["edges_bgr"])
        cv2.namedWindow("Original | Processed (press q to quit early)", cv2.WINDOW_NORMAL)
        cv2.imshow("Original | Processed (press q to quit early)", preview)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"  Processed {frame_num} frame(s) -> {out_path}")


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXT))
    if not files:
        print(f"No videos found in {INPUT_DIR}")
        print("Add your video file(s) there (mp4, avi, mov, or mkv) and run again.")
        return

    for filename in files:
        in_path = os.path.join(INPUT_DIR, filename)
        name = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{name}_processed.mp4")
        print(f"Processing: {filename}")
        process_one(in_path, out_path)

    print(f"\nDone. Processed video(s) saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
