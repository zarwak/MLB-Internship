"""
Real-time webcam version of the same pipeline (processor.py) used by
process_video.py - identical grayscale -> blur -> Canny steps, just fed by
the webcam instead of a video file. This is what makes it "real-time":
there's no file to finish reading - it keeps reading, processing, and
showing new frames forever until you press 'q'.
"""
import os
import cv2

from processor import process_frame, open_writer, side_by_side

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUT_PATH = os.path.join(OUTPUT_DIR, "webcam_processed.mp4")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0). Is another app using it, or no camera present?")

    # grab one frame first so we know the real width/height before opening
    # the writer - VideoWriter needs an exact frame size declared up front
    ret, frame = cap.read()
    if not ret:
        raise SystemExit("Webcam opened but returned no frame.")
    height, width = frame.shape[:2]
    writer = open_writer(OUT_PATH, cap.get(cv2.CAP_PROP_FPS), width, height)

    print("Webcam live - press 'q' to quit.")
    frame_num = 0
    while ret:
        frame_num += 1
        result = process_frame(frame)
        writer.write(result["edges_bgr"])

        preview = side_by_side(result["original"], result["edges_bgr"])
        cv2.namedWindow("Webcam: Original | Processed (press q to quit)", cv2.WINDOW_NORMAL)
        cv2.imshow("Webcam: Original | Processed (press q to quit)", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        ret, frame = cap.read()

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved {frame_num} frame(s) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
