"""
Capturing a WEBCAM instead of a video file - the only difference is what you
pass to cv2.VideoCapture(). A file path opens that file; the integer 0 opens
the first connected camera instead. Everything else (the read-loop, frame,
waitKey) works exactly the same, because OpenCV treats both as the same kind
of "video source" - that's why this script barely differs from
01_read_video_basics.py.

A live webcam has no natural "end" like a file does (ret never becomes False
on its own once it's running), so 'q' is the only way this loop stops.
"""
import cv2
from utils import resize_for_display

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam (index 0). Is another app using it, or no camera present?")

print("Webcam opened - press 'q' to quit.")
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam stopped sending frames.")
        break
    frame_num += 1

    cv2.putText(frame, f"Frame {frame_num}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.namedWindow("Webcam (press q to quit)", cv2.WINDOW_NORMAL)
    cv2.imshow("Webcam (press q to quit)", resize_for_display(frame))

    # webcams keep sending frames indefinitely - waitKey(1) here (not 30 like
    # the file version) so the preview feels responsive/real-time instead of
    # artificially slowed down
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(f"Captured {frame_num} frame(s) total")
cap.release()
cv2.destroyAllWindows()
