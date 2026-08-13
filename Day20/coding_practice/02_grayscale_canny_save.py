"""
Per-frame processing + SAVING a video. Grayscale and Canny work on a video
exactly like they work on a single image (Day18) - the only new part is
doing it inside the read-loop, and writing each result out with
cv2.VideoWriter instead of cv2.imwrite.

cv2.VideoWriter is stricter than saving a single image: you must declare the
fps, frame size, and codec (fourcc) BEFORE writing the first frame, and every
frame you write after that must match that exact size - or the file comes
out broken/empty.
"""
import os
import cv2
from utils import get_sample_video, print_video_properties, open_writer, resize_for_display, OUTPUTS_DIR

path = get_sample_video()
cap = cv2.VideoCapture(path)
props = print_video_properties(cap)

out_path = os.path.join(OUTPUTS_DIR, "02_grayscale_canny.mp4")
# edges are single-channel (black/white) but a standard video file expects
# 3 color channels per frame - cv2.cvtColor back to BGR keeps the frame
# grayscale-looking to the eye while satisfying that format requirement.
writer = open_writer(out_path, props["fps"], props["width"], props["height"])

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    writer.write(edges_bgr)  # full resolution - only the PREVIEW below is scaled down
    cv2.namedWindow("Grayscale + Canny (press q to quit early)", cv2.WINDOW_NORMAL)
    cv2.imshow("Grayscale + Canny (press q to quit early)", resize_for_display(edges_bgr))
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Processed {frame_num} frame(s)")
print(f"Saved: {out_path}")
