"""
READING a video - the basics. A video file isn't one big object OpenCV loads
into memory - cv2.VideoCapture opens a pointer into the file, and every
.read() call hands back ONE frame (a normal BGR image, same as
cv2.imread would give you) plus a True/False flag saying whether a frame was
actually grabbed. False means "no more frames" - that's how the video ends.

So "processing a video" is really just: loop, read one frame, do something
with it, repeat until read() says False.
"""
import cv2
from utils import get_sample_video, print_video_properties, resize_for_display

path = get_sample_video()
print(f"Reading: {path}")

cap = cv2.VideoCapture(path)
if not cap.isOpened():
    raise SystemExit(f"Could not open video: {path}")

print_video_properties(cap)

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:  # ret=False means the video ran out of frames
        break
    frame_num += 1

    cv2.putText(frame, f"Frame {frame_num}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.namedWindow("Frame by frame (press q to quit early)", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame by frame (press q to quit early)", resize_for_display(frame))

    # waitKey(30) both draws the window AND pauses ~30ms so frames don't
    # fly by instantly - it also lets us check for a 'q' keypress to bail out
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

print(f"Displayed {frame_num} frame(s) total")

# always release the file handle and close windows - skip this and the file
# can stay "locked" or the window can hang around after the script ends
cap.release()
cv2.destroyAllWindows()
