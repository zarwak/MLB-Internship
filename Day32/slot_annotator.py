"""
Day 32 - interactive, one-time manual parking-slot annotator.

`generate_grid_layout()` (see parking_detection.py) is fast but only lines
up with real painted spaces by coincidence - real rows are rarely
perfectly equal-width, rarely start on a frame-fraction boundary, and
camera tilt throws an equal grid off further from one edge of the lot to
the other (see README "Real challenges faced" for two concrete cases this
bit us on). This tool is the reliable alternative: open a video's first
frame, drag a box over every real parking space by eye, and save the
result as a normal ParkingSpace layout JSON - the same file format
`load_layout()` reads everywhere else in this project (fractional 0-1
coordinates, so it's resolution-independent and reusable for any video
from the same camera angle).

This is a local, one-time offline tool - not part of the deployed Streamlit
app - so it needs a GUI-capable OpenCV build. requirements.txt intentionally
pins opencv-python-headless (correct for the app's GUI-less server host,
where cv2.imshow would fail anyway), so run this tool from an environment
that also has the regular package installed:
    pip install opencv-python
(opencv-python and opencv-python-headless both install into the same `cv2`
package path and can't coexist - installing this one after
requirements.txt replaces headless with the GUI build in that environment;
reinstall opencv-python-headless afterward if you need to run the Streamlit
app again from the same environment).

Usage:
    python slot_annotator.py sample_videos/your_video.mp4 sample_videos/your_video_spaces.json

Controls:
    left-drag   draw a box over one parking space, release to commit it
    u           undo the last box
    r           clear all boxes
    s           save all boxes to the output JSON and exit
    q / Esc     quit without saving

Only needs to be done once per camera angle - the saved JSON is reused for
every future video from that same camera, and only needs redoing if the
camera moves or the physical layout changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from parking_detection import ParkingSpace, save_layout

DISPLAY_MAX_SIDE = 1280  # cap the annotation window so a large frame still fits on screen
BOX_COLOR = (0, 255, 0)
DRAG_COLOR = (0, 165, 255)
TEXT_COLOR = (0, 255, 0)


def _extract_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"could not read a frame from: {video_path}")
    return frame


def _fit_to_screen(frame, max_side: int = DISPLAY_MAX_SIDE):
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return frame.copy(), 1.0
    display = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return display, scale


class _Annotator:
    def __init__(self, base_display):
        self.base_display = base_display
        self.boxes: list[tuple[int, int, int, int]] = []  # in DISPLAY pixel coords, x1,y1,x2,y2
        self.dragging = False
        self.drag_start = (0, 0)
        self.drag_end = (0, 0)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x1, y1 = self.drag_start
            x2, y2 = x, y
            if abs(x2 - x1) >= 5 and abs(y2 - y1) >= 5:
                self.boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    def render(self):
        frame = self.base_display.copy()
        for idx, (x1, y1, x2, y2) in enumerate(self.boxes, start=1):
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.putText(frame, str(idx), (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
        if self.dragging:
            cv2.rectangle(frame, self.drag_start, self.drag_end, DRAG_COLOR, 2)
        cv2.putText(frame, f"spaces: {len(self.boxes)}  |  drag=new box  u=undo  r=clear  s=save  q=quit",
                    (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame


def annotate(video_path: str, output_path: str) -> None:
    frame = _extract_first_frame(video_path)
    orig_h, orig_w = frame.shape[:2]
    display, scale = _fit_to_screen(frame)

    ann = _Annotator(display)
    window = "Slot annotator - drag a box per space, s to save, q to quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, ann.on_mouse)

    saved = False
    while True:
        cv2.imshow(window, ann.render())
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u"):
            if ann.boxes:
                ann.boxes.pop()
        elif key == ord("r"):
            ann.boxes.clear()
        elif key == ord("s"):
            saved = True
            break
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()

    if not saved:
        print("Quit without saving - nothing written.")
        return
    if not ann.boxes:
        print("No spaces drawn - nothing written.")
        return

    layout = []
    for idx, (x1, y1, x2, y2) in enumerate(ann.boxes, start=1):
        # display pixels -> original-resolution pixels -> fraction of frame (0-1),
        # the same resolution-independent convention every other layout in this
        # project uses (see ParkingSpace docstring in parking_detection.py).
        polygon = [
            (x1 / scale / orig_w, y1 / scale / orig_h),
            (x2 / scale / orig_w, y1 / scale / orig_h),
            (x2 / scale / orig_w, y2 / scale / orig_h),
            (x1 / scale / orig_w, y2 / scale / orig_h),
        ]
        layout.append(ParkingSpace(id=idx, polygon=polygon))

    save_layout(output_path, layout)
    print(f"Saved {len(layout)} spaces to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="path to a video from the camera angle you're calibrating")
    parser.add_argument("output_json", help="where to save the resulting layout JSON")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    annotate(args.video, args.output_json)


if __name__ == "__main__":
    main()
