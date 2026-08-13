"""
Helper functions shared by all Day 20 practice scripts.
We don't have a real video dropped in yet (that goes in
video_processing_tool/input_videos/), so like Day17/Day18/Day19 generated a
fake sample image, this generates a fake sample VIDEO - a ball bouncing
around a canvas for a few seconds - so I could practice reading/writing
video frame-by-frame before touching real footage.
"""
import os
import cv2
import numpy as np

# folder this file lives in, so videos/outputs always save HERE
# no matter which folder you were in when you ran "python 01_read_video_basics.py"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

CANVAS_W, CANVAS_H = 640, 480
SAMPLE_FPS = 24
SAMPLE_SECONDS = 5

VALID_EXT = (".mp4", ".avi", ".mov", ".mkv")


def _ensure_folders():
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _create_sample_video(path):
    # a bouncing ball + a growing/shrinking rectangle - two moving things so
    # there's always something with a clear edge for Canny to find later
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, SAMPLE_FPS, (CANVAS_W, CANVAS_H))

    ball_pos = np.array([80, 80], dtype=float)
    ball_vel = np.array([6, 4], dtype=float)
    radius = 30

    total_frames = SAMPLE_FPS * SAMPLE_SECONDS
    for i in range(total_frames):
        frame = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)  # white background

        # bounce off the walls by flipping velocity when we'd cross an edge
        ball_pos += ball_vel
        for axis, limit in enumerate((CANVAS_W, CANVAS_H)):
            if ball_pos[axis] - radius < 0 or ball_pos[axis] + radius > limit:
                ball_vel[axis] *= -1
        cv2.circle(frame, tuple(ball_pos.astype(int)), radius, (30, 30, 200), -1)

        # a rectangle that pulses in size, independent of the ball, so a
        # frame always has more than one shape/edge to process
        pulse = 40 + int(30 * abs(np.sin(i / 15)))
        cx, cy = CANVAS_W - 150, CANVAS_H - 120
        cv2.rectangle(frame, (cx - pulse, cy - pulse), (cx + pulse, cy + pulse), (30, 150, 30), -1)

        cv2.putText(frame, f"frame {i+1}/{total_frames}", (20, CANVAS_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        writer.write(frame)

    writer.release()


def get_sample_video():
    # uses whatever real video you've dropped into videos/ - only falls back
    # to generating the fake bouncing-shapes video if that folder is empty
    _ensure_folders()
    existing = sorted(f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(VALID_EXT))
    if existing:
        return os.path.join(VIDEOS_DIR, existing[0])

    path = os.path.join(VIDEOS_DIR, "sample_bouncing_shapes.mp4")
    _create_sample_video(path)
    return path


def print_video_properties(cap, label="Video"):
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"--- {label} properties ---")
    print(f"FPS: {fps:.2f}")
    print(f"Width x Height: {width} x {height}")
    print(f"Total frames: {frame_count}")
    if fps > 0 and frame_count > 0:
        print(f"Duration: {frame_count / fps:.1f} sec")
    return {"fps": fps, "width": width, "height": height, "frame_count": frame_count}


def open_writer(path, fps, width, height):
    # some sources (a live webcam especially) report fps as 0 - a video file
    # can't be written with 0 fps, so fall back to a sane default instead
    if not fps or fps <= 1:
        fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def resize_for_display(frame, max_width=800):
    # cv2.imshow shows a frame at its exact native resolution by default -
    # for a high-res video/webcam that makes the window balloon to fill most
    # of the screen ("too zoomed in"). This scales DOWN ONLY (never up) for
    # the preview window; it never touches the frame you process or save.
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))


def side_by_side(left, right):
    # for live preview only: puts two same-size BGR frames next to each
    # other with a thin divider, so "original vs processed" is one window
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    divider = np.full((left.shape[0], 4, 3), 128, dtype=np.uint8)
    return np.hstack([left, divider, right])
