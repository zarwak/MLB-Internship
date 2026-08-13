"""
Real-Time Video Processing Tool - the core pipeline.
Order: grayscale -> Gaussian blur -> Canny edge detection. Same three
techniques from Day18/Day19, applied to ONE frame at a time - "video
processing" is really just this pipeline run in a loop, once per frame.

Also holds the small video-I/O helpers (reading properties, opening a
writer, building a side-by-side preview) that every script in this folder
needs, so process_video.py, webcam_live.py, app.py, and the Challenge Task
can all reuse the same code instead of copy-pasting it.
"""
import cv2
import numpy as np

CANNY_LOW = 50
CANNY_HIGH = 150
BLUR_KSIZE = (5, 5)


def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def blur(gray, ksize=BLUR_KSIZE):
    """Smooths out camera/compression noise BEFORE edge detection, so Canny
    reacts to real object outlines instead of tiny random pixel jitter."""
    return cv2.GaussianBlur(gray, ksize, 0)


def detect_edges(blurred, low=CANNY_LOW, high=CANNY_HIGH):
    return cv2.Canny(blurred, low, high)


def resize_for_processing(frame, target_width=None):
    """Optional speed/quality trade-off: a smaller frame means grayscale,
    blur, and Canny all have fewer pixels to touch, so processing is faster
    but the result is coarser. None (the default) leaves the frame at its
    original size - only the interactive app exposes this as a slider,
    since the local scripts are fine processing at full resolution."""
    if not target_width or frame.shape[1] <= target_width:
        return frame
    scale = target_width / frame.shape[1]
    return cv2.resize(frame, (target_width, int(frame.shape[0] * scale)))


def process_frame(frame, blur_ksize=BLUR_KSIZE, canny_low=CANNY_LOW, canny_high=CANNY_HIGH, resize_width=None):
    """Runs the full pipeline on one BGR frame. blur_ksize/canny_low/
    canny_high/resize_width all default to the same fixed values the local
    scripts have always used, so passing nothing keeps old behavior exactly
    - the Streamlit app is the only caller that overrides them, via sliders.

    Returns every intermediate step - each converted back to 3-channel BGR
    so it can go straight into a preview window or a standard video file,
    which both expect 3 channels - plus the raw single-channel edge map, so
    callers (process_video.py, webcam_live.py, app.py, the Challenge Task)
    can show or save any stage without re-running the pipeline."""
    working = resize_for_processing(frame, resize_width)

    gray = to_grayscale(working)
    blurred = blur(gray, blur_ksize)
    edges = detect_edges(blurred, canny_low, canny_high)

    return {
        "original": frame,       # untouched, full resolution - for display/comparison
        "resized": working,      # what actually got processed (== original if no resize_width)
        "gray_bgr": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "blurred_bgr": cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
        "edges": edges,
        "edges_bgr": cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
    }


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
    # the preview window; it never touches the frame that gets processed or
    # written to the output video.
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))


def side_by_side(left, right):
    # for live preview only: puts two same-size BGR frames next to each
    # other with a thin divider, so "original vs processed" is one window.
    # Each half is capped at max_width/2 BEFORE combining, so the combined
    # window (not just each half) stays a reasonable on-screen size.
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    left = resize_for_display(left, max_width=500)
    right = resize_for_display(right, max_width=500)
    if right.shape[:2] != left.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    divider = np.full((left.shape[0], 4, 3), 128, dtype=np.uint8)
    return np.hstack([left, divider, right])
