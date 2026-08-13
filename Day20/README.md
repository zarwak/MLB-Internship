# Day 20 - Video Processing with OpenCV

This is my Day 20 work. After several days processing single images
(transformations, edges, contours), today was about applying the same kind
of pipeline to VIDEO - which turned out to mostly be "the same image
processing, just run in a loop, once per frame."

## What's in this folder

- **coding_practice/** - small practice scripts, one file per topic:
  reading a video frame by frame, printing its properties, applying
  grayscale + Canny and saving the result, and capturing a live webcam.
  A shared `utils.py` generates a synthetic sample video (a bouncing ball +
  pulsing rectangle) to practice on before touching real footage.
- **video_processing_tool/** - the real mini project. Point it at a video
  (or your webcam) and it grayscales, blurs, and Canny-edge-detects every
  frame, shows an original-vs-processed live preview, and saves the
  processed result as a new video file. It also has a Streamlit web app so
  anyone can try it without touching code.
- **Challenge Task/** - runs the same tool on 3 different videos, saving
  the original and processed version of each, plus a summary table.

Each folder has its own `HOW_TO_RUN.txt` with step by step instructions.

## How OpenCV reads a video

A video file isn't one big object OpenCV loads into memory all at once -
`cv2.VideoCapture` opens a pointer into the file, and every `.read()` call
hands back exactly ONE frame (a normal BGR image, same as `cv2.imread`
would give you) plus a `True`/`False` flag saying whether a frame was
actually grabbed. `False` means "no more frames" - that's how you detect
the video has ended. So "processing a video" is really just: loop, read one
frame, do something with it, repeat until `.read()` says `False`.

A webcam uses the exact same `VideoCapture`/`.read()` interface - the only
difference is you pass the camera index (`0`) instead of a file path. The
loop never runs out on its own, though, since there's no "end of file" for
a live camera - you stop it yourself (pressing 'q').

## What FPS means

FPS (Frames Per Second) is how many still frames are shown per second to
create the illusion of smooth motion - a typical video is 24-30 FPS. It
matters in three places here:

- **Reading**: `cap.get(cv2.CAP_PROP_FPS)` tells you the video's intended
  playback speed.
- **Live preview**: `cv2.waitKey(delay)` controls how long each frame stays
  on screen. Deriving `delay` from the source's own FPS (`1000 / fps`,
  since `waitKey` takes milliseconds) makes the preview play at roughly
  real speed instead of racing through frames or crawling.
- **Saving**: `cv2.VideoWriter` needs an FPS value declared up front. Get
  it wrong and the output video's picture is fine but plays too fast or
  too slow. Webcams often report `0` FPS before frames start flowing, so
  the writer falls back to a sane default (20) when that happens.

## Processing techniques applied, in order

1. **Grayscale** (`cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`) - edge
   detection only cares about brightness changes, not color, so dropping
   color channels is both correct and 3x less data to process per frame.
2. **Gaussian Blur** (`cv2.GaussianBlur`) - smooths out camera/compression
   noise BEFORE edge detection. Without this, Canny reacts to random pixel
   jitter as if it were real edges, producing a noisy, speckled result.
3. **Canny Edge Detection** (`cv2.Canny`) - finds sharp brightness changes
   using two thresholds: anything above the high threshold is kept as a
   definite edge, anything below the low one is discarded, and anything in
   between is kept only if it connects to a definite edge. That "keep it if
   it's connected" rule is what keeps Canny's lines clean and continuous
   instead of broken and speckled.

The processed video that gets saved is the final Canny edge result; the
live preview window shows original and processed side by side so you can
compare them while it runs.

## Real-world uses of video processing

- **Surveillance/security** - motion and edge detection to flag activity
  in a camera feed without a person watching 24/7.
- **Traffic monitoring** - detecting and counting vehicles or lane markings
  frame by frame from a road camera.
- **Object tracking** - the same "process every frame" loop, but tracking
  where a detected object moves between frames.
- **Quality inspection on a production line** - a camera watching parts go
  by, checking each frame against expected shape/edges.
- **Video calls/AR filters** - real-time per-frame processing (background
  blur, face filters) is the exact same read-process-show loop used here.

## Challenges/blockers I faced

- **`cv2.VideoWriter` is strict.** Unlike saving a single image, you must
  declare the exact FPS, frame size, and codec (`fourcc`) before writing
  the first frame, and every frame written afterward must match that size
  exactly - a mismatch produces a silently broken or empty file rather than
  a clear error. Reading the source's real width/height with
  `cap.get(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT)` and reusing those exact values
  for the writer avoids this.
- **Webcams report FPS as 0** until frames actually start flowing, so
  `open_writer` in `processor.py` falls back to a default FPS (20) instead
  of trying to create a writer with an invalid 0 FPS.
- **A deployed web app can't open a real `cv2.imshow()` window or a
  continuous webcam stream** - those only work with a display attached to
  the machine running the script, which a cloud server doesn't have. The
  Streamlit app works around this two ways: for uploaded videos, it updates
  one image element in place every so often (not every frame - a full
  browser round-trip per frame makes the app crawl) instead of a live
  window; for webcam, it uses `st.camera_input`, which takes one snapshot
  at a time through the browser instead of a persistent live feed.
- **Grayscale/edge frames are single-channel, but video files expect 3.**
  `cv2.Canny` outputs a black/white single-channel image, but
  `cv2.VideoWriter` (and a standard video file) expects 3 color channels
  per frame. Converting back with `cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)`
  keeps the frame looking grayscale to the eye while satisfying that format
  requirement.

## Links

- Streamlit app: https://videoprocessingtool.streamlit.app/
- GitHub repo: https://github.com/zarwak/MLB-Internship/tree/main/Day20

