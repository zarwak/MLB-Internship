# Day 21 - Computer Vision Image Processing Studio

This is my Day 21 work. After several days learning individual OpenCV
techniques one at a time (blur, edges, contours, shapes, rotation...),
today was about combining all of them into one real, deployable
application - and understanding *why* a Streamlit/Gradio app is put
together the way it is, not just copy-pasting one that runs.

## What's in this folder

- **coding_practice/** - the literal first pass at the assignment: a
  Streamlit interface with upload -> dropdown of 7 operations -> process
  -> display -> download/save. Deliberately bare, no extra parameters, so
  the core pattern (a dictionary of `image in -> image out` functions
  driving the dropdown) is easy to see without any UI polish in the way.
- **cv_image_studio/** - the mini project *and* the Challenge Task
  combined into one app. Per-operation parameter sliders, a two-column
  original/processed view, sample images built in, and both a download
  button and a save-to-disk button, PLUS everything from the Challenge
  Task (see below) - the 3 extra operations and the Pipeline mode. Two
  modes in the sidebar: **Single Operation** and **Chain Multiple
  Filters (Pipeline)**.

Each folder has its own `HOW_TO_RUN.txt` with step-by-step instructions,
and a `make_sample_images.py` that generates synthetic test images so
neither app needs a real photo to try out.

## Challenge Task - what was implemented (inside cv_image_studio)

The Challenge Task isn't a separate folder - it's built into
`cv_image_studio`, since everything it needed (the 7 base operations, the
same slider-building pattern) was already there. What it adds:

- **Brightness & Contrast Adjustment** - `contrast * pixel + brightness`,
  clipped to 0-255. Useful because a lot of real photos are too
  dark/washed out for the other operations (blur, edges, contours) to
  work well on - fixing exposure first is a normal step in any real
  editing pipeline.
- **Flip** - pure array reversal along an axis, no math involved. Small,
  but genuinely useful (mirroring, correcting a sideways photo).
- **Threshold** - turns a photo into pure black/white, with 3 methods:
  Binary (fixed cutoff), Binary Inverted, and Otsu (auto-picks the cutoff
  from the image's own histogram - same idea used inside Day19's shape
  detector).
- **Pipeline mode** (the "make it more useful" part) - pick several
  operations in the sidebar and they run in sequence, each one's output
  feeding the next one's input. Real photo editing is never just one
  filter (adjust exposure, then sharpen, then maybe threshold), and
  because every operation shares the same `image in -> image out` shape,
  chaining them needed no special-case code.

Switch to it via the "Mode" radio button in `cv_image_studio`'s sidebar.

## The core idea behind both apps: a dispatch dictionary

Every operation - grayscale, blur, edges, whatever - is written as a
plain function with the exact same shape: takes an image in, returns an
image out, and does nothing else (no printing, no UI code, nothing
global). That means every operation can live as one entry in a
dictionary:

```python
OPERATIONS = {
    "Grayscale": apply_grayscale,
    "Blur": apply_blur,
    ...
}
```

The dropdown just picks a key, and the app calls `OPERATIONS[choice](image)`.
No `if operation == "Grayscale": ... elif operation == "Blur": ...` chain
that grows every time a new operation is added. This is the single
biggest difference between "code that works" and "code you can keep
adding to without it turning into a mess" - and it's also *why* chaining
operations together in `cv_image_studio`'s Pipeline mode was almost free
to build: since every function has the same shape, feeding one function's
output into the next function's input just works.

## Streamlit's rerun model (the thing that actually trips people up)

A Streamlit script reruns from top to bottom on *every* interaction -
click a button, move a slider, pick a dropdown option, the whole file
executes again like you just hit "Run" fresh. It is not an app that sits
waiting for one specific callback to fire. This is why:

- moving a slider re-reads everything above it in the script too
- anything that needs to survive across a rerun (like "the image the user
  already uploaded") has to be explicit, either by re-reading it from the
  same widget each time (what these apps do) or storing it in
  `st.session_state`

Once this clicked, weird "why did my image reset when I touched an
unrelated slider" bugs stopped being mysterious.

## The 7 core operations, conceptually

| Operation | The actual idea |
|---|---|
| Grayscale | Collapse 3 color channels into 1 brightness value - most of the operations below only care about brightness/structure, not color. |
| Blur | Slide a small kernel over the image and replace each pixel with a weighted average of its neighbors (convolution - the same core op every CNN uses). |
| Edge Detection (Canny) | Find pixels where brightness changes sharply; two thresholds (low/high) separate real edges from noise. |
| Rotation | Build a 2x3 rotation matrix around a center point and angle, then remap every pixel through it - literally the rotation math from geometry class. |
| Image Enhancement (Sharpen) | Unsharp masking: `original + amount * (original - blurred)` - subtract a blurred copy to emphasize edges. An old darkroom trick, still used today. |
| Contour Detection | After thresholding to black/white, trace the *continuous outline* of each connected white blob. |
| Shape Detection | Take a contour, simplify it to a polygon (`approxPolyDP`), then classify by corner count: 3 = triangle, 4 = square/rectangle (by aspect ratio), more + round = circle. |

Plus, in `cv_image_studio`'s Challenge Task features: **Brightness &
Contrast** (`contrast * pixel + brightness`, clipped to 0-255), **Flip**
(pure array reversal, no math), and **Threshold** (the simplest possible
segmentation - cut the image into pure black/white at some brightness
value, optionally auto-picked via Otsu's method).

## What a Hugging Face Space actually is

A Space is just a git repository that Hugging Face turns into a running
container. You push files the same way you'd push to GitHub; a YAML
header at the top of `README.md` tells HF how to run the app
(`sdk: streamlit`, `app_file: app.py`); HF installs `requirements.txt`
and runs it. Updating later is just another `git push` - no dashboard
clicking. Not deployed yet - that's a separate step for when I'm ready to
do it under my own HF account.

## Deliverables status

- [x] Source code (2 self-contained folders: `coding_practice/`,
      `cv_image_studio/`)
- [x] `app.py` in each folder
- [x] `requirements.txt` in each folder
- [x] `README.md` (this file)
- [x] Sample input images (generated, `sample_input_images/` /
      `sample_images/`)
- [ ] GitHub repository link - _TODO after pushing_
- [ ] Hugging Face Space link - _TODO after deploying_
- [ ] Screen recording - _TODO_
