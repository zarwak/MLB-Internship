# Day 18 - Edge Detection & Morphological Operations

This is my Day 18 work. After two days of fixing up images (transformations,
enhancement), today was about finding SHAPES in them - specifically, finding
a document's edges and cleaning that edge map up enough to actually locate
the page boundary and draw a box around it.

## What's in this folder

- **coding_practice/** - small practice scripts, one file per topic. Each
  one runs against a fake sample document (generated in code, no real photo
  needed) so I could learn each technique in isolation before combining them.
- **document_boundary_tool/** - the real mini project. You give it a photo
  of a document and it finds the page's edges and draws a box around the
  boundary it detected. It also has a Streamlit web app so anyone can try it
  without touching code.
- **Challenge Task/** - runs the same tool on 10 document photos and saves
  the original, edge detection result, morphological cleanup result, and
  final boundary-detection result for each one, plus a summary table.

Each folder has its own `HOW_TO_RUN.txt` with step by step instructions.

## Sobel vs Laplacian vs Canny

All three try to answer the same question - "where does the brightness
change sharply?" - but go about it differently:

- **Sobel** checks ONE direction at a time (horizontal edges separately from
  vertical edges), then you combine them if you want edges in every
  direction. It's basically "how fast is brightness changing here, and which
  way." Cheap and fast, but the edges it finds are thick and a bit soft.
- **Laplacian** checks ALL directions at once, using the rate of change of
  the rate of change. One pass, edges in every direction - but it reacts to
  noise more than Sobel does, so blurring first matters even more.
- **Canny** is really Sobel plus extra smarts on top: it thins edges down to
  a single clean line, then uses two thresholds (a low and a high one) to
  decide what's a real edge - anything above the high threshold is
  definitely kept, anything below the low one is thrown away, and anything
  in between is kept only if it connects to a definite edge. That "keep it
  if it's connected" rule is what makes Canny so much cleaner than the other
  two - it doesn't lose a faint-but-real edge (like a soft shadow line) as
  long as it connects to a strong one. This is why the document boundary
  tool uses Canny, not Sobel or Laplacian.

## Why each morphological operation matters

These all work on binary (black/white) images and clean up the shape of the
white regions using a small sliding window (the "kernel"):

- **Erosion** shrinks white regions - removes small white noise specks, but
  can also shrink or break up real shapes if used too much.
- **Dilation** grows white regions - fills small black gaps, useful for
  reconnecting an edge line that got broken by a shadow.
- **Opening** (erode then dilate) removes small noise specks WITHOUT
  shrinking the real shape - the shape "grows back" in the dilate step, the
  noise doesn't.
- **Closing** (dilate then erode) fills small gaps/holes WITHOUT growing the
  real shape - the opposite trade-off from opening.
- **Morphological Gradient** (dilation minus erosion) leaves just the
  outline of each white region - a cheap way to turn a filled shape into an
  edge map.
- **Top Hat** (original minus opening) highlights small bright details that
  are smaller than the kernel.
- **Black Hat** (closing minus original) highlights small dark gaps that are
  smaller than the kernel.

## Which combination gave the best results

For the document boundary tool: **Gaussian blur -> auto-thresholded Canny ->
morphological closing -> largest 4-corner contour**. Closing (not opening)
turned out to matter the most out of the morphological operations - real
photos often have small gaps in the page's edge line caused by shadows or
uneven lighting, and closing bridges exactly those gaps so the outline forms
one connected shape instead of several broken pieces. Auto-thresholding
Canny from the image's own median brightness (instead of one hardcoded
threshold pair) made the edge step adapt across differently-lit photos
instead of needing to be re-tuned per image.

## Challenges I faced detecting document boundaries

- **Fixed Canny thresholds don't generalize.** A threshold pair that works
  on a bright, evenly-lit photo can miss edges entirely on a darker one.
  Switched to picking thresholds from the image's own median brightness
  instead of guessing fixed numbers.
- **Shadows and uneven lighting break the edge line into pieces**, so the
  page contour doesn't come out as one clean closed shape - `findContours`
  just sees several disconnected blobs instead of a rectangle. Morphological
  closing (with a couple iterations) fixes most of this by bridging the
  small gaps back together.
- **Not every photo has a clean 4-corner contour to find** - a heavily
  angled shadow, very low contrast between the page and the background, or
  a page with no real border can all mean `approxPolyDP` never simplifies
  down to a clean 4-point shape. Rather than the tool just failing, it falls
  back to a rotated bounding box of the largest shape it did find, and marks
  that result as "approximate" (drawn in orange instead of green) so it's
  obvious the result is a best guess, not a confirmed detection.
- **Large phone photos slow everything down and don't help accuracy** -
  contour detection doesn't need full resolution, so images are resized down
  before processing and the detected points are scaled back up to the
  original resolution afterward, only for the final drawing step.
- **A page with almost no contrast against its background is a real
  limitation.** If there's no border drawn on the page and the background is
  nearly the same color/brightness as the paper, there's genuinely no strong
  edge for Canny to find at the page boundary - this is a fundamental limit
  of contour-based detection, not something a threshold tweak fixes. Worth
  watching for in the Challenge Task results.

## Links

- Streamlit app: [add your public app link here]
- GitHub repo: [add your repo link here]
- Screen recording: [add your recording link here]
