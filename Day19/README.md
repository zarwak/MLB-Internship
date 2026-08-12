# Day 19 - Contours & Shape Detection

This is my Day 19 work. After Day 18 found a document's edges with Canny +
morphology, today was about a different (and often simpler) way to find
objects: **contours** - and using them to not just locate shapes but say
*what* they are (circle, square, rectangle, triangle, polygon) and measure
them.

## What's in this folder

- **coding_practice/** - small practice scripts, one file per topic. Each
  one runs against a fake sample "shapes" image (generated in code, no real
  photo needed) so I could learn each technique in isolation before
  combining them.
- **shape_detection_system/** - the real mini project. You give it a photo
  containing shapes and it finds every shape, labels it, and reports its
  area and perimeter. It also has a Streamlit web app so anyone can try it
  without touching code.
- **Challenge Task/** - runs the same tool on 10 shape photos and saves the
  original, contour detection result, and final labeled-shapes result for
  each one, plus a summary table.

Each folder has its own `HOW_TO_RUN.txt` with step by step instructions.

## What are contours?

A contour is a curve joining all the continuous points along a boundary
that share the same color or intensity - in practice, the outline of a blob
of white pixels in a binary (black/white) image. It's just a list of (x, y)
points tracing that blob's edge.

## How contour detection works

1. **Grayscale + blur** - drop color info we don't need, smooth out noise
   that would otherwise create tiny fake contours.
2. **Threshold** - turn the image into pure black and white with Otsu's
   method, which picks the black/white cutoff from the image's own
   brightness histogram instead of a hardcoded number. `THRESH_BINARY_INV`
   makes shapes the white "foreground" (the common case: shapes drawn on a
   lighter background). If that guess is wrong, the white area ends up
   being *most* of the frame (the background) instead of a few small blobs
   (the shapes) - so the tool checks for that and flips the threshold
   automatically.
3. **`cv2.findContours`** - traces the outer boundary of every white blob.
   `RETR_EXTERNAL` keeps only outer boundaries (ignores holes inside
   shapes); `CHAIN_APPROX_SIMPLE` compresses straight-line segments down to
   their endpoints instead of storing every pixel along the line.
4. **Filter by size** - tiny contours (a few stray pixels of noise) are
   dropped before anything downstream sees them.
5. **Measure** - `cv2.contourArea` (enclosed area) and `cv2.arcLength`
   (perimeter, in pixels) on what's left.
6. **Classify** - see below.
7. **Draw + label** - outline, bounding rectangle, and a text label with the
   shape's name, area, and perimeter, all on a copy of the original image.

## Which shapes my program can detect

Classification is corner-counting on a simplified version of the contour,
plus one extra check for round shapes:

- `cv2.approxPolyDP(contour, 0.02 * perimeter, True)` simplifies a contour
  down to its key corner points - `epsilon` (2% of that shape's own
  perimeter) controls how aggressive the simplification is, so it scales
  with each shape's size instead of using one fixed pixel value.
- **3 corners -> Triangle.**
- **4 corners -> Square or Rectangle**, told apart by the aspect ratio of
  the bounding box (close to 1:1 = square, otherwise rectangle).
- **Otherwise, check circularity** = `4*pi*area / perimeter^2`, which is
  1.0 for a perfect circle and drops for anything elongated or angular.
  Above 0.80 -> **Circle**.
- **Everything else -> Polygon**, labeled with however many corners
  `approxPolyDP` found (e.g. "Polygon (5 sides)" for a pentagon).

## Real-world uses of shape detection

- **Manufacturing/QA** - checking that machine-cut or 3D-printed parts
  match their expected shape and size on an assembly line.
- **Document processing** - finding checkboxes, stamps, or table cells
  (rectangles) on a scanned form.
- **Traffic sign recognition** - a sign's outer shape (triangle, octagon,
  circle) narrows down what *kind* of sign it is before reading its text.
- **Sorting/counting** - counting coins (circles) or packages (rectangles)
  on a conveyor belt.
- **Robotics** - identifying and localizing simple objects to pick up.

## Challenges I faced

- **Shapes can be lighter OR darker than their background.** A hardcoded
  `THRESH_BINARY_INV` assumes shapes are the darker ink/print on a lighter
  background - true for most hand-drawn or printed shapes, but not
  guaranteed. Fixed by checking how much of the thresholded image ended up
  white: if it's more than half (meaning the *background* got picked as the
  foreground, not the shapes), the threshold is inverted back.
- **Round-ish polygons get confused with circles.** Testing against my
  synthetic sample image (which includes a hexagon), the hexagon's
  circularity came out around 0.90 - above my 0.80 "is it a circle"
  threshold - so it gets classified as a Circle instead of "Polygon (6
  sides)". A regular pentagon (circularity ~0.86) stayed under the
  threshold and classified correctly in my test, so the exact cutoff
  matters and isn't foolproof. This is a genuine limitation of a
  corners+circularity heuristic, not a bug to "solve away" - a
  higher-corner-count regular polygon and a circle really do look alike by
  these two measurements alone. Worth watching for in the Challenge Task
  results, especially with hexagons/octagons.
- **`approxPolyDP`'s epsilon has to scale with the shape**, not be one fixed
  pixel value - a small shape and a large shape both need roughly the same
  *proportion* of simplification. Using 2% of each contour's own perimeter
  (instead of a fixed number of pixels) means both a tiny shape and a huge
  one get simplified consistently.
- **Tiny noise contours** (stray pixels from JPEG compression, dust,
  uneven lighting) show up as their own "contours" after thresholding.
  Filtering out anything below a minimum-area threshold (scaled to the
  image's own size, not a fixed pixel count) before measuring or
  classifying keeps these out of the results.

## Links

- Streamlit app: [\[add your public app link here\]](https://shapedetectionsystem.streamlit.app/)
- GitHub repo: [\[add your repo link here\]](https://github.com/zarwak/MLB-Internship/tree/main/Day19)
