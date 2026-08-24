"""
Image segmentation: thresholding methods, watershed, and background removal.

This module is the shared engine for the Day-26 coding practice scripts and
the Streamlit app. Everything works on plain grayscale/BGR numpy arrays, so
it does not care whether the image came from disk or from a web upload.

Run it directly to segment a single image with every method and write a
side-by-side comparison:

    python segmentation.py sample_images/doc01_invoice_clean.jpg
    python segmentation.py my.jpg --method otsu --out out.png
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Defaults. Kept in one place so the app, the CLI and the practice scripts
# all report numbers produced by the same settings.
# ---------------------------------------------------------------------------

BINARY_THRESH = 127
ADAPTIVE_BLOCK_SIZE = 35     # must be odd; size of the local neighbourhood
ADAPTIVE_C = 10              # constant subtracted from the local mean/gaussian
MORPH_KERNEL = 5             # cleanup kernel for background removal


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    """A binary mask produced by one thresholding method."""

    mask: np.ndarray                        # uint8, 0 or 255, same size as input
    method: str
    threshold_value: float | None           # the scalar cutoff, when there is one
    elapsed_ms: float
    params: dict = field(default_factory=dict)

    @property
    def foreground_ratio(self) -> float:
        """Fraction of pixels that ended up white (0-100)."""
        return 100.0 * float(np.count_nonzero(self.mask)) / self.mask.size


@dataclass
class SegmentationResult:
    """Foreground/background split: a clean mask plus the object cut out."""

    mask: np.ndarray            # uint8, 0/255, object = 255
    foreground: np.ndarray      # BGR, background pixels zeroed
    n_components: int           # connected foreground blobs after cleanup
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(source) -> np.ndarray:
    """Accept a path or a BGR/RGB array -> return a BGR array."""
    if isinstance(source, np.ndarray):
        image = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"no such image: {path}")
        # imdecode rather than imread so non-ASCII paths work on Windows
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode as an image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(image)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_max_side(image: np.ndarray, max_side: int = 1000) -> np.ndarray:
    """Shrink very large uploads so the app stays responsive.

    Only ever downscales - upscaling would invent detail it doesn't have.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _odd(n: int) -> int:
    """Block sizes for adaptive thresholding must be odd and >= 3."""
    n = int(n)
    return n if n % 2 == 1 else n + 1


# ---------------------------------------------------------------------------
# Thresholding methods
# ---------------------------------------------------------------------------

def binary_threshold(image: np.ndarray, thresh: int = BINARY_THRESH,
                     invert: bool = False) -> ThresholdResult:
    """A single global cutoff: pixel > thresh -> 255, else 0.

    Cheapest and most predictable method, but only works when lighting is
    even across the whole image - one cutoff has to be right for every pixel.
    """
    gray = to_gray(image)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    start = time.perf_counter()
    value, mask = cv2.threshold(gray, thresh, 255, flag)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return ThresholdResult(mask=mask, method="binary", threshold_value=value,
                           elapsed_ms=elapsed_ms,
                           params={"thresh": thresh, "invert": invert})


def adaptive_threshold(image: np.ndarray, block_size: int = ADAPTIVE_BLOCK_SIZE,
                       c: int = ADAPTIVE_C, method: str = "gaussian",
                       invert: bool = False) -> ThresholdResult:
    """Threshold each pixel against the *local* mean/gaussian-weighted mean.

    Because the cutoff is recomputed per neighbourhood, this survives
    lighting gradients and shadows that break a single global threshold -
    at the cost of being noisier on flat, evenly-lit regions.
    """
    gray = to_gray(image)
    block_size = _odd(block_size)
    adaptive_method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
                       else cv2.ADAPTIVE_THRESH_MEAN_C)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    start = time.perf_counter()
    mask = cv2.adaptiveThreshold(gray, 255, adaptive_method, flag, block_size, c)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return ThresholdResult(mask=mask, method=f"adaptive-{method}", threshold_value=None,
                           elapsed_ms=elapsed_ms,
                           params={"block_size": block_size, "c": c,
                                  "method": method, "invert": invert})


def otsu_threshold(image: np.ndarray, invert: bool = False,
                   blur: bool = True) -> ThresholdResult:
    """Pick the global cutoff automatically from the image's histogram.

    Otsu scans every possible threshold and keeps the one that best splits
    the histogram into two classes (minimises within-class variance). It
    still assumes one cutoff fits the whole image - a bimodal histogram
    (clear foreground vs. background peaks) is what it needs to work well.
    A light Gaussian blur first removes the noise spikes that otherwise
    shift the automatically-picked value.
    """
    gray = to_gray(image)
    if blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    flag |= cv2.THRESH_OTSU

    start = time.perf_counter()
    value, mask = cv2.threshold(gray, 0, 255, flag)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return ThresholdResult(mask=mask, method="otsu", threshold_value=value,
                           elapsed_ms=elapsed_ms,
                           params={"invert": invert, "blur": blur})


THRESHOLD_METHODS = {
    "binary": binary_threshold,
    "adaptive_mean": lambda img, **kw: adaptive_threshold(img, method="mean", **kw),
    "adaptive_gaussian": lambda img, **kw: adaptive_threshold(img, method="gaussian", **kw),
    "otsu": otsu_threshold,
}


# ---------------------------------------------------------------------------
# Watershed - separating touching objects
# ---------------------------------------------------------------------------

def _local_maxima_seeds(dist: np.ndarray, min_distance: int, rel_threshold: float = 0.35) -> np.ndarray:
    """One seed per local peak in a distance-transform map.

    A single global threshold on the distance map (`dist > 0.5 * dist.max()`,
    as many introductory tutorials do) only separates objects whose peak is
    much lower than the tallest one - two similarly-sized *touching* circles
    both stay above the cutoff and merge back into one seed. Instead, this
    keeps a pixel only when it is the local maximum within its own
    `min_distance` neighbourhood, which recovers one seed per object
    regardless of how the peaks compare to each other.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_distance, min_distance))
    dilated = cv2.dilate(dist, kernel)
    peak_mask = (dist == dilated) & (dist > rel_threshold * dist.max())
    return (peak_mask.astype(np.uint8)) * 255


def watershed_segmentation(image: np.ndarray, invert: bool = False,
                           min_distance: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Split touching foreground blobs using the watershed algorithm.

    Otsu/adaptive thresholding can find "this is foreground" but two round
    objects touching each other come out as one blob with no boundary
    between them. Watershed fixes that in four steps:

      1. Otsu-threshold to get a rough foreground mask.
      2. Distance transform: every foreground pixel gets the distance to
         the nearest background pixel, so blob *centres* score highest.
      3. Find one seed per local peak of the distance map (see
         `_local_maxima_seeds`) - one per object, even touching ones,
         because the seam between two circles is a *local* dip even when
         it is not below half the tallest peak in the image.
      4. Run cv2.watershed with those seeds as markers; it floods outward
         from each seed and draws a boundary (-1) where two floods meet.

    Returns (mask, markers, n_objects) - mask is the cleaned binary
    foreground, markers is the labelled image (-1 = watershed boundary,
    1 = background, 2..N = one label per separated object), n_objects is
    the number of separated foreground blobs.
    """
    gray = to_gray(image)
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                            flag | cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)

    if min_distance is None:
        min_distance = _odd(max(9, int(round(0.05 * min(gray.shape[:2])))))

    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    sure_fg = _local_maxima_seeds(dist, min_distance)

    unknown = cv2.subtract(sure_bg, sure_fg)
    n_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1                 # background becomes 1, not 0
    markers[unknown == 255] = 0           # 0 = "let watershed decide"

    markers = cv2.watershed(color.copy(), markers)
    n_objects = n_labels - 1              # exclude the background label

    return mask, markers, n_objects


def draw_watershed(image: np.ndarray, markers: np.ndarray,
                   boundary_color=(0, 0, 255)) -> np.ndarray:
    """Overlay watershed boundaries and a distinct tint per separated object."""
    color = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = color.copy()

    rng = np.random.default_rng(0)
    labels = [l for l in np.unique(markers) if l > 1]
    for label in labels:
        tint = rng.integers(60, 255, size=3).tolist()
        overlay[markers == label] = tint

    blended = cv2.addWeighted(color, 0.55, overlay, 0.45, 0)
    blended[markers == -1] = boundary_color
    return blended


# ---------------------------------------------------------------------------
# Foreground / background segmentation ("cut the object out")
# ---------------------------------------------------------------------------

def segment_foreground(image: np.ndarray, method: str = "otsu",
                       invert: bool = False, morph_kernel: int = MORPH_KERNEL,
                       min_area_ratio: float = 0.002) -> SegmentationResult:
    """Produce a clean foreground mask and the object cut out of the image.

    Thresholding alone gives a noisy mask (salt-and-pepper pixels, small
    stray blobs). This adds the two cleanup steps that turn a raw threshold
    into a usable segmentation:

      - Morphological opening (erode then dilate) removes small noise specks
        without shrinking the main object back to its original size.
      - Morphological closing (dilate then erode) fills small holes inside
        the object (e.g. a bright reflection inside a dark shape).
      - Contour filtering drops any surviving blob smaller than
        `min_area_ratio` of the image - these are near-guaranteed noise, not
        a second object.
    """
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = to_gray(image)

    start = time.perf_counter()
    thresholder = THRESHOLD_METHODS.get(method, otsu_threshold)
    raw_mask = thresholder(gray, invert=invert).mask

    k = max(3, morph_kernel | 1)   # odd, >= 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * gray.size
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]

    final_mask = np.zeros_like(cleaned)
    cv2.drawContours(final_mask, kept, -1, 255, thickness=cv2.FILLED)

    foreground = cv2.bitwise_and(color, color, mask=final_mask)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return SegmentationResult(mask=final_mask, foreground=foreground,
                              n_components=len(kept), elapsed_ms=elapsed_ms)


def remove_background(image: np.ndarray, method: str = "otsu",
                      invert: bool = False, bg_color=(255, 255, 255)) -> np.ndarray:
    """Replace the background with a flat colour instead of black.

    Built on `segment_foreground` - useful on its own for document/object
    photos destined for a catalogue or a print-friendly export.
    """
    result = segment_foreground(image, method=method, invert=invert)
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    canvas = np.full_like(color, bg_color, dtype=np.uint8)
    inv_mask = cv2.bitwise_not(result.mask)
    background = cv2.bitwise_and(canvas, canvas, mask=inv_mask)
    return cv2.add(result.foreground, background)


# ---------------------------------------------------------------------------
# Layout helpers used by the practice scripts and the CLI
# ---------------------------------------------------------------------------

def label_image(image: np.ndarray, text: str, height: int = 34) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bar = np.full((height, image.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, text, (10, height - 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def hstack_padded(images, gap: int = 8) -> np.ndarray:
    images = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im for im in images]
    tallest = max(img.shape[0] for img in images)
    padded = []
    for index, img in enumerate(images):
        pad = tallest - img.shape[0]
        if pad:
            img = np.vstack([img, np.full((pad, img.shape[1], 3), 32, np.uint8)])
        padded.append(img)
        if gap and index < len(images) - 1:
            padded.append(np.full((tallest, gap, 3), 32, np.uint8))
    return np.hstack(padded)


def grid_padded(rows: list[list[np.ndarray]], gap: int = 8) -> np.ndarray:
    """Stack a list of rows (each a list of same-shaped panels) into a grid."""
    row_images = [hstack_padded(row, gap=gap) for row in rows]
    widest = max(img.shape[1] for img in row_images)
    padded = []
    for index, img in enumerate(row_images):
        pad = widest - img.shape[1]
        if pad:
            img = np.hstack([img, np.full((img.shape[0], pad, 3), 32, np.uint8)])
        padded.append(img)
        if gap and index < len(row_images) - 1:
            padded.append(np.full((gap, widest, 3), 32, np.uint8))
    return np.vstack(padded)


VIS_EXT = ".jpg"
JPEG_QUALITY = 92


def imwrite(path, image: np.ndarray) -> None:
    """Write an image, creating parent folders and tolerating odd paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix or ".png"
    params = ([int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
              if suffix.lower() in {".jpg", ".jpeg"} else [])

    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise IOError(f"could not encode image for {path}")
    encoded.tofile(str(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Segment an image with binary / adaptive / Otsu thresholding.")
    parser.add_argument("image", help="path to an input image")
    parser.add_argument("--method",
                        choices=["binary", "adaptive_mean", "adaptive_gaussian",
                                "otsu", "all"],
                        default="all")
    parser.add_argument("--invert", action="store_true",
                        help="use THRESH_BINARY_INV (dark object on light bg)")
    parser.add_argument("--out", default=None, help="where to write the visualisation")
    args = parser.parse_args()

    image = resize_max_side(load_image(args.image))
    gray = to_gray(image)

    methods = list(THRESHOLD_METHODS) if args.method == "all" else [args.method]
    panels = [label_image(gray, "grayscale")]
    for name in methods:
        result = THRESHOLD_METHODS[name](gray, invert=args.invert)
        value_str = f", t={result.threshold_value:.0f}" if result.threshold_value else ""
        caption = f"{name}{value_str} ({result.foreground_ratio:.1f}% fg, {result.elapsed_ms:.1f} ms)"
        print(caption)
        panels.append(label_image(result.mask, caption))

    out = args.out
    if out is None:
        out = Path("sample_outputs") / f"segment_{Path(args.image).stem}_{args.method}{VIS_EXT}"
    imwrite(out, hstack_padded(panels))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
