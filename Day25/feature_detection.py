"""
Feature detection: Harris corners and ORB keypoints.

This module is the shared engine for the Day-25 coding practice scripts and
for the Gradio app. Everything here works on plain BGR numpy arrays, so it
does not care whether the image came from disk or from a web upload.

Run it directly to detect on a single image and write a visualisation:

    python feature_detection.py sample_images/pair02_graffiti_wall_a.png
    python feature_detection.py my.jpg --method harris --out out.png
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

HARRIS_BLOCK_SIZE = 2       # size of the neighbourhood used to build M
HARRIS_KSIZE = 3            # Sobel aperture for the image gradients
HARRIS_K = 0.04             # the k in  R = det(M) - k * trace(M)^2
HARRIS_THRESHOLD = 0.01     # keep responses above 1% of the strongest one

ORB_N_FEATURES = 1000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class HarrisResult:
    """Corners found by Harris, plus the raw response map behind them."""

    corners: np.ndarray                     # (N, 2) float32 array of (x, y)
    response: np.ndarray                    # float32 response map, image sized
    elapsed_ms: float
    params: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(len(self.corners))


@dataclass
class OrbResult:
    """Keypoints and the 32-byte binary descriptors that go with them."""

    keypoints: tuple                        # tuple of cv2.KeyPoint
    descriptors: np.ndarray | None          # (N, 32) uint8, or None if nothing found
    elapsed_ms: float
    params: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(len(self.keypoints))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(source) -> np.ndarray:
    """Accept a path, a BGR array, or an RGB array from Gradio -> return BGR.

    Gradio hands us RGB numpy arrays; OpenCV wants BGR. Anything that is
    already a 3-channel array is assumed to be BGR, so callers coming from
    Gradio should convert first with `rgb_to_bgr`.
    """
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


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert a Gradio-style RGB array into the BGR OpenCV expects."""
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
    return load_image(image)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert back to RGB so Gradio displays the colours correctly."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_max_side(image: np.ndarray, max_side: int = 1000) -> np.ndarray:
    """Shrink very large uploads so the app stays responsive.

    Only ever downscales - upscaling would invent detail and inflate the
    keypoint counts without adding real information.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Harris corner detection
# ---------------------------------------------------------------------------

def detect_harris(
    image: np.ndarray,
    block_size: int = HARRIS_BLOCK_SIZE,
    ksize: int = HARRIS_KSIZE,
    k: float = HARRIS_K,
    threshold: float = HARRIS_THRESHOLD,
    refine: bool = True,
) -> HarrisResult:
    """Detect corners with `cv2.cornerHarris`.

    `cornerHarris` gives back a response map, not a list of corners, so there
    are two extra steps most tutorials gloss over:

    1. Threshold the map at `threshold * response.max()`.
    2. Collapse each surviving blob to a single point. A strong corner lights
       up a small cluster of pixels, and counting raw pixels would report one
       corner as thirty. Connected components + centroids fixes that, and is
       the difference between "4000 corners" and the few hundred real ones.

    With `refine=True` the centroids are then pushed to sub-pixel accuracy.
    """
    gray = to_gray(image)
    gray32 = np.float32(gray)

    start = time.perf_counter()
    response = cv2.cornerHarris(gray32, block_size, ksize, k)
    response = cv2.dilate(response, None)      # fatten peaks so they survive thresholding

    peak = float(response.max())
    params = {"block_size": block_size, "ksize": ksize, "k": k,
              "threshold": threshold, "refine": refine}

    if peak <= 0:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return HarrisResult(np.empty((0, 2), np.float32), response, elapsed_ms, params)

    mask = (response > threshold * peak).astype(np.uint8)

    # One centroid per connected blob. Label 0 is the background, so skip it.
    n_labels, _, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    corners = centroids[1:].astype(np.float32) if n_labels > 1 else np.empty((0, 2), np.float32)

    if refine and len(corners):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners = cv2.cornerSubPix(
            gray, np.ascontiguousarray(corners).reshape(-1, 1, 2),
            winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria,
        ).reshape(-1, 2)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return HarrisResult(corners=corners, response=response,
                        elapsed_ms=elapsed_ms, params=params)


def draw_harris(image: np.ndarray, result: HarrisResult,
                color=(0, 0, 255), radius: int = 3) -> np.ndarray:
    """Mark every Harris corner with a small filled circle."""
    canvas = image.copy()
    for x, y in result.corners:
        cv2.circle(canvas, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)
    return canvas


def draw_harris_response(image: np.ndarray, result: HarrisResult) -> np.ndarray:
    """Heat-map view of the raw response, useful for explaining the threshold."""
    response = result.response
    spread = float(response.max() - response.min())
    if spread <= 0:
        normalised = np.zeros(response.shape, np.uint8)
    else:
        normalised = ((response - response.min()) / spread * 255).astype(np.uint8)
    heat = cv2.applyColorMap(normalised, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.45, heat, 0.55, 0)


# ---------------------------------------------------------------------------
# ORB detection
# ---------------------------------------------------------------------------

def build_orb(n_features: int = ORB_N_FEATURES,
              scale_factor: float = ORB_SCALE_FACTOR,
              n_levels: int = ORB_N_LEVELS):
    return cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=scale_factor,
        nlevels=n_levels,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,   # ORB ranks its FAST corners by Harris response
        patchSize=31,
        fastThreshold=20,
    )


def detect_orb(image: np.ndarray, n_features: int = ORB_N_FEATURES,
               scale_factor: float = ORB_SCALE_FACTOR,
               n_levels: int = ORB_N_LEVELS) -> OrbResult:
    """Detect ORB keypoints and compute their descriptors in one pass.

    Note `scoreType=ORB_HARRIS_SCORE`: ORB finds candidates with FAST and then
    ranks them with the *same* Harris response used above. ORB is not really a
    rival to Harris so much as Harris wrapped in a scale pyramid, an
    orientation estimate and a binary descriptor.
    """
    gray = to_gray(image)
    orb = build_orb(n_features, scale_factor, n_levels)

    start = time.perf_counter()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return OrbResult(
        keypoints=tuple(keypoints or ()),
        descriptors=descriptors,
        elapsed_ms=elapsed_ms,
        params={"n_features": n_features, "scale_factor": scale_factor,
                "n_levels": n_levels},
    )


def draw_orb(image: np.ndarray, result: OrbResult, rich: bool = True,
             color=(0, 255, 0), max_draw: int | None = 300) -> np.ndarray:
    """Draw ORB keypoints.

    `rich=True` draws each keypoint as a circle sized by the pyramid level it
    was found on, with a radius line showing its measured orientation - the
    two things Harris does not give you.

    A full 1000 rich circles turns any image into a solid green blob, so by
    default only the `max_draw` strongest keypoints get circles. Pass
    `max_draw=None` to draw all of them, or `rich=False` for plain dots.
    """
    keypoints = result.keypoints
    if max_draw is not None and len(keypoints) > max_draw:
        keypoints = tuple(sorted(keypoints, key=lambda kp: kp.response,
                                 reverse=True)[:max_draw])

    flags = (cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich
             else cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    return cv2.drawKeypoints(image, keypoints, None, color=color, flags=flags)


# ---------------------------------------------------------------------------
# Layout helpers used by the practice scripts
# ---------------------------------------------------------------------------

def label_image(image: np.ndarray, text: str, height: int = 34) -> np.ndarray:
    """Put a caption bar above an image."""
    bar = np.full((height, image.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, text, (10, height - 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def hstack_padded(images, gap: int = 8) -> np.ndarray:
    """Stack images side by side, padding to a common height."""
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


# Extension for the generated visualisations. These are photographs with
# annotation drawn over them, so JPEG costs nothing visible and keeps
# sample_outputs/ small enough to live in the repo - PNG made it 29 MB.
# The matplotlib chart in practice 05 stays PNG, where text has to stay sharp.
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
        description="Detect Harris corners and/or ORB keypoints in an image.")
    parser.add_argument("image", help="path to an input image")
    parser.add_argument("--method", choices=["harris", "orb", "both"], default="both")
    parser.add_argument("--out", default=None, help="where to write the visualisation")
    parser.add_argument("--orb-features", type=int, default=ORB_N_FEATURES)
    parser.add_argument("--harris-threshold", type=float, default=HARRIS_THRESHOLD)
    args = parser.parse_args()

    image = resize_max_side(load_image(args.image))
    panels = []

    if args.method in ("harris", "both"):
        harris = detect_harris(image, threshold=args.harris_threshold)
        print(f"Harris : {harris.count:5d} corners   in {harris.elapsed_ms:7.2f} ms")
        panels.append(label_image(
            draw_harris(image, harris),
            f"Harris - {harris.count} corners ({harris.elapsed_ms:.1f} ms)"))

    if args.method in ("orb", "both"):
        orb = detect_orb(image, n_features=args.orb_features)
        if orb.descriptors is None:
            desc = "none"
        else:
            desc = f"{orb.descriptors.shape[0]}x{orb.descriptors.shape[1]} uint8"
        print(f"ORB    : {orb.count:5d} keypoints in {orb.elapsed_ms:7.2f} ms   descriptors: {desc}")
        shown = min(orb.count, 300)
        caption = f"ORB - {orb.count} keypoints ({orb.elapsed_ms:.1f} ms)"
        if shown < orb.count:
            caption += f", strongest {shown} drawn"
        panels.append(label_image(draw_orb(image, orb), caption))

    out = args.out
    if out is None:
        out = Path("sample_outputs") / f"detect_{Path(args.image).stem}_{args.method}{VIS_EXT}"
    imwrite(out, hstack_padded(panels) if len(panels) > 1 else panels[0])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
