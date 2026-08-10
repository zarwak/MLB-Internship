"""
Document Image Enhancement Tool - the core pipeline.
Order: load -> straighten (if tilted) -> grayscale -> remove noise ->
       fix brightness/contrast -> sharpen -> done.
Each step is its own small function, so you can test/understand one at a time.
"""
import cv2
import numpy as np


def find_document_corners(img):
    """Try to find the document's 4 corners in the photo.
    Returns 4 points, or None if no clear 4-corner shape was found (page not tilted / not visible)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # color isn't needed to find edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)    # smooth first so edges aren't noisy
    edges = cv2.Canny(blurred, 60, 160)            # outline of shapes in the photo
    edges = cv2.dilate(edges, None, iterations=1)  # thicken edges so the outline fully connects

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # check the 5 biggest shapes

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)  # simplify shape into straight edges
        if len(approx) == 4:  # a document page looks like a 4-corner shape
            return approx.reshape(4, 2).astype("float32")
    return None  # could not find a clear page outline


def order_points(pts):
    """Sorts 4 points into a fixed order: top-left, top-right, bottom-right, bottom-left.
    warpPerspective needs the corners in this exact, consistent order to work correctly."""
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)            # x+y is smallest at top-left, biggest at bottom-right
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)    # x-y is smallest at top-right, biggest at bottom-left
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def correct_perspective(img):
    """Straightens the document if we can find its 4 corners.
    If no clear page outline is found, just returns the original image unchanged."""
    corners = find_document_corners(img)
    if corners is None:
        print("  - No tilted page detected, skipping perspective correction.")
        return img

    (tl, tr, br, bl) = order_points(corners)

    # work out the flat document's width/height from the corner distances
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    dst = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])

    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), dst)  # build the "straighten" matrix
    return cv2.warpPerspective(img, M, (width, height))


def to_grayscale(img):
    """Color isn't needed to read text - grayscale is simpler and standard before OCR."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray_img):
    """Smooths out speckles/grain from a scan or camera photo, while keeping text edges fairly sharp."""
    return cv2.bilateralFilter(gray_img, d=9, sigmaColor=75, sigmaSpace=75)


def enhance_brightness_contrast(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE = smart local contrast boost. It fixes contrast in small regions instead of the
    whole image at once, so it doesn't blow out bright spots or crush dark shadows on the page.
    clip_limit: higher = stronger contrast boost."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)


def apply_brightness(img, beta):
    """Adds (beta > 0) or removes (beta < 0) plain brightness, on top of any other enhancement."""
    if beta == 0:
        return img
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def sharpen(img):
    """Makes text edges look crisper (boosts the center pixel, subtracts its neighbors)."""
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def enhance_document(img):
    """Runs every step in order and returns the final enhanced image.
    This is the fixed pipeline used by process.py for batch processing."""
    straightened = correct_perspective(img)                  # step 1: fix tilt
    gray = to_grayscale(straightened)                         # step 2: drop color
    denoised = reduce_noise(gray)                              # step 3: clean noise
    contrast_fixed = enhance_brightness_contrast(denoised)     # step 4: fix brightness/contrast
    final = sharpen(contrast_fixed)                             # step 5: sharpen text
    return final


def run_pipeline(img, correct_tilt=True, convert_gray=True, denoise=True,
                  fix_contrast=True, clip_limit=2.0, brightness_beta=0, sharpen_image=True):
    """Same steps as enhance_document, but each one can be switched on/off and tuned.
    Used by the Streamlit app so the user can pick which steps to apply.
    Returns a dictionary of every step's result, so the UI can display each one."""
    steps = {"Original": img}
    current = img

    if correct_tilt:
        current = correct_perspective(current)
        steps["Perspective Corrected"] = current

    if convert_gray:
        current = to_grayscale(current)
        steps["Grayscale"] = current

    if denoise:
        current = reduce_noise(current)
        steps["Noise Reduced"] = current

    if fix_contrast:
        if current.ndim == 2:  # already single-channel (grayscale)
            current = enhance_brightness_contrast(current, clip_limit=clip_limit)
        else:  # CLAHE needs a single channel, so apply it only to the lightness channel
            lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = enhance_brightness_contrast(l_channel, clip_limit=clip_limit)
            current = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        steps["Contrast Enhanced"] = current

    if brightness_beta != 0:
        current = apply_brightness(current, brightness_beta)
        steps["Brightness Adjusted"] = current

    if sharpen_image:
        current = sharpen(current)
        steps["Sharpened"] = current

    steps["Final"] = current
    return steps
