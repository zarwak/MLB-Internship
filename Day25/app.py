"""
Day 25 - Image Feature Matching System (Streamlit app).

Upload two images, get back the ORB keypoints found in each, the matches
between them, and the numbers behind both.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from feature_detection import (
    bgr_to_rgb,
    detect_harris,
    detect_orb,
    draw_harris,
    draw_orb,
    load_image,
    resize_max_side,
)
from feature_matching import (
    MIN_INLIERS_FOR_HOMOGRAPHY,
    draw_detected_object,
    draw_matches,
    match_images,
)

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "sample_images"
MAX_SIDE = 1000     # uploads are downscaled to this before anything else

st.set_page_config(page_title="Image Feature Matching System",
                   page_icon="🔍", layout="wide")


# ---------------------------------------------------------------------------
# Sample pairs
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def example_pairs() -> dict[str, tuple[str, str]]:
    """Map a readable label to the two files of each sample pair on disk."""
    pairs: dict[str, tuple[str, str]] = {}
    for path_a in sorted(SAMPLES.glob("pair*_a.*")):
        stem = path_a.name.rsplit("_a.", 1)[0]
        candidates = sorted(SAMPLES.glob(f"{stem}_b.*"))
        if candidates:
            label = stem.split("_", 1)[1].replace("_", " ")
            pairs[f"{stem[:6]} - {label}"] = (str(path_a), str(candidates[0]))
    return pairs


@st.cache_data(show_spinner=False)
def load_bgr(path: str) -> np.ndarray:
    return resize_max_side(load_image(path), MAX_SIDE)


def decode_upload(upload) -> np.ndarray:
    """Turn an uploaded file into a downscaled BGR array."""
    data = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {upload.name} as an image")
    return resize_max_side(image, MAX_SIDE)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def verdict(result) -> tuple[str, str]:
    """Turn the numbers into a one-line judgement.

    The thresholds mirror what the RANSAC step can actually support: below
    ~10 inliers a homography is not identifiable, so anything under that is
    reported as no match however many good matches the ratio test produced.
    """
    if result.n_good == 0:
        return "error", "No matches at all. These images share no repeatable structure."
    if result.n_inliers < MIN_INLIERS_FOR_HOMOGRAPHY:
        return "error", (
            f"{result.n_good} good matches, but only {result.n_inliers} of them agree "
            "on a consistent geometry. Treat this as no match - the survivors are "
            "coincidences on similar-looking texture.")
    if result.inlier_rate >= 80:
        return "success", (
            f"Strong match. {result.n_inliers} of {result.n_good} good matches "
            f"({result.inlier_rate:.0f}%) agree on one transform.")
    if result.inlier_rate >= 50:
        return "warning", (
            f"Decent match. {result.n_inliers} of {result.n_good} "
            f"({result.inlier_rate:.0f}%) are geometrically consistent - the rest are "
            "noise, or the scene is not flat enough for a single homography.")
    return "warning", (
        f"Weak match. Only {result.inlier_rate:.0f}% of the good matches survive the "
        "geometry check, so most of them are wrong.")


def side_by_side(panel_a: np.ndarray, panel_b: np.ndarray) -> np.ndarray:
    height = max(panel_a.shape[0], panel_b.shape[0])
    padded = []
    for panel in (panel_a, panel_b):
        pad = height - panel.shape[0]
        if pad:
            panel = np.vstack([panel, np.full((pad, panel.shape[1], 3), 32, np.uint8)])
        padded.append(panel)
    gap = np.full((height, 8, 3), 32, np.uint8)
    return np.hstack([padded[0], gap, padded[1]])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("🔍 Image Feature Matching System")
st.markdown(
    "Finds the points two images have in common, using **ORB** keypoints matched "
    "with a **Brute Force** matcher over Hamming distance.\n\n"
    "Every match has to survive two filters before it is counted: **Lowe's ratio "
    "test** keeps a match only when the best candidate is clearly better than the "
    "second best, then **RANSAC** keeps the survivors that agree on a single "
    "geometric transform. The second number is the one to trust - a large pile of "
    "good matches with a low inlier rate means the ratio test let coincidences through."
)

with st.sidebar:
    st.header("Input")
    pairs = example_pairs()
    source = st.radio("Image source",
                      ["Sample pair", "Upload your own"],
                      label_visibility="collapsed")

    image_a = image_b = None
    name_a, name_b = "image A", "image B"

    if source == "Sample pair":
        if not pairs:
            st.error("No sample images found. Run `python download_samples.py` first.")
        else:
            choice = st.selectbox("Sample pair", list(pairs))
            path_a, path_b = pairs[choice]
            image_a, image_b = load_bgr(path_a), load_bgr(path_b)
            name_a, name_b = Path(path_a).name, Path(path_b).name
    else:
        upload_a = st.file_uploader("Image A", type=["png", "jpg", "jpeg", "bmp", "webp"])
        upload_b = st.file_uploader("Image B", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if upload_a and upload_b:
            image_a, image_b = decode_upload(upload_a), decode_upload(upload_b)
            name_a, name_b = upload_a.name, upload_b.name

    st.header("Settings")
    n_features = st.slider("ORB keypoint budget", 100, 5000, 1000, 100,
                           help="More keypoints finds more matches but costs time")
    ratio = st.slider("Lowe ratio threshold", 0.50, 0.95, 0.75, 0.05,
                      help="Lower = stricter, fewer but more reliable matches")
    max_draw = st.slider("Match lines to draw", 10, 200, 50, 10,
                         help="Display only - does not change the counts")
    show_harris = st.checkbox(
        "Show Harris corners instead of ORB keypoints",
        help="Comparison only; matching always uses ORB")

if image_a is None or image_b is None:
    st.info("Pick a sample pair in the sidebar, or upload two images of your own.")
    st.stop()

with st.spinner("Detecting and matching features..."):
    result = match_images(image_a, image_b,
                          n_features=int(n_features), ratio=float(ratio))

# ---- headline numbers -----------------------------------------------------
level, message = verdict(result)
getattr(st, level)(message)

columns = st.columns(5)
columns[0].metric("Keypoints in A", result.count_a)
columns[1].metric("Keypoints in B", result.count_b)
columns[2].metric("Good matches", result.n_good,
                  help=f"Survived Lowe's ratio test at {result.ratio:.2f}")
columns[3].metric("Verified matches", result.n_inliers,
                  help="Good matches that agree on one homography (RANSAC)")
columns[4].metric("Inlier rate", f"{result.inlier_rate:.1f}%",
                  help="Verified / good - the quality signal")

# ---- views ----------------------------------------------------------------
tab_matches, tab_keypoints, tab_located, tab_detail = st.tabs(
    ["Matches", "Keypoints", "Location", "Details"])

with tab_matches:
    canvas = draw_matches(image_a, image_b, result, max_draw=int(max_draw))
    shown = min(int(max_draw), result.n_inliers or result.n_good)
    st.image(bgr_to_rgb(canvas), use_container_width=True,
             caption=f"Top {shown} matches drawn "
                     f"({'RANSAC-verified' if result.inlier_matches else 'ratio test only'})")
    if result.n_good == 0:
        st.warning("Nothing survived the ratio test - there is nothing to draw.")

with tab_keypoints:
    if show_harris:
        panel_a = draw_harris(image_a, detect_harris(image_a))
        panel_b = draw_harris(image_b, detect_harris(image_b))
        caption = "Harris corners - position only, no scale or orientation"
    else:
        panel_a = draw_orb(image_a, detect_orb(image_a, n_features=int(n_features)))
        panel_b = draw_orb(image_b, detect_orb(image_b, n_features=int(n_features)))
        caption = ("ORB keypoints - circle size is the descriptor patch, "
                   "the line is the measured orientation (strongest 300 drawn)")
    st.image(bgr_to_rgb(side_by_side(panel_a, panel_b)),
             use_container_width=True, caption=caption)

with tab_located:
    located = draw_detected_object(image_a, image_b, result)
    if located is None:
        st.warning(
            f"Needs at least {MIN_INLIERS_FOR_HOMOGRAPHY} verified matches to fit a "
            f"homography worth trusting - this pair has {result.n_inliers}.")
    else:
        st.image(bgr_to_rgb(located), use_container_width=True,
                 caption="Image A's border projected into image B via the homography")

with tab_detail:
    distances = [m.distance for m in result.good_matches]
    rows = [
        ("Image A", f"{name_a} - {image_a.shape[1]} x {image_a.shape[0]} px"),
        ("Image B", f"{name_b} - {image_b.shape[1]} x {image_b.shape[0]} px"),
        ("Match rate", f"{result.match_rate:.1f}% (good / smaller keypoint set)"),
        ("Detection time", f"{result.detect_ms:.0f} ms (both images)"),
        ("Matching time", f"{result.match_ms:.0f} ms (brute force, Hamming)"),
    ]
    if distances:
        rows.append(("Hamming distance",
                     f"best {min(distances):.0f}, mean {sum(distances) / len(distances):.1f}, "
                     f"worst {max(distances):.0f} (out of 256 bits)"))

    # Markdown rather than st.table, which would prepend a meaningless
    # row-index column to a two-column key/value list.
    table = ["| Measurement | Value |", "|---|---|"]
    table += [f"| {label} | {value} |" for label, value in rows]
    st.markdown("\n".join(table))

    for note in result.notes:
        st.caption(f"note: {note}")

    st.markdown(
        "**Why two match counts?** Brute force returns a nearest neighbour for "
        "*every* descriptor, including keypoints with no counterpart at all. The "
        "ratio test removes the ambiguous ones; RANSAC then removes the ones that "
        "do not fit the same transform as the majority. On a clean pair the two "
        "numbers are close. When they diverge, trust the second."
    )
    if result.homography is not None:
        st.markdown("**Estimated homography (A to B)**")
        st.code(np.array2string(result.homography, precision=4, suppress_small=True))

st.caption("Day 25 - ORB feature detection and matching with OpenCV. "
           "Images are downscaled to 1000 px on the long side before processing.")
