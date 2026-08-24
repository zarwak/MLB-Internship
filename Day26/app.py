"""
Day 26 - Document & Object Segmentation Tool (Streamlit app).

Upload an image, pick a segmentation method, see the mask and the numbers
behind it, and download the result.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from segmentation import (THRESHOLD_METHODS, adaptive_threshold, bgr_to_rgb,
                          binary_threshold, load_image, otsu_threshold,
                          remove_background, resize_max_side, segment_foreground,
                          to_gray, watershed_segmentation, draw_watershed)

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "sample_images"
MAX_SIDE = 1000     # uploads are downscaled to this before anything else

st.set_page_config(page_title="Document & Object Segmentation Tool",
                   page_icon="✂️", layout="wide")

METHOD_LABELS = {
    "Binary Threshold": "binary",
    "Adaptive Threshold (Mean)": "adaptive_mean",
    "Adaptive Threshold (Gaussian)": "adaptive_gaussian",
    "Otsu Threshold": "otsu",
    "Watershed (separate touching objects)": "watershed",
    "Foreground / Background Segmentation": "fgbg",
}


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def sample_paths() -> dict[str, str]:
    return {p.stem: str(p) for p in sorted(SAMPLES.glob("*.jpg"))}


@st.cache_data(show_spinner=False)
def load_bgr(path: str) -> np.ndarray:
    return resize_max_side(load_image(path), MAX_SIDE)


def decode_upload(upload) -> np.ndarray:
    data = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {upload.name} as an image")
    return resize_max_side(image, MAX_SIDE)


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise IOError("could not encode image")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("✂️ Document & Object Segmentation Tool")
st.markdown(
    "Separates the main object or document text from its background using "
    "classic OpenCV segmentation: **Binary**, **Adaptive** and **Otsu** "
    "thresholding, **Watershed** for touching objects, and a full "
    "**foreground/background** pipeline with morphology cleanup."
)

with st.sidebar:
    st.header("Input")
    samples = sample_paths()
    source = st.radio("Image source", ["Sample image", "Upload your own"],
                      label_visibility="collapsed")

    image = None
    image_name = "image"

    if source == "Sample image":
        if not samples:
            st.error("No sample images found. Run `python generate_samples.py` first.")
        else:
            choice = st.selectbox("Sample image", list(samples))
            image = load_bgr(samples[choice])
            image_name = choice
    else:
        upload = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if upload:
            image = decode_upload(upload)
            image_name = Path(upload.name).stem

    st.header("Segmentation method")
    method_label = st.selectbox("Method", list(METHOD_LABELS), index=3)
    method = METHOD_LABELS[method_label]

    invert = st.checkbox("Invert (dark object on light background)", value=True,
                         help="Most document/object photos are dark subject on "
                              "a lighter background - leave this on unless your "
                              "result looks inverted.")

    st.header("Method parameters")
    if method == "binary":
        thresh = st.slider("Threshold", 0, 255, 127, 1)
    elif method in ("adaptive_mean", "adaptive_gaussian"):
        block_size = st.slider("Block size (neighbourhood, odd)", 3, 199, 35, 2)
        c = st.slider("C (subtracted from local mean)", -20, 30, 10, 1)
    elif method == "otsu":
        blur = st.checkbox("Gaussian blur before Otsu", value=True,
                           help="Removes noise spikes that shift the auto-picked threshold")
    elif method == "watershed":
        min_distance = st.slider("Minimum seed distance", 5, 199, 41, 2,
                                 help="Smaller = can separate closer object centres, "
                                      "but risks over-splitting a single object")
    elif method == "fgbg":
        base_method = st.selectbox("Threshold used to find the object",
                                   ["otsu", "adaptive_gaussian", "binary"], index=0)
        morph_kernel = st.slider("Cleanup kernel size", 3, 51, 5, 2)
        bg_choice = st.color_picker("Replacement background colour", "#FFFFFF")
        bg_color = tuple(int(bg_choice.lstrip("#")[i:i + 2], 16) for i in (4, 2, 0))  # BGR

if image is None:
    st.info("Pick a sample image in the sidebar, or upload one of your own.")
    st.stop()

gray = to_gray(image)

# ---- run the selected method ----------------------------------------------
with st.spinner("Segmenting..."):
    if method == "binary":
        result = binary_threshold(gray, thresh=thresh, invert=invert)
        display_mask = result.mask
    elif method in ("adaptive_mean", "adaptive_gaussian"):
        kind = "mean" if method == "adaptive_mean" else "gaussian"
        result = adaptive_threshold(gray, block_size=block_size, c=c,
                                    method=kind, invert=invert)
        display_mask = result.mask
    elif method == "otsu":
        result = otsu_threshold(gray, invert=invert, blur=blur)
        display_mask = result.mask
    elif method == "watershed":
        mask, markers, n_objects = watershed_segmentation(image, invert=invert,
                                                           min_distance=min_distance)
        display_mask = draw_watershed(image, markers)
    elif method == "fgbg":
        seg = segment_foreground(image, method=base_method, invert=invert,
                                 morph_kernel=morph_kernel)
        removed = remove_background(image, method=base_method, invert=invert,
                                    bg_color=bg_color)
        display_mask = removed

# ---- headline numbers -------------------------------------------------------
columns = st.columns(4)
columns[0].metric("Image size", f"{image.shape[1]} x {image.shape[0]}")

if method in ("binary", "adaptive_mean", "adaptive_gaussian", "otsu"):
    columns[1].metric("Foreground", f"{result.foreground_ratio:.1f}%")
    columns[2].metric("Threshold value",
                      f"{result.threshold_value:.0f}" if result.threshold_value is not None else "local (per pixel)")
    columns[3].metric("Time", f"{result.elapsed_ms:.1f} ms")
elif method == "watershed":
    columns[1].metric("Objects separated", n_objects)
    columns[2].metric("Foreground", f"{100 * np.count_nonzero(mask) / mask.size:.1f}%")
    columns[3].metric("Method", "Otsu + distance transform")
elif method == "fgbg":
    columns[1].metric("Components kept", seg.n_components)
    columns[2].metric("Foreground", f"{100 * np.count_nonzero(seg.mask) / seg.mask.size:.1f}%")
    columns[3].metric("Time", f"{seg.elapsed_ms:.1f} ms")

# ---- views ------------------------------------------------------------------
tab_result, tab_compare, tab_fgbg, tab_details = st.tabs(
    ["Selected Method", "Compare All Thresholds", "Foreground / Background", "Details"])

with tab_result:
    col_a, col_b, col_c = st.columns(3)
    col_a.image(bgr_to_rgb(image), use_container_width=True, caption="Original")
    col_b.image(gray, use_container_width=True, caption="Grayscale")
    caption = method_label
    if method in ("binary", "adaptive_mean", "adaptive_gaussian", "otsu"):
        col_c.image(display_mask, use_container_width=True, caption=caption)
        download_bytes = encode_png(display_mask)
    else:
        col_c.image(bgr_to_rgb(display_mask), use_container_width=True, caption=caption)
        download_bytes = encode_png(display_mask)

    st.download_button("Download this result (PNG)", data=download_bytes,
                       file_name=f"{image_name}_{method}.png", mime="image/png")

with tab_compare:
    st.caption("Every thresholding family run on the same grayscale image, so "
              "you can see where they agree and where they don't.")
    cols = st.columns(4)
    for col, (name, fn) in zip(cols, THRESHOLD_METHODS.items()):
        r = fn(gray, invert=invert)
        value = f", t={r.threshold_value:.0f}" if r.threshold_value is not None else ""
        col.image(r.mask, use_container_width=True,
                 caption=f"{name}{value}\n{r.foreground_ratio:.1f}% fg, {r.elapsed_ms:.1f} ms")

with tab_fgbg:
    st.caption("Otsu/adaptive thresholding + morphological open/close + "
              "contour-area filtering, to get a clean single-object mask "
              "instead of a noisy raw threshold.")
    seg_default = segment_foreground(image, method="otsu", invert=invert)
    removed_default = remove_background(image, method="otsu", invert=invert,
                                        bg_color=(255, 255, 255))
    col_a, col_b, col_c = st.columns(3)
    col_a.image(seg_default.mask, use_container_width=True,
               caption=f"Cleaned mask ({seg_default.n_components} component(s))")
    col_b.image(bgr_to_rgb(seg_default.foreground), use_container_width=True,
               caption="Foreground cut out (background zeroed)")
    col_c.image(bgr_to_rgb(removed_default), use_container_width=True,
               caption="Background replaced with white")
    st.download_button("Download foreground cut-out (PNG)",
                       data=encode_png(seg_default.foreground),
                       file_name=f"{image_name}_foreground.png", mime="image/png")

with tab_details:
    rows = [
        ("Image", f"{image_name} - {image.shape[1]} x {image.shape[0]} px"),
        ("Method", method_label),
        ("Invert", str(invert)),
    ]
    if method in ("binary", "adaptive_mean", "adaptive_gaussian", "otsu"):
        rows.append(("Parameters", str(result.params)))
        rows.append(("Foreground ratio", f"{result.foreground_ratio:.2f}%"))
    table = ["| Field | Value |", "|---|---|"]
    table += [f"| {label} | {value} |" for label, value in rows]
    st.markdown("\n".join(table))

    st.markdown(
        "**Method notes**\n\n"
        "- **Binary** - one global cutoff, fastest, needs even lighting.\n"
        "- **Adaptive (Mean/Gaussian)** - a per-pixel local cutoff, survives "
        "shadows and lighting gradients, but only reliably marks the *edges* "
        "of objects much larger than the block size (the interior looks "
        "locally uniform, so it doesn't cross its own local threshold).\n"
        "- **Otsu** - picks one global cutoff automatically from the "
        "histogram; matches or beats a hand-picked binary threshold on "
        "evenly lit, bimodal images, but inherits binary's weakness to "
        "lighting gradients.\n"
        "- **Watershed** - splits touching objects that thresholding alone "
        "reports as a single connected blob.\n"
        "- **Foreground/Background** - thresholding plus morphological "
        "open/close and contour-area filtering, which is what actually "
        "produces a clean single-object mask in practice."
    )

st.caption("Day 26 - Binary / Adaptive / Otsu thresholding, watershed and "
          "foreground segmentation with OpenCV. Uploads are downscaled to "
          "1000 px on the long side before processing.")
