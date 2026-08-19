"""
Day 23 Mini Project - Document OCR Web Application (Streamlit).

Upload a document image -> it gets cleaned up -> the chosen engine reads
it -> the original and the extracted text sit side by side -> download
the text as a .txt file.

All the actual OCR work lives in ocr_pipeline.py, one folder up. This
file is only the interface: widgets, layout, downloads. Keeping the two
apart means the same pipeline can be run from the command line
(coding_practice/batch_test.py) without Streamlit involved at all, and
this app is the one place that pipeline gets a user-facing front end.

WHY st.cache_resource around the readers: Streamlit reruns this entire
script from the top every time you touch any widget. Without caching,
ticking a checkbox would reload the OCR model from disk - several dead
seconds per click. cache_resource is the cache meant for exactly this:
heavy objects built once and shared across every rerun and user. It is
keyed per engine name here, since each engine is its own separate set of
model files - loading one does not warm another.
"""

import sys
from pathlib import Path

import streamlit as st

# ocr_pipeline.py lives in Day23/, one level above this file - added to
# sys.path here (rather than moving/duplicating the module) so
# coding_practice/batch_test.py and this app share one copy of the OCR
# logic instead of drifting apart. Path(__file__) makes this work the
# same way locally and once deployed, regardless of the process's cwd.
BASE_DIR = Path(__file__).parent
DAY_DIR = BASE_DIR.parent
sys.path.insert(0, str(DAY_DIR))

from ocr_pipeline import (ENGINE_NAMES, ENGINES, PRESET_LABELS, PRESETS,
                          get_reader, load_image, run_ocr, run_ocr_auto)

SAMPLES_DIR = DAY_DIR / "sample_images"

st.set_page_config(page_title="Document OCR Reader", layout="wide")


@st.cache_resource(show_spinner="Loading the OCR model (first run only)...")
def load_reader(engine: str):
    return get_reader(engine=engine)


def confidence_note(score: float) -> tuple[str, str]:
    """Turns a raw confidence number into something a person can act on.
    A bare '0.62' does not tell anyone whether to trust the output."""
    if score >= 0.85:
        return "Looks good", "The model was confident about nearly every line."
    if score >= 0.65:
        return "Mostly fine", "Worth skimming the text for a few wrong words."
    if score > 0:
        return "Shaky", "Try another preprocessing option, or a sharper image."
    return "Nothing found", "No text was detected at all."


# st.metric's delta coloring only understands numbers (sign of a number
# decides red vs green) - handed a plain string like "Shaky" it cannot
# tell that from "Looks good" and colors both the same misleading green,
# arrow and all. Rendering our own badge instead of relying on that
# means the color always matches what the label actually says.
CONFIDENCE_BADGE_COLORS = {
    "Looks good": ("#166534", "#DCFCE7"),
    "Mostly fine": ("#92400E", "#FEF3C7"),
    "Shaky": ("#9A3412", "#FFEDD5"),
    "Nothing found": ("#991B1B", "#FEE2E2"),
}


def confidence_badge(label: str) -> str:
    text_color, background_color = CONFIDENCE_BADGE_COLORS[label]
    return ('<span style="background-color:{bg};color:{fg};padding:2px 10px;'
            'border-radius:999px;font-size:0.85rem;font-weight:600;">{label}'
            '</span>').format(bg=background_color, fg=text_color, label=label)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")

    engine_choice = st.selectbox(
        "OCR Engine", options=list(ENGINES), index=0,
        format_func=lambda key: ENGINE_NAMES[key],
        help="Four OCR engines are available, each with its own "
             "detection and recognition models - they do not always "
             "read an image the same way.")

    auto_mode = st.toggle(
        "Auto mode", value=True,
        help="Tries a few preprocessing recipes and keeps whichever one "
             "the model was most confident about. Slower, because it "
             "reads the image more than once.")

    if auto_mode:
        preset = None
        st.caption("Auto mode will try: light, contrast, and no preprocessing.")
    else:
        preset = st.selectbox(
            "Preprocessing", options=list(PRESETS.keys()), index=1,
            format_func=lambda key: PRESET_LABELS[key])

    straighten = st.checkbox(
        "Straighten a tilted page", value=False,
        help="Detects how far the page is rotated and rotates it back. "
             "Leave off for images that are already straight - rotating "
             "resamples every pixel and slightly softens the text.")


# --------------------------------------------------------------- main layout
st.title("Document OCR Reader")
st.write("Upload a document image and get the text out of it. Works on "
         "printed documents, receipts, invoices, forms, book pages and "
         "ID cards.")

uploaded_file = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"],
    help="A photo or a scan. Straight-on and in focus works best.")

if uploaded_file is None:
    st.info("Upload an image to get started.")
    sample_files = sorted(SAMPLES_DIR.glob("*.png")) if SAMPLES_DIR.exists() else []
    if sample_files:
        with st.expander("No image handy? Here are the sample images in this repo"):
            columns = st.columns(4)
            for i, path in enumerate(sample_files[:8]):
                with columns[i % 4]:
                    st.image(str(path), caption=path.stem, width="stretch")
    st.stop()

image = load_image(uploaded_file)

reader = load_reader(engine_choice)
with st.spinner("Reading the image..."):
    if auto_mode:
        result, processed, angle = run_ocr_auto(image, straighten=straighten,
                                                engine=engine_choice, reader=reader)
    else:
        result, processed, angle = run_ocr(image, preset=preset, straighten=straighten,
                                           engine=engine_choice, reader=reader)

if angle:
    st.info("Page was tilted by {:.1f} degrees - straightened before "
            "reading.".format(angle))

left, right = st.columns(2)

with left:
    st.subheader("Original image")
    st.image(image, width="stretch")
    with st.expander("What the model actually saw (after preprocessing)"):
        st.image(processed, width="stretch")
        st.caption("Preprocessing used: **{}**".format(result.preset or "none"))

with right:
    st.subheader("Extracted text")

    if not result.text:
        st.error("No text found in this image.")
        st.write("Things that usually help:")
        tips = []
        if auto_mode:
            tips.append("Turn **Auto mode** off and try the *scanned document* preset")
        else:
            tips.append("Try the *scanned document* preset, or switch **Auto mode** on")
        tips.append("Tick **Straighten a tilted page** if the photo is crooked")
        tips.append("Try a different **OCR Engine** - they do not always fail on the same images")
        tips.append("Use a larger or sharper image - very small text is the most common cause")
        st.write("\n".join("- " + t for t in tips))
        st.stop()

    label, advice = confidence_note(result.avg_confidence)
    a, b, c = st.columns(3)
    a.metric("Confidence", "{:.0%}".format(result.avg_confidence))
    b.metric("Lines found", result.line_count)
    c.metric("Characters", result.char_count)
    st.markdown(confidence_badge(label), unsafe_allow_html=True)
    st.caption(advice)

    edited_text = st.text_area("Text (you can fix any mistakes before "
                               "downloading)", result.text, height=340)

    st.download_button(
        "Download as .txt",
        data=edited_text.encode("utf-8"),
        file_name=Path(uploaded_file.name).stem + "_extracted.txt",
        mime="text/plain",
        type="primary")

# The per-line confidences are the most useful debugging view there is:
# a bad average usually comes from two or three specific bad lines, not
# from the whole page being wrong, and this shows which ones.
with st.expander("Line-by-line confidence ({} lines)".format(result.line_count)):
    st.caption("Lines under 50% are the ones most likely to be misread.")
    for text, confidence in result.lines:
        marker = " (low)" if confidence < 0.5 else ""
        st.write("`{:.2f}{}`  {}".format(confidence, marker, text))
