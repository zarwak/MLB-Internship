"""
Day 25 - Image Feature Matching System (Gradio app).

Upload two images, get back the ORB keypoints found in each, the matches
between them, and the numbers behind both.

Run locally:  python gradio_app.py
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np

from feature_detection import (
    bgr_to_rgb,
    detect_harris,
    detect_orb,
    draw_harris,
    draw_orb,
    resize_max_side,
    rgb_to_bgr,
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


def _example_pairs() -> list[list[str]]:
    """Build the example gallery from whatever pairs are on disk."""
    examples = []
    for path_a in sorted(SAMPLES.glob("pair*_a.*")):
        candidates = sorted(SAMPLES.glob(path_a.name.replace("_a.", "_b.").rsplit(".", 1)[0] + ".*"))
        matches_b = [p for p in candidates if p.exists()]
        if matches_b:
            examples.append([str(path_a), str(matches_b[0])])
    return examples


def _verdict(result) -> tuple[str, str]:
    """Turn the numbers into a one-line judgement.

    The threshold logic mirrors what the RANSAC step can actually support:
    below ~10 inliers a homography is not identifiable, so anything under
    that is reported as "no reliable match" however many good matches the
    ratio test produced.
    """
    if result.n_good == 0:
        return "❌", "No matches at all. These images share no repeatable structure."
    if result.n_inliers < MIN_INLIERS_FOR_HOMOGRAPHY:
        return "❌", (f"{result.n_good} good matches, but only {result.n_inliers} of them "
                     "agree on a consistent geometry. Treat this as no match - "
                     "the survivors are coincidences on similar-looking texture.")
    if result.inlier_rate >= 80:
        return "✅", (f"Strong match. {result.n_inliers} of {result.n_good} good matches "
                     f"({result.inlier_rate:.0f}%) agree on one transform.")
    if result.inlier_rate >= 50:
        return "🟡", (f"Decent match. {result.n_inliers} of {result.n_good} "
                     f"({result.inlier_rate:.0f}%) are geometrically consistent - the rest "
                     "are noise, or the scene is not flat enough for a single homography.")
    return "🟠", (f"Weak match. Only {result.inlier_rate:.0f}% of the good matches survive "
                 "the geometry check, so most of them are wrong.")


def _stats_markdown(result, shape_a, shape_b) -> str:
    icon, verdict = _verdict(result)

    distances = [m.distance for m in result.good_matches]
    distance_row = ""
    if distances:
        distance_row = (
            f"| Hamming distance (best / mean / worst) | "
            f"{min(distances):.0f} / {sum(distances) / len(distances):.1f} / "
            f"{max(distances):.0f} | out of 256 bits |\n")

    notes = ""
    if result.notes:
        notes = "\n".join(f"- {note}" for note in result.notes)
        notes = f"\n**Notes**\n\n{notes}\n"

    return f"""
## {icon} {verdict}

| Measurement | Value | |
|---|---:|---|
| **Keypoints in image A** | **{result.count_a}** | {shape_a[1]} x {shape_a[0]} px |
| **Keypoints in image B** | **{result.count_b}** | {shape_b[1]} x {shape_b[0]} px |
| **Good matches** (Lowe ratio @ {result.ratio:.2f}) | **{result.n_good}** | survived the ratio test |
| **Verified matches** (RANSAC) | **{result.n_inliers}** | agree on one homography |
| Match rate | {result.match_rate:.1f}% | good / smaller keypoint set |
| Inlier rate | {result.inlier_rate:.1f}% | verified / good |
| Detection time | {result.detect_ms:.0f} ms | both images |
| Matching time | {result.match_ms:.0f} ms | brute force, Hamming |
{distance_row}
{notes}
"""


def analyse(image_a, image_b, n_features, ratio, max_draw, show_harris):
    """Main callback. Returns (matches, keypoints, homography, stats)."""
    if image_a is None or image_b is None:
        return None, None, None, "Upload two images, or pick one of the examples below."

    bgr_a = resize_max_side(rgb_to_bgr(np.asarray(image_a)), MAX_SIDE)
    bgr_b = resize_max_side(rgb_to_bgr(np.asarray(image_b)), MAX_SIDE)

    result = match_images(bgr_a, bgr_b, n_features=int(n_features), ratio=float(ratio))

    match_view = bgr_to_rgb(draw_matches(bgr_a, bgr_b, result, max_draw=int(max_draw)))

    # Keypoint view. Harris is offered alongside ORB purely for comparison -
    # only ORB's keypoints are ever used for the matching above.
    if show_harris:
        panel_a = draw_harris(bgr_a, detect_harris(bgr_a))
        panel_b = draw_harris(bgr_b, detect_harris(bgr_b))
    else:
        panel_a = draw_orb(bgr_a, detect_orb(bgr_a, n_features=int(n_features)))
        panel_b = draw_orb(bgr_b, detect_orb(bgr_b, n_features=int(n_features)))

    height = max(panel_a.shape[0], panel_b.shape[0])
    padded = []
    for panel in (panel_a, panel_b):
        pad = height - panel.shape[0]
        if pad:
            panel = np.vstack([panel, np.full((pad, panel.shape[1], 3), 32, np.uint8)])
        padded.append(panel)
    gap = np.full((height, 8, 3), 32, np.uint8)
    keypoint_view = bgr_to_rgb(np.hstack([padded[0], gap, padded[1]]))

    located = draw_detected_object(bgr_a, bgr_b, result)
    located_view = bgr_to_rgb(located) if located is not None else None

    return (match_view, keypoint_view, located_view,
            _stats_markdown(result, bgr_a.shape, bgr_b.shape))


DESCRIPTION = """
# 🔍 Image Feature Matching System

Finds the points two images have in common, using **ORB** keypoints matched
with a **Brute Force** matcher over Hamming distance.

Every match then has to survive two filters before it is counted:

1. **Lowe's ratio test** - keep a match only when the best candidate is clearly
   better than the second best. Without it, every keypoint gets a "match"
   whether or not it has a real counterpart.
2. **RANSAC** - of the survivors, keep the ones that agree on a single
   geometric transform between the two images.

The second number is the one to trust. A large pile of good matches with a low
inlier rate means the ratio test let coincidences through.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Image Feature Matching System",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            input_a = gr.Image(label="Image A", type="numpy", height=280)
            input_b = gr.Image(label="Image B", type="numpy", height=280)

        with gr.Accordion("Settings", open=False):
            with gr.Row():
                n_features = gr.Slider(100, 5000, value=1000, step=100,
                                       label="ORB keypoint budget",
                                       info="More keypoints finds more matches "
                                            "but costs time")
                ratio = gr.Slider(0.5, 0.95, value=0.75, step=0.05,
                                  label="Lowe ratio threshold",
                                  info="Lower = stricter, fewer but better matches")
            with gr.Row():
                max_draw = gr.Slider(10, 200, value=50, step=10,
                                     label="Match lines to draw",
                                     info="Display only - does not change the counts")
                show_harris = gr.Checkbox(
                    value=False, label="Show Harris corners instead of ORB keypoints",
                    info="Comparison only; matching always uses ORB")

        run_button = gr.Button("Find matching features", variant="primary", size="lg")

        stats = gr.Markdown()

        with gr.Tab("Matches"):
            out_matches = gr.Image(label="Matched keypoints", height=460)
        with gr.Tab("Keypoints"):
            out_keypoints = gr.Image(label="Detected keypoints in each image", height=460)
        with gr.Tab("Location"):
            out_located = gr.Image(
                label="Image A's border projected into image B (needs a reliable match)",
                height=460)

        inputs = [input_a, input_b, n_features, ratio, max_draw, show_harris]
        outputs = [out_matches, out_keypoints, out_located, stats]

        run_button.click(analyse, inputs=inputs, outputs=outputs)

        examples = _example_pairs()
        if examples:
            gr.Markdown("### Sample pairs - click one to load it, then press the button")
            gr.Examples(examples=examples, inputs=[input_a, input_b], examples_per_page=5)

        gr.Markdown(
            "Day 25 - ORB feature detection and matching with OpenCV. "
            "Images are downscaled to 1000 px on the long side before processing.")

    return demo


if __name__ == "__main__":
    build_ui().launch()
