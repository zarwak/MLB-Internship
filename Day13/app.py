"""
Day 13 - CNN X-Ray: watch the trained CNN look at an image
===========================================================
An interactive Streamlit app that runs the trained Fashion MNIST CNN on
one image at a time and shows what happens INSIDE it - the feature maps
after every convolution and pooling layer, then the final 10 probabilities.

It uses NO TensorFlow. The weights were exported to a .npz and the
forward pass is plain NumPy (see cnn_numpy.py), so this app installs in
seconds and starts instantly.

It also uses no matplotlib - the viridis colormap below is applied with
NumPy alone, which keeps the deployment at streamlit + numpy.

Run with:
    streamlit run app.py
"""

import os

import numpy as np
import streamlit as st

from cnn_numpy import CLASS_NAMES, forward, load_weights

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

ACCENT = "#0ea5e9"
GREEN = "#16a34a"
RED = "#dc2626"
MUTED = "#94a3b8"

st.set_page_config(page_title="CNN X-Ray - Fashion MNIST", layout="wide")


# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 1400px; }

      .xr-title { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em;
                  margin: 0 0 .2rem 0; }
      .xr-sub   { color: #94a3b8; font-size: .95rem; margin: 0 0 1.4rem 0; }

      /* stat tiles */
      .xr-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                  gap: .75rem; margin-bottom: 1.6rem; }
      .xr-stat  { border: 1px solid rgba(148,163,184,.25); border-radius: 10px;
                  padding: .8rem 1rem; background: rgba(148,163,184,.06); }
      .xr-stat .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
                    color: #94a3b8; margin-bottom: .3rem; }
      .xr-stat .v { font-size: 1.5rem; font-weight: 700; line-height: 1.1;
                    font-variant-numeric: tabular-nums; }
      .xr-stat .d { font-size: .75rem; color: #16a34a; margin-top: .15rem; }

      /* verdict */
      .xr-verdict { border-radius: 10px; padding: .85rem 1rem; margin-top: .9rem;
                    border: 1px solid; font-size: .95rem; }
      .xr-ok  { border-color: rgba(22,163,74,.45);  background: rgba(22,163,74,.10); }
      .xr-bad { border-color: rgba(220,38,38,.45);  background: rgba(220,38,38,.10); }
      .xr-verdict b { font-weight: 700; }

      /* probability rows */
      .xr-row   { display: grid; grid-template-columns: 108px 1fr 62px;
                  align-items: center; gap: .6rem; margin-bottom: .3rem; }
      .xr-name  { font-size: .84rem; color: #cbd5e1; white-space: nowrap;
                  overflow: hidden; text-overflow: ellipsis; }
      .xr-track { height: 16px; border-radius: 4px; background: rgba(148,163,184,.16);
                  overflow: hidden; }
      .xr-fill  { height: 100%; border-radius: 4px; }
      .xr-val   { font-size: .8rem; text-align: right; color: #94a3b8;
                  font-variant-numeric: tabular-nums; }
      .xr-row.is-key .xr-name { color: #f1f5f9; font-weight: 600; }

      /* layer header */
      .xr-layer { display: flex; align-items: baseline; gap: .7rem; flex-wrap: wrap;
                  border-left: 3px solid #0ea5e9; padding: .1rem 0 .1rem .7rem;
                  margin: 1.5rem 0 .7rem 0; }
      .xr-layer .n { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                     font-size: 1rem; font-weight: 700; }
      .xr-layer .s { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                     font-size: .78rem; color: #0ea5e9; }
      .xr-layer .t { font-size: .82rem; color: #94a3b8; }

      .xr-legend { font-size: .8rem; color: #94a3b8; margin-top: .5rem; }
      .xr-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                   margin-right: .3rem; vertical-align: middle; }

      @media (prefers-color-scheme: light) {
        .xr-name { color: #475569; }
        .xr-row.is-key .xr-name { color: #0f172a; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Viridis colormap, in NumPy - no matplotlib needed
# ----------------------------------------------------------------------
_VIRIDIS_STOPS = [
    "#440154", "#481a6c", "#472f7d", "#414487", "#39568c", "#31688e",
    "#2a788e", "#23888e", "#1f988b", "#22a884", "#35b779", "#54c568",
    "#7ad151", "#a5db36", "#d2e21b", "#fde725",
]


@st.cache_resource
def viridis_lut():
    """Build a 256-entry RGB lookup table by interpolating the stops above."""
    stops = np.array(
        [[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in _VIRIDIS_STOPS],
        dtype=np.float32,
    )
    src = np.linspace(0.0, 1.0, len(stops))
    dst = np.linspace(0.0, 1.0, 256)
    return np.stack(
        [np.interp(dst, src, stops[:, c]) for c in range(3)], axis=1
    ).astype(np.uint8)


def colorize(m, lut, scale=1):
    """Normalize a 2D feature map to 0-1, map it through the LUT, upscale."""
    rng = float(m.max() - m.min())
    norm = (m - m.min()) / rng if rng > 1e-8 else np.zeros_like(m)
    rgb = lut[(norm * 255).astype(np.uint8)]
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb


def upscale_gray(img_u8, scale):
    """Nearest-neighbour upscale so a 28x28 image stays crisp, not blurry."""
    return np.repeat(np.repeat(img_u8, scale, axis=0), scale, axis=1)


def pct(p):
    """Format a probability honestly.

    Softmax almost never returns exactly 0 or 1, so plain "%.1f%%" prints a
    confident-looking "100.0%" or a useless "0.0%". Clamp both ends instead.
    """
    if p > 0.9995:
        return "over 99.9%"
    if p < 0.001:
        return "under 0.1%"
    return f"{p:.1%}"


# ----------------------------------------------------------------------
# Load the exported model + data once, then keep it in memory
# ----------------------------------------------------------------------
@st.cache_resource
def get_weights():
    return load_weights(os.path.join(OUTPUTS_DIR, "cnn_weights.npz"))


@st.cache_data
def get_data():
    with np.load(os.path.join(OUTPUTS_DIR, "test_data.npz")) as z:
        return z["images"], z["labels"], z["predicted"], z["confidence"]


try:
    W = get_weights()
    images, labels, predicted, confidence = get_data()
except FileNotFoundError:
    st.error(
        "Exported files not found. Run these two first:\n\n"
        "```\npython 2_cnn_fashion_mnist.py\npython export_for_deploy.py\n```"
    )
    st.stop()

LUT = viridis_lut()
ACCURACY = (predicted == labels).mean()
N_WRONG = int((predicted != labels).sum())


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="xr-title">CNN X-Ray &mdash; Fashion MNIST</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="xr-sub">Day 13 &middot; A Convolutional Neural Network, opened up. '
    'Pick an image and watch it travel through every layer.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="xr-stats">
      <div class="xr-stat"><div class="k">Test accuracy</div>
        <div class="v">{ACCURACY*100:.2f}%</div>
        <div class="d">{ACCURACY*100 - 86.31:+.2f} pts vs Day 12 ANN</div></div>
      <div class="xr-stat"><div class="k">Test images</div>
        <div class="v">{len(labels):,}</div></div>
      <div class="xr-stat"><div class="k">Got wrong</div>
        <div class="v">{N_WRONG:,}</div></div>
      <div class="xr-stat"><div class="k">Parameters</div>
        <div class="v">458,570</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.subheader("Choose an image")

    pool_choice = st.radio(
        "Image pool",
        ["All test images", "Only the ones it got wrong",
         "Only the ones it got right", "One class"],
    )

    if pool_choice == "Only the ones it got wrong":
        pool = np.where(predicted != labels)[0]
    elif pool_choice == "Only the ones it got right":
        pool = np.where(predicted == labels)[0]
    elif pool_choice == "One class":
        chosen = st.selectbox("Class", CLASS_NAMES)
        pool = np.where(labels == CLASS_NAMES.index(chosen))[0]
    else:
        pool = np.arange(len(labels))

    st.caption(f"{len(pool):,} images in this pool")

    if "idx" not in st.session_state or st.session_state.idx not in pool:
        st.session_state.idx = int(pool[0])

    if st.button("Pick a random image", use_container_width=True, type="primary"):
        st.session_state.idx = int(np.random.choice(pool))

    position = st.slider(
        "Or step through the pool", 0, max(len(pool) - 1, 0),
        int(np.searchsorted(pool, st.session_state.idx).clip(0, len(pool) - 1)),
    )
    if st.button("Use this position", use_container_width=True):
        st.session_state.idx = int(pool[position])

    st.divider()
    st.subheader("Display")
    n_maps = st.slider("Feature maps per layer", 4, 16, 8, step=4)
    show_stage = st.multiselect(
        "Layers to show",
        ["conv_1", "pool_1", "conv_2", "pool_2", "conv_3"],
        default=["conv_1", "pool_1", "conv_2", "pool_2"],
    )

idx = int(st.session_state.idx)


# ----------------------------------------------------------------------
# Run the CNN on the chosen image
# ----------------------------------------------------------------------
image_u8 = images[idx]
image = (image_u8.astype("float32") / 255.0)[..., None]     # (28, 28, 1)
probs, maps = forward(image, W, return_maps=True)

pred = int(probs.argmax())
actual = int(labels[idx])
correct = pred == actual

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.markdown("##### The input")
    st.image(upscale_gray(image_u8, 9), width=288, clamp=True)
    st.caption(f"True label: **{CLASS_NAMES[actual]}** &nbsp;·&nbsp; test image #{idx}")

    if correct:
        st.markdown(
            f'<div class="xr-verdict xr-ok">Predicted <b>{CLASS_NAMES[pred]}</b> &mdash; '
            f'correct, {pct(probs[pred])} confident.</div>',
            unsafe_allow_html=True,
        )
    else:
        rank = int((np.argsort(probs)[::-1] == actual).argmax()) + 1
        st.markdown(
            f'<div class="xr-verdict xr-bad">Predicted <b>{CLASS_NAMES[pred]}</b> at '
            f'{pct(probs[pred])}. The answer was <b>{CLASS_NAMES[actual]}</b>, which it '
            f'ranked #{rank} at {pct(probs[actual])}.</div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown("##### The 10 output probabilities")

    rows = []
    for i in np.argsort(probs)[::-1]:
        if i == pred:
            colour = GREEN if correct else RED
        elif i == actual:
            colour = ACCENT
        else:
            colour = MUTED
        key = " is-key" if i in (pred, actual) else ""
        width = max(float(probs[i]) * 100.0, 0.6)
        rows.append(
            f'<div class="xr-row{key}">'
            f'<div class="xr-name">{CLASS_NAMES[i]}</div>'
            f'<div class="xr-track"><div class="xr-fill" style="width:{width:.2f}%;'
            f'background:{colour};"></div></div>'
            f'<div class="xr-val">{probs[i]:.4f}</div></div>'
        )

    legend = (
        f'<div class="xr-legend">'
        f'<span class="xr-swatch" style="background:{GREEN}"></span>correct prediction'
        f'&nbsp;&nbsp;<span class="xr-swatch" style="background:{RED}"></span>wrong prediction'
        f'&nbsp;&nbsp;<span class="xr-swatch" style="background:{ACCENT}"></span>'
        f'the answer it should have given</div>'
    )
    st.markdown("".join(rows) + legend, unsafe_allow_html=True)

st.divider()


# ----------------------------------------------------------------------
# The feature maps - the actual point of this app
# ----------------------------------------------------------------------
st.markdown("##### Inside the network &mdash; feature maps for this image")
st.caption(
    "Each square is one filter's answer to the question \"where did I find my pattern?\". "
    "Bright means found strongly. Early layers still look like the garment, because they "
    "are finding edges. Deeper layers look abstract, because they have stopped drawing "
    "the object and started summarising it."
)

STAGE_NOTE = {
    "conv_1": "32 filters find simple edges and blobs",
    "pool_1": "2x2 max pooling halves it, nothing to learn here",
    "conv_2": "64 filters combine edges into shapes",
    "pool_2": "halved again",
    "conv_3": "64 filters, the most abstract patterns",
}

for stage in ["conv_1", "pool_1", "conv_2", "pool_2", "conv_3"]:
    if stage not in show_stage:
        continue
    fmap = maps[stage]
    h, w, c = fmap.shape
    st.markdown(
        f'<div class="xr-layer"><span class="n">{stage}</span>'
        f'<span class="s">{h} x {w} x {c}</span>'
        f'<span class="t">{STAGE_NOTE[stage]}</span></div>',
        unsafe_allow_html=True,
    )
    scale = max(1, 112 // h)
    cols = st.columns(n_maps)
    for k in range(n_maps):
        cols[k].image(
            colorize(fmap[:, :, k], LUT, scale),
            caption=f"filter {k}",
            use_container_width=True,
        )

st.divider()

with st.expander("How this app runs a CNN without TensorFlow"):
    st.markdown(
        """
`2_cnn_fashion_mnist.py` trains the model with TensorFlow and saves it.
`export_for_deploy.py` then pulls out just the **learned numbers** and
saves them to a small `.npz`.

Running a trained CNN is only four kinds of arithmetic:

```
convolution   ->  multiply a 3x3 window by 9 learned numbers and add up
ReLU          ->  max(0, x)
max pooling   ->  take the biggest value in each 2x2 block
dense+softmax ->  one matrix multiply, then e^x / sum(e^x)
```

NumPy does all of that. TensorFlow is only needed to **train** a model - it
is never needed to **run** one. That cuts this deployment from ~600 MB of
dependencies to ~30 MB, and the export script verifies the NumPy answers
match TensorFlow's to about 1e-6 before shipping.

Even the colour on the feature maps above is NumPy: the viridis palette is
16 hex stops interpolated into a 256-entry lookup table, so matplotlib
stays out of the deployment too.
        """
    )
