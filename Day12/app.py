"""
Day 12 - ANN Simulation UI
===========================
An interactive Streamlit app that shows the trained Fashion MNIST ANN
actually running: an image goes in, you watch it flow through each layer,
and the 10 output probabilities fill in live.

NO TENSORFLOW REQUIRED
----------------------
This app only performs a FORWARD PASS, which for our network is four
plain matrix operations:

    flatten -> (x @ W1 + b1) -> ReLU
            -> (x @ W2 + b2) -> ReLU
            -> (x @ W3 + b3) -> softmax

TensorFlow is needed to TRAIN the model (backpropagation, gradients, the
optimizer) but not to RUN it. So the app loads exported weights and does
the arithmetic in NumPy. This keeps the deployment tiny and fast, and
sidesteps the fact that TensorFlow has no wheels for newer Python versions.

Verified identical to the TensorFlow model: 100% label agreement,
max probability difference 1.2e-06. See export_for_deploy.py.

Run with:
    streamlit run app.py

Requires outputs/model_weights.npz and outputs/test_data.npz, both
produced by export_for_deploy.py.
"""

import os
import time

import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "outputs", "model_weights.npz")
DATA_PATH = os.path.join(BASE_DIR, "outputs", "test_data.npz")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

st.set_page_config(page_title="ANN Simulation - Fashion MNIST",
                   page_icon="👕", layout="wide")


# ======================================================================
# Styling
# ======================================================================
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .prob-row {
        display: flex; align-items: center;
        margin-bottom: 6px; font-size: 0.85rem;
    }
    .prob-label {
        width: 100px; flex: none; text-align: right; padding-right: 10px;
        font-family: ui-monospace, monospace; opacity: 0.85;
    }
    .prob-track {
        flex: 1 1 auto; min-width: 0; height: 26px;
        background: rgba(128,128,128,0.15);
        border-radius: 4px; overflow: hidden; position: relative;
    }
    .prob-fill {
        height: 100%; border-radius: 4px;
        transition: width 0.05s linear;
    }
    .prob-value {
        width: 68px; flex: none; padding-left: 10px;
        font-family: ui-monospace, monospace; font-size: 0.8rem;
    }
    .verdict {
        padding: 14px 18px; border-radius: 8px;
        font-size: 1.05rem; font-weight: 600; margin-top: 8px;
    }
    .verdict-ok   { background: rgba(34,197,94,0.15);  border-left: 5px solid #22c55e; }
    .verdict-bad  { background: rgba(239,68,68,0.15);  border-left: 5px solid #ef4444; }

    .layer-card {
        border: 1px solid rgba(128,128,128,0.25); border-radius: 8px;
        padding: 10px 14px; margin-bottom: 8px;
    }
    .layer-title {
        font-size: 0.78rem; text-transform: uppercase;
        letter-spacing: 0.05em; opacity: 0.7; margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# Loading (cached so it happens once, not on every interaction)
# ======================================================================
@st.cache_resource(show_spinner="Loading model weights...")
def load_weights():
    if not os.path.exists(WEIGHTS_PATH):
        return None
    data = dict(np.load(WEIGHTS_PATH))
    n = int(data["n_dense"])
    return {
        "n": n,
        "W": [data[f"W{i}"] for i in range(n)],
        "b": [data[f"b{i}"] for i in range(n)],
        "params": sum(int(data[f"W{i}"].size + data[f"b{i}"].size) for i in range(n)),
    }


@st.cache_data(show_spinner="Loading Fashion MNIST test set...")
def load_test_data():
    if not os.path.exists(DATA_PATH):
        return None, None
    d = np.load(DATA_PATH)
    return d["X_test"].astype("float32") / 255.0, d["y_test"].astype("int32")


def forward_pass(w, images):
    """
    Push images through the network, keeping every intermediate
    activation. This is what makes the simulation a simulation rather
    than just a prediction.

    Returns a list: [flattened, hidden1, hidden2, ..., probabilities]
    """
    x = images.reshape(len(images), -1)          # Flatten
    activations = [x]
    for i in range(w["n"]):
        x = x @ w["W"][i] + w["b"][i]            # Dense: weights + bias
        if i < w["n"] - 1:
            x = np.maximum(0.0, x)               # ReLU on hidden layers
        else:
            e = np.exp(x - x.max(axis=1, keepdims=True))   # Softmax on output
            x = e / e.sum(axis=1, keepdims=True)
        activations.append(x)
    return activations


@st.cache_data(show_spinner="Running the model over all 10,000 test images...")
def predict_all(_w, X_test):
    """Predict once up front so filtering by 'correct/wrong' is instant."""
    return forward_pass(_w, X_test)[-1]


# ======================================================================
# Header
# ======================================================================
st.title("🧠 ANN Simulation — Watch the Network Think")
st.caption(
    "A trained Artificial Neural Network classifying Fashion MNIST images. "
    "Each image is flattened to 784 numbers, passed through two hidden layers, "
    "and turned into 10 probabilities."
)

weights = load_weights()
X_test, y_test = load_test_data()

if weights is None or X_test is None:
    st.error(
        "Exported model files not found.\n\n"
        f"Expected `{WEIGHTS_PATH}` and `{DATA_PATH}`.\n\n"
        "Generate them by running, in order:\n\n"
        "```\npython 4_ann_fashion_mnist.py\npython export_for_deploy.py\n```"
    )
    st.stop()

all_preds = predict_all(weights, X_test)
all_labels = all_preds.argmax(axis=1)


# ======================================================================
# Sidebar controls
# ======================================================================
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("Which images?")
    pool_choice = st.radio(
        "Image pool",
        ["All test images", "Only ones it got WRONG", "Only ones it got RIGHT",
         "Pick a specific class"],
        label_visibility="collapsed",
    )

    if pool_choice == "All test images":
        pool = np.arange(len(X_test))
    elif pool_choice == "Only ones it got WRONG":
        pool = np.where(all_labels != y_test)[0]
    elif pool_choice == "Only ones it got RIGHT":
        pool = np.where(all_labels == y_test)[0]
    else:
        chosen = st.selectbox("Class", CLASS_NAMES, index=6)
        pool = np.where(y_test == CLASS_NAMES.index(chosen))[0]

    st.caption(f"**{len(pool):,}** images in this pool")

    st.divider()
    st.subheader("Animation")
    animate = st.toggle("Animate the probability bars", value=True)
    speed = st.select_slider(
        "Speed", options=["Slow", "Medium", "Fast"], value="Medium",
        disabled=not animate,
    )
    frame_delay = {"Slow": 0.055, "Medium": 0.025, "Fast": 0.008}[speed]

    st.divider()
    st.subheader("Auto-play")
    autoplay_n = st.number_input("Images to run through", 2, 50, 10)
    autoplay = st.button("▶️  Run simulation", use_container_width=True,
                         type="primary")

    st.divider()
    if st.button("🔄 Reset score", use_container_width=True):
        st.session_state.seen = 0
        st.session_state.correct = 0
        st.session_state.history = []
        st.rerun()

    st.divider()
    overall = float((all_labels == y_test).mean())
    st.caption(
        f"Parameters: **{weights['params']:,}**  \n"
        f"Test accuracy: **{overall:.2%}**  \n"
        f"Inference: NumPy (no TensorFlow needed)"
    )


# ======================================================================
# Session state
# ======================================================================
if "seen" not in st.session_state:
    st.session_state.seen = 0
    st.session_state.correct = 0
    st.session_state.history = []
    st.session_state.idx = int(pool[0]) if len(pool) else 0

if len(pool) == 0:
    st.warning("No images match that filter.")
    st.stop()


# ======================================================================
# Rendering helpers
# ======================================================================
def probability_bars(probs, true_label, reveal=1.0):
    """
    Build the HTML for the 10 probability bars.
    `reveal` scales every bar, so animating it 0 -> 1 makes them fill in.
    """
    top = int(np.argmax(probs))
    rows = []
    for i, p in enumerate(probs):
        shown = p * reveal
        if i == top and i == true_label:
            color = "#22c55e"                     # right answer, and it won
        elif i == top:
            color = "#ef4444"                     # wrong answer, but it won
        elif i == true_label:
            color = "#3b82f6"                     # the correct answer it missed
        else:
            color = "rgba(148,163,184,0.55)"
        width = max(shown * 100, 0.4)
        rows.append(
            f'<div class="prob-row">'
            f'<div class="prob-label">{CLASS_NAMES[i]}</div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width:{width:.2f}%;background:{color};"></div>'
            f'</div>'
            f'<div class="prob-value">{shown:.4f}</div>'
            f'</div>'
        )
    return "".join(rows)


def activation_strip(values, max_show=64):
    """Render a layer's activations as a row of intensity blocks."""
    vals = values[:max_show]
    hi = float(vals.max()) if vals.max() > 0 else 1.0
    blocks = []
    for v in vals:
        alpha = float(v) / hi
        blocks.append(
            f'<span style="display:inline-block;width:11px;height:26px;'
            f'margin-right:2px;border-radius:2px;'
            f'background:rgba(14,165,233,{alpha:.3f});"></span>'
        )
    extra = f' <span style="opacity:0.55;font-size:0.75rem;">+{len(values)-max_show} more</span>' \
            if len(values) > max_show else ""
    return "".join(blocks) + extra


def render_frame(idx, reveal, placeholders):
    """Draw one frame of the simulation for test image `idx`."""
    img_ph, layers_ph, bars_ph, verdict_ph = placeholders

    probs = all_preds[idx]
    pred = int(all_labels[idx])
    true = int(y_test[idx])
    correct = pred == true

    with img_ph.container():
        st.markdown("##### 1️⃣  Input image")
        st.image(X_test[idx], width=230, clamp=True)
        st.caption(f"Test image #{idx}  ·  28 × 28 greyscale")
        st.markdown(f"**Actual label:** {CLASS_NAMES[true]}")

    with layers_ph.container():
        st.markdown("##### 2️⃣  Signal through the layers")
        acts = [a[0] for a in forward_pass(weights, X_test[idx][np.newaxis, ...])]

        st.markdown(
            f'<div class="layer-card">'
            f'<div class="layer-title">Flatten → 784 values (first 64)</div>'
            f'{activation_strip(acts[0])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="layer-card">'
            f'<div class="layer-title">Hidden layer 1 → 128 ReLU neurons '
            f'({int((acts[1] > 0).sum())} firing)</div>'
            f'{activation_strip(acts[1])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="layer-card">'
            f'<div class="layer-title">Hidden layer 2 → 64 ReLU neurons '
            f'({int((acts[2] > 0).sum())} firing)</div>'
            f'{activation_strip(acts[2])}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Brighter = stronger activation. A dark block is a neuron that "
            "output exactly 0 — ReLU switched it off for this image."
        )

    with bars_ph.container():
        st.markdown("##### 3️⃣  Output layer — 10 softmax probabilities")
        st.markdown(probability_bars(probs, true, reveal), unsafe_allow_html=True)

    if reveal >= 1.0:
        with verdict_ph.container():
            if correct:
                st.markdown(
                    f'<div class="verdict verdict-ok">✅ CORRECT — '
                    f'predicted <b>{CLASS_NAMES[pred]}</b> '
                    f'with {probs[pred]:.1%} confidence</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="verdict verdict-bad">❌ WRONG — predicted '
                    f'<b>{CLASS_NAMES[pred]}</b> ({probs[pred]:.1%}) but it was '
                    f'really <b>{CLASS_NAMES[true]}</b> '
                    f'({probs[true]:.1%})</div>',
                    unsafe_allow_html=True,
                )
    else:
        verdict_ph.empty()

    return correct


def run_one(idx, placeholders, do_animate):
    """Animate the bars filling in, then record the result."""
    if do_animate:
        for reveal in np.linspace(0.0, 1.0, 22):
            render_frame(idx, float(reveal), placeholders)
            time.sleep(frame_delay)
    correct = render_frame(idx, 1.0, placeholders)

    st.session_state.seen += 1
    st.session_state.correct += int(correct)
    st.session_state.history.append(
        {"idx": int(idx), "pred": int(all_labels[idx]),
         "true": int(y_test[idx]), "correct": bool(correct)}
    )
    paint_scoreboard()          # keep the counters live during auto-play
    return correct


# ======================================================================
# Scoreboard
# ======================================================================
# The scoreboard sits above the simulation but must reflect results
# produced BELOW it, so it is drawn into placeholders and repainted after
# every image. Without this it would always be one run out of date.
score_cols = st.columns(4)
score_slots = [c.empty() for c in score_cols]


def paint_scoreboard():
    seen = st.session_state.seen
    correct = st.session_state.correct
    acc = (correct / seen * 100) if seen else 0.0
    score_slots[0].metric("Images processed", f"{seen:,}")
    score_slots[1].metric("Correct", f"{correct:,}")
    score_slots[2].metric("Wrong", f"{seen - correct:,}")
    score_slots[3].metric("Live accuracy", f"{acc:.1f}%")


paint_scoreboard()
st.divider()


# ======================================================================
# Image picker
# ======================================================================
pick_col, btn_col = st.columns([3, 1])
with pick_col:
    position = st.slider(
        "Position in the selected pool", 0, max(len(pool) - 1, 0),
        0, key="position",
    )
with btn_col:
    st.write("")
    next_clicked = st.button("🎲  Random image", use_container_width=True)

if next_clicked:
    st.session_state.idx = int(np.random.choice(pool))
else:
    st.session_state.idx = int(pool[position])

st.divider()


# ======================================================================
# The simulation panels
# ======================================================================
# Row 1: the input image beside the layer activations.
# Row 2: the probability bars get the FULL page width - they are the
# part worth looking at, and cramming them into a third of the screen
# squashes the bars into nothing.
top_left, top_right = st.columns([1, 1.6])
bars_container = st.container()
verdict_container = st.container()

placeholders = (
    top_left.empty(),
    top_right.empty(),
    bars_container.empty(),
    verdict_container.empty(),
)

if autoplay:
    chosen = np.random.choice(pool, size=min(autoplay_n, len(pool)),
                              replace=False)
    progress = st.progress(0.0, text="Running simulation...")
    for n, idx in enumerate(chosen, start=1):
        run_one(int(idx), placeholders, animate)
        progress.progress(n / len(chosen),
                          text=f"Running simulation... {n}/{len(chosen)}")
        time.sleep(0.35)
    progress.empty()
    st.success(f"Ran {len(chosen)} images. See the scoreboard above.")
else:
    render_frame(st.session_state.idx, 1.0, placeholders)


# ======================================================================
# History
# ======================================================================
if st.session_state.history:
    st.divider()
    st.markdown("##### 📜 Recent predictions")
    recent = st.session_state.history[-12:][::-1]
    st.caption("Most recent first. Tick = correct, cross = wrong.")

    # Lay out 6 per row so the thumbnails stay a readable size.
    PER_ROW = 6
    for start in range(0, len(recent), PER_ROW):
        chunk = recent[start:start + PER_ROW]
        cols = st.columns(PER_ROW)
        for col, h in zip(cols, chunk):
            with col:
                st.image(X_test[h["idx"]], width=78, clamp=True)
                mark = "✅" if h["correct"] else "❌"
                if h["correct"]:
                    st.caption(f"{mark} {CLASS_NAMES[h['pred']]}")
                else:
                    st.caption(
                        f"{mark} said **{CLASS_NAMES[h['pred']]}**  \n"
                        f"was {CLASS_NAMES[h['true']]}"
                    )


# ======================================================================
# Explainer
# ======================================================================
with st.expander("ℹ️  What am I actually looking at?"):
    st.markdown(f"""
**The three panels follow one image through the network.**

1. **Input image** — a 28 × 28 greyscale picture. To the model this is just
   784 numbers between 0.0 and 1.0.

2. **Signal through the layers** — the actual activations for *this* image,
   computed live. `Flatten` unrolls the grid into 784 values. Hidden layer 1
   compresses that to 128 numbers, hidden layer 2 to 64. Brighter blocks are
   more strongly activated neurons; completely dark blocks are neurons where
   ReLU output exactly zero. The "firing" count changes with every image —
   that is the network responding to different visual features.

3. **Output probabilities** — 10 numbers from softmax that always sum to 1.0.
   The tallest bar is the prediction.
   🟢 green = predicted correctly · 🔴 red = predicted wrongly ·
   🔵 blue = the answer it *should* have given.

**Try this:** set the pool to *Only ones it got WRONG* and step through. Almost
every failure is between **T-shirt, Pullover, Coat and Shirt** — four upper-body
garments that look nearly identical at 28 × 28. Watch the blue bar sitting right
next to the red one: the model was often a hair away from being right.

---

**Model:** Flatten(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax),
{weights['params']:,} parameters, **{overall:.2%}** accuracy on the full
10,000-image test set.

**A note on how this runs:** the model was *trained* with TensorFlow/Keras, but
this app runs it with **NumPy alone**. A forward pass is just
`x @ W + b`, ReLU, and softmax — four matrix operations. TensorFlow is needed
for backpropagation and gradient descent during training, not for inference.
Exporting the weights keeps the deployed app small and fast. It was verified
against the TensorFlow model: **100% identical predictions**, largest
probability difference 1.2 × 10⁻⁶.
""")
