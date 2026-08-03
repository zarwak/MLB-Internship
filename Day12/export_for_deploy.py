"""
Day 12 - Export the trained model for deployment
=================================================

WHY THIS EXISTS
---------------
The simulation app only ever runs a FORWARD PASS. For our network that is
four operations:

    flatten  ->  (x @ W1 + b1) -> ReLU
             ->  (x @ W2 + b2) -> ReLU
             ->  (x @ W3 + b3) -> softmax

That is plain matrix arithmetic. NumPy does it perfectly well. TensorFlow
is only genuinely needed for TRAINING - for backpropagation, gradients and
the optimizer.

Carrying TensorFlow into deployment costs ~500 MB of RAM just to import,
which does not fit comfortably in Streamlit Community Cloud's 1 GB limit,
and TF has no wheels at all for Python 3.14. So instead we export:

    1. the learned weights, as a plain .npz
    2. the test images, as a plain .npz

and the deployed app runs on NumPy alone.

Run this AFTER 4_ann_fashion_mnist.py:
    python export_for_deploy.py
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "fashion_mnist_ann.keras")
WEIGHTS_PATH = os.path.join(OUTPUTS_DIR, "model_weights.npz")
DATA_PATH = os.path.join(OUTPUTS_DIR, "test_data.npz")


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ----------------------------------------------------------------------
# 1. Export the learned weights
# ----------------------------------------------------------------------
print_header("EXPORTING MODEL WEIGHTS")

if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        f"No trained model at {MODEL_PATH}.\n"
        "Run `python 4_ann_fashion_mnist.py` first."
    )

model = keras.models.load_model(MODEL_PATH)
print(f"Loaded model with {model.count_params():,} parameters.")

arrays = {}
dense_i = 0
for layer in model.layers:
    weights = layer.get_weights()
    if not weights:                      # Flatten has nothing to save
        print(f"  {layer.name:<16} no weights (skipped)")
        continue
    W, b = weights
    arrays[f"W{dense_i}"] = W.astype("float32")
    arrays[f"b{dense_i}"] = b.astype("float32")
    print(f"  {layer.name:<16} W{dense_i} {str(W.shape):<12} b{dense_i} {str(b.shape)}")
    dense_i += 1

arrays["n_dense"] = np.array(dense_i)
np.savez_compressed(WEIGHTS_PATH, **arrays)

size_kb = os.path.getsize(WEIGHTS_PATH) / 1024
print(f"\nSaved {WEIGHTS_PATH}")
print(f"  {dense_i} Dense layers, {size_kb:,.0f} KB")


# ----------------------------------------------------------------------
# 2. Export the test images
# ----------------------------------------------------------------------
print_header("EXPORTING TEST DATA")

(_, _), (X_test, y_test) = fashion_mnist.load_data()

# Keep them as uint8 (0-255). Storing float32 would be 4x bigger for no
# gain - the app divides by 255 when it loads them.
np.savez_compressed(DATA_PATH,
                    X_test=X_test.astype("uint8"),
                    y_test=y_test.astype("uint8"))

size_mb = os.path.getsize(DATA_PATH) / (1024 * 1024)
print(f"Saved {DATA_PATH}")
print(f"  {X_test.shape[0]:,} images, {size_mb:.2f} MB")


# ----------------------------------------------------------------------
# 3. Verify: NumPy must reproduce TensorFlow exactly
# ----------------------------------------------------------------------
print_header("VERIFYING NUMPY MATCHES TENSORFLOW")


def numpy_forward(images, arrays):
    """The same forward pass the deployed app will run."""
    n = int(arrays["n_dense"])
    x = images.reshape(len(images), -1)          # Flatten
    for i in range(n):
        x = x @ arrays[f"W{i}"] + arrays[f"b{i}"]
        if i < n - 1:
            x = np.maximum(0.0, x)               # ReLU on hidden layers
        else:
            e = np.exp(x - x.max(axis=1, keepdims=True))   # Softmax on output
            x = e / e.sum(axis=1, keepdims=True)
    return x


loaded = dict(np.load(WEIGHTS_PATH))
sample = (X_test[:500].astype("float32") / 255.0)

tf_probs = model.predict(sample, verbose=0)
np_probs = numpy_forward(sample, loaded)

max_diff = float(np.abs(tf_probs - np_probs).max())
tf_labels = tf_probs.argmax(axis=1)
np_labels = np_probs.argmax(axis=1)
agreement = float((tf_labels == np_labels).mean())

print(f"Compared {len(sample)} images:")
print(f"  Largest probability difference : {max_diff:.3e}")
print(f"  Predicted-label agreement      : {agreement:.4%}")

if agreement == 1.0 and max_diff < 1e-5:
    print("\n  PASS - NumPy reproduces TensorFlow exactly.")
    print("  The deployed app will give identical answers without TF installed.")
else:
    raise SystemExit("\n  FAIL - outputs diverge. Do not deploy.")

print("\n" + "=" * 70)
print("  Ready to deploy. The app needs only: streamlit, numpy")
print("=" * 70 + "\n")
