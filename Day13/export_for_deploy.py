"""
Day 13 - Export the trained CNN for a TensorFlow-free deployment
=================================================================
WHY THIS FILE EXISTS

Training a network needs TensorFlow: backpropagation, gradients, Adam.
RUNNING a trained network does not. A forward pass through this CNN is
just convolution, max, matrix multiply and softmax - all of which NumPy
can do in a few lines.

That matters because importing TensorFlow costs ~500 MB of RAM, and
Streamlit Community Cloud's free tier only has 1 GB. So we export the
learned weights to a plain .npz and let the app do the arithmetic.

This script also PROVES the NumPy version matches TensorFlow, rather
than assuming it - it runs both over the same images and compares.

Run AFTER training:
    python 2_cnn_fashion_mnist.py
    python export_for_deploy.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist

from cnn_numpy import forward          # the TensorFlow-free forward pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "fashion_mnist_cnn.keras")


# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  EXPORTING THE CNN FOR DEPLOYMENT")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found: {MODEL_PATH}\n"
                         f"Run `python 2_cnn_fashion_mnist.py` first.")

    model = keras.models.load_model(MODEL_PATH)
    print(f"\nLoaded model: {MODEL_PATH}  ({model.count_params():,} params)")

    # --- 1. Pull the learned weights out ------------------------------
    c1w, c1b = model.get_layer("conv_1").get_weights()
    c2w, c2b = model.get_layer("conv_2").get_weights()
    c3w, c3b = model.get_layer("conv_3").get_weights()
    d1w, d1b = model.get_layer("dense_1").get_weights()
    ow, ob = model.get_layer("output_layer").get_weights()

    weights = {
        "conv1_W": c1w.astype("float32"), "conv1_b": c1b.astype("float32"),
        "conv2_W": c2w.astype("float32"), "conv2_b": c2b.astype("float32"),
        "conv3_W": c3w.astype("float32"), "conv3_b": c3b.astype("float32"),
        "dense1_W": d1w.astype("float32"), "dense1_b": d1b.astype("float32"),
        "out_W": ow.astype("float32"), "out_b": ob.astype("float32"),
    }

    weights_path = os.path.join(OUTPUTS_DIR, "cnn_weights.npz")
    np.savez_compressed(weights_path, **weights)
    print(f"Weights saved : {weights_path} "
          f"({os.path.getsize(weights_path)/1024:.0f} KB)")

    # --- 2. Test data + the model's answers ---------------------------
    (_, _), (X_test_raw, y_test) = fashion_mnist.load_data()
    X_test = np.expand_dims(X_test_raw.astype("float32") / 255.0, -1)

    tf_probs = model.predict(X_test, verbose=0)
    tf_labels = tf_probs.argmax(axis=1)
    tf_acc = (tf_labels == y_test).mean()
    print(f"\nTensorFlow test accuracy : {tf_acc*100:.2f}%")

    data_path = os.path.join(OUTPUTS_DIR, "test_data.npz")
    np.savez_compressed(
        data_path,
        images=X_test_raw.astype("uint8"),      # uint8 keeps the file small
        labels=y_test.astype("uint8"),
        predicted=tf_labels.astype("uint8"),
        confidence=tf_probs.max(axis=1).astype("float32"),
    )
    print(f"Test data saved: {data_path} "
          f"({os.path.getsize(data_path)/1024/1024:.1f} MB)")

    # --- 3. PROVE the NumPy version agrees ----------------------------
    print("\nVerifying the NumPy forward pass against TensorFlow...")
    n = 300
    np_probs = np.stack([forward(X_test[i], weights) for i in range(n)])
    max_diff = np.abs(np_probs - tf_probs[:n]).max()
    agreement = (np_probs.argmax(axis=1) == tf_labels[:n]).mean()

    print(f"  Compared {n} images:")
    print(f"    Largest probability difference : {max_diff:.3e}")
    print(f"    Predicted-label agreement      : {agreement*100:.4f}%")
    if max_diff < 1e-4 and agreement == 1.0:
        print("\n  MATCH. The app can run this CNN with NumPy alone -")
        print("  no TensorFlow, no 500 MB import, instant cold start.")
    else:
        raise SystemExit("  MISMATCH - do not deploy until this is fixed.")

    print("\n" + "=" * 70)
    print("  Next:  streamlit run app.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
