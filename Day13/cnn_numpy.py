"""
The trained CNN's forward pass, in pure NumPy.
==============================================
No TensorFlow anywhere in this file - and that is the whole point.

Training a network needs TensorFlow (backpropagation, gradients, Adam).
RUNNING one does not: a forward pass is convolution, max, matrix
multiply and softmax. Every one of those is a NumPy one-liner.

Used by both `export_for_deploy.py` (to verify) and `app.py` (to run).
"""

import numpy as np

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def conv2d_same(x, W, b):
    """2D convolution with padding='same'.

    x : (H, W, C_in)              one image, or one stack of feature maps
    W : (kh, kw, C_in, C_out)     the learned filters
    b : (C_out,)

    'same' padding adds a ring of zeros around the input so the output
    keeps the same height and width as the input.
    """
    kh, kw, c_in, c_out = W.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((ph, ph), (pw, pw), (0, 0)))
    H, Wd = x.shape[0], x.shape[1]

    # im2col: lay every kh x kw window out as a row, so the whole
    # convolution collapses into one matrix multiply. Identical result
    # to a slow nested loop, far faster.
    cols = np.empty((H * Wd, kh * kw * c_in), dtype=np.float32)
    i = 0
    for r in range(H):
        for c in range(Wd):
            cols[i] = xp[r:r + kh, c:c + kw, :].ravel()
            i += 1
    return (cols @ W.reshape(-1, c_out) + b).reshape(H, Wd, c_out)


def relu(x):
    """max(0, x) - keep positives, zero everything else."""
    return np.maximum(0.0, x)


def maxpool2(x):
    """2x2 max pooling - keep only the biggest value in each 2x2 block."""
    H, W, C = x.shape
    return x[:H - H % 2, :W - W % 2, :].reshape(H // 2, 2, W // 2, 2, C).max(axis=(1, 3))


def softmax(z):
    """Turn 10 raw scores into 10 probabilities that add up to 1."""
    e = np.exp(z - z.max())          # subtract the max for numerical safety
    return e / e.sum()


def forward(image, w, return_maps=False):
    """Run one 28x28x1 image (values 0..1) through the trained CNN."""
    a1 = relu(conv2d_same(image, w["conv1_W"], w["conv1_b"]))   # 28x28x32
    p1 = maxpool2(a1)                                           # 14x14x32
    a2 = relu(conv2d_same(p1, w["conv2_W"], w["conv2_b"]))      # 14x14x64
    p2 = maxpool2(a2)                                           #  7x 7x64
    a3 = relu(conv2d_same(p2, w["conv3_W"], w["conv3_b"]))      #  7x 7x64
    flat = a3.reshape(-1)                                       # 3136
    d1 = relu(flat @ w["dense1_W"] + w["dense1_b"])             # 128
    # Dropout is a TRAINING-only layer. At prediction time it does nothing.
    probs = softmax(d1 @ w["out_W"] + w["out_b"])               # 10

    if return_maps:
        return probs, {
            "conv_1": a1, "pool_1": p1,
            "conv_2": a2, "pool_2": p2,
            "conv_3": a3, "dense_1": d1,
        }
    return probs


def load_weights(path):
    """Load the exported .npz into a plain dict of arrays."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
