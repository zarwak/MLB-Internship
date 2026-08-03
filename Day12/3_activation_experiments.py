"""
Day 12 - Practice 3: Experimenting with Activation Functions
=============================================================

GOAL
----
Build the SAME network three times, changing only the hidden layer's
activation function: ReLU, Sigmoid, Tanh. Then answer two questions:

  Q1. Does the activation function change the model STRUCTURE?
  Q2. Does it change how well the model LEARNS?

The answers are "no" and "yes, a lot" - and understanding why is the
whole point of this exercise.

Run with:
    python 3_activation_experiments.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hush TF's startup noise

import numpy as np
import matplotlib
matplotlib.use("Agg")               # save figures to disk, do not open a window
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Fixed seed so every run gives the same numbers - otherwise the comparison
# between activations would be polluted by random weight initialisation.
SEED = 42
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

ACTIVATIONS = ["relu", "sigmoid", "tanh"]


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ======================================================================
# PART 1 - What do these functions actually look like?
# ======================================================================
print_header("PART 1: THE THREE ACTIVATION FUNCTIONS")

print("""
ReLU  -  Rectified Linear Unit
    f(x) = max(0, x)
    Range   : [0, inf)
    Shape   : flat at 0 for negatives, then a straight 45-degree line
    Good    : very fast, no vanishing gradient for positive inputs.
              The default choice for hidden layers today.
    Bad     : "dying ReLU" - a neuron stuck outputting 0 has zero
              gradient and can never recover.

Sigmoid  -  Logistic function
    f(x) = 1 / (1 + e^-x)
    Range   : (0, 1)
    Shape   : smooth S-curve
    Good    : output reads directly as a probability, which is why it is
              the standard OUTPUT activation for binary classification.
    Bad     : saturates. For large |x| the curve is nearly flat, so the
              gradient is nearly 0 and learning stalls. This is the
              classic "vanishing gradient" problem.

Tanh  -  Hyperbolic tangent
    f(x) = (e^x - e^-x) / (e^x + e^-x)
    Range   : (-1, 1)
    Shape   : S-curve like sigmoid, but centred on 0
    Good    : zero-centred output usually converges faster than sigmoid.
    Bad     : still saturates at the extremes, same vanishing gradient
              problem, just less severe than sigmoid.
""")

# Plot them so the shapes are not just words.
x = np.linspace(-6, 6, 400)
curves = {
    "ReLU":    np.maximum(0, x),
    "Sigmoid": 1 / (1 + np.exp(-x)),
    "Tanh":    np.tanh(x),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, y) in zip(axes, curves.items()):
    ax.plot(x, y, linewidth=2.5, color="#0ea5e9")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.set_xlabel("input (x)")
    ax.set_ylabel("output f(x)")
    ax.grid(alpha=0.3)
fig.suptitle("Activation Functions", fontsize=15, fontweight="bold")
fig.tight_layout()
curve_path = os.path.join(IMAGES_DIR, "activation_functions.png")
fig.savefig(curve_path, dpi=120)
plt.close(fig)
print(f"Saved: {curve_path}")


# ======================================================================
# PART 2 - Does the activation change the model STRUCTURE?
# ======================================================================
print_header("PART 2: DOES THE ACTIVATION CHANGE THE STRUCTURE?")

N_FEATURES = 4
N_HIDDEN = 8
N_CLASSES = 3


def build_model(activation):
    """Identical architecture every time - only the hidden activation varies."""
    keras.utils.set_random_seed(SEED)      # same starting weights each build
    model = keras.Sequential(
        [
            layers.Input(shape=(N_FEATURES,)),
            layers.Dense(N_HIDDEN, activation=activation, name=f"hidden_{activation}"),
            layers.Dense(N_CLASSES, activation="softmax", name="output"),
        ],
        name=f"model_{activation}",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


models = {}
for act in ACTIVATIONS:
    print(f"\n{'-' * 70}")
    print(f"MODEL WITH activation='{act}'")
    print("-" * 70)
    m = build_model(act)
    models[act] = m
    m.summary()

print_header("STRUCTURAL COMPARISON")
print(f"{'Activation':<12} {'Layers':<10} {'Total Params':<15} {'Output Shape':<15}")
print("-" * 60)
for act, m in models.items():
    out_shape = str(m.layers[-1].output.shape)
    print(f"{act:<12} {len(m.layers):<10} {m.count_params():<15} {out_shape:<15}")

print("""
OBSERVATION 1 - The structure is IDENTICAL.
  Same layer count. Same 67 parameters. Same output shape.

  Why? Because an activation function has no weights of its own. It is a
  fixed mathematical function applied element-wise to the output of the
  Dense layer. Changing max(0,x) to 1/(1+e^-x) does not add or remove a
  single learnable number.

  So: activation choice affects BEHAVIOUR, not ARCHITECTURE.
""")


# ======================================================================
# PART 3 - Does the activation change how the model LEARNS?
# ======================================================================
print_header("PART 3: DOES THE ACTIVATION CHANGE THE LEARNING?")

print("""
model.summary() cannot show this, so we have to actually train. We use a
small synthetic 3-class dataset - deliberately non-linear, so the network
has to bend its decision boundary rather than draw a straight line.
""")

# Build a synthetic dataset: three interleaved spirals.
def make_spirals(points_per_class=300, n_classes=3, noise=0.20, seed=SEED):
    rng = np.random.default_rng(seed)
    X = np.zeros((points_per_class * n_classes, 2))
    y = np.zeros(points_per_class * n_classes, dtype=int)
    for c in range(n_classes):
        idx = range(points_per_class * c, points_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, points_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, points_per_class) + rng.normal(0, noise, points_per_class)
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = c
    return X, y


X_raw, y = make_spirals()
# Add two engineered features so the input width stays 4, matching the models above.
X = np.c_[X_raw, X_raw[:, 0] ** 2, X_raw[:, 1] ** 2].astype("float32")

# Shuffle, then split 80/20 into train and validation.
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X))
X, y = X[perm], y[perm]
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"Dataset  : {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"Train/Val: {len(X_train)} / {len(X_val)}")

histories = {}
results = {}

for act in ACTIVATIONS:
    print(f"\nTraining with activation='{act}' ...")
    model = build_model(act)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        verbose=0,
    )
    histories[act] = hist.history
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    results[act] = {
        "val_accuracy": acc,
        "val_loss": loss,
        "final_train_acc": hist.history["accuracy"][-1],
        "params": model.count_params(),
    }
    print(f"   done - validation accuracy {acc:.4f}, validation loss {loss:.4f}")


print_header("LEARNING COMPARISON (60 epochs, identical seed and data)")
print(f"{'Activation':<12} {'Params':<10} {'Train Acc':<12} {'Val Acc':<12} {'Val Loss':<12}")
print("-" * 62)
for act, r in results.items():
    print(f"{act:<12} {r['params']:<10} {r['final_train_acc']:<12.4f} "
          f"{r['val_accuracy']:<12.4f} {r['val_loss']:<12.4f}")

best = max(results, key=lambda a: results[a]["val_accuracy"])
print(f"\nBest performer here: {best.upper()} "
      f"({results[best]['val_accuracy']:.4f} validation accuracy)")


# ----------------------------------------------------------------------
# Plot the learning curves side by side
# ----------------------------------------------------------------------
colors = {"relu": "#0ea5e9", "sigmoid": "#f97316", "tanh": "#22c55e"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for act in ACTIVATIONS:
    ax1.plot(histories[act]["accuracy"], color=colors[act], linewidth=2, label=f"{act} (train)")
    ax1.plot(histories[act]["val_accuracy"], color=colors[act], linewidth=1.5,
             linestyle="--", alpha=0.7, label=f"{act} (val)")
    ax2.plot(histories[act]["loss"], color=colors[act], linewidth=2, label=f"{act} (train)")
    ax2.plot(histories[act]["val_loss"], color=colors[act], linewidth=1.5,
             linestyle="--", alpha=0.7, label=f"{act} (val)")

ax1.set_title("Accuracy per Epoch", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2.set_title("Loss per Epoch", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

fig.suptitle("Same Architecture, Different Activation Functions",
             fontsize=15, fontweight="bold")
fig.tight_layout()
comp_path = os.path.join(IMAGES_DIR, "activation_comparison.png")
fig.savefig(comp_path, dpi=120)
plt.close(fig)
print(f"\nSaved: {comp_path}")


# ======================================================================
# PART 4 - What did we learn?
# ======================================================================
print_header("CONCLUSIONS")

print("""
1. STRUCTURE is unaffected.
   All three models have exactly the same layers, shapes and parameter
   count. An activation function carries no weights, so model.summary()
   looks identical no matter which one you pick.

2. LEARNING SPEED and FINAL ACCURACY are very much affected.
   ReLU normally climbs fastest. Sigmoid is usually the slowest starter
   because its gradient is at most 0.25, so weight updates are small and
   the signal shrinks as it travels backwards through the layers.
   Tanh sits in between - zero-centred, so it beats sigmoid, but it still
   saturates at the extremes.

3. PRACTICAL RULES OF THUMB
   Hidden layers          -> ReLU (start here, always)
   Binary classification  -> Sigmoid on the output layer
   Multi-class output     -> Softmax on the output layer
   Regression output      -> no activation (linear)
   RNN / LSTM internals   -> Tanh and Sigmoid (they are built in)

4. THE REASON ACTIVATIONS EXIST AT ALL
   Without one, Dense -> Dense is just matrix multiply -> matrix multiply,
   which algebraically collapses into a SINGLE matrix multiply. A 50-layer
   network with no activations has exactly the expressive power of one
   layer. The non-linearity is what makes depth worth having.
""")

print("=" * 70)
print("  Next: 4_ann_fashion_mnist.py  (the mini project)")
print("=" * 70 + "\n")
