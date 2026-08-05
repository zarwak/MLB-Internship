"""
Day 13 - Practice: Convolutional Neural Networks, step by step
==============================================================
Three small practices, in one script, in the order the task lists them.

    PRACTICE 1  Load Fashion MNIST, look at 10+ images, normalize it
    PRACTICE 2  Build a simple CNN (Conv -> Pool -> Flatten -> Dense)
                and train it for a few epochs
    PRACTICE 3  Evaluate it: train accuracy, test accuracy, loss,
                and predictions on sample images

This file is the "learning" script. The full mini project with all the
graphs lives in 2_cnn_fashion_mnist.py.

Run with:
    python 1_cnn_practice.py

Everything it draws lands in ./images/.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SEED = 42
EPOCHS = 5          # short on purpose - this is the practice run
BATCH_SIZE = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

keras.utils.set_random_seed(SEED)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ======================================================================
# PRACTICE 1 - LOAD, VISUALIZE, NORMALIZE
# ======================================================================
print_header("PRACTICE 1: LOAD THE DATASET")

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Loaded Fashion MNIST.")
print(f"  Training images : {X_train.shape}   labels: {y_train.shape}")
print(f"  Test images     : {X_test.shape}   labels: {y_test.shape}")
print(f"  Pixel range     : {X_train.min()} to {X_train.max()}  (dtype {X_train.dtype})")

# --- Visualize 12 sample images with their labels ---------------------
fig, axes = plt.subplots(3, 4, figsize=(11, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(f"{CLASS_NAMES[y_train[i]]}\n(label {y_train[i]})", fontsize=10)
    ax.axis("off")
fig.suptitle("Practice 1 - 12 Sample Images with Labels", fontsize=15, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "practice_samples.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"\nSaved: {p}")

# --- Normalize ---------------------------------------------------------
print_header("PRACTICE 1: NORMALIZE")

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
print(f"  New pixel range : {X_train.min():.1f} to {X_train.max():.1f}")

# --- Add the channel dimension ----------------------------------------
# THIS IS THE ONE NEW STEP COMPARED TO DAY 12'S ANN.
# A Dense layer wants a flat vector. A Conv2D layer wants a real picture,
# and it insists on being told how many colour channels the picture has.
#   (60000, 28, 28)     -> what load_data() gives us
#   (60000, 28, 28, 1)  -> what Conv2D needs.  1 = greyscale.
# A colour dataset would end in 3 (red, green, blue).
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
print(f"  Reshaped for Conv2D : {X_train.shape}  (the trailing 1 = greyscale)")


# ======================================================================
# PRACTICE 2 - BUILD AND TRAIN A SIMPLE CNN
# ======================================================================
print_header("PRACTICE 2: BUILD A SIMPLE CNN")

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1), name="input_layer"),

        # CONVOLUTION LAYER
        # 32 filters, each a 3x3 window that slides across the image
        # looking for one small pattern (an edge, a corner, a curve).
        # Output: 32 "feature maps" of 26x26 - one map per filter,
        # each map saying "here is where my pattern was found".
        # Params: (3*3*1 + 1) * 32 = 320
        layers.Conv2D(32, (3, 3), activation="relu", name="conv_1"),

        # POOLING LAYER
        # Takes each 2x2 block and keeps only the biggest number.
        # 26x26 -> 13x13. Four times less data, main signal kept.
        # Params: 0 - pooling has nothing to learn, it just shrinks.
        layers.MaxPooling2D((2, 2), name="pool_1"),

        # FLATTEN
        # 13x13x32 = 5408 numbers, unrolled into one long line so the
        # Dense layers can read them. Params: 0.
        layers.Flatten(name="flatten"),

        # FULLY CONNECTED (DENSE) LAYER
        # Now the "thinking" part: combines the found patterns into a
        # decision. Params: 5408*64 + 64 = 346,176
        layers.Dense(64, activation="relu", name="dense_1"),

        # OUTPUT LAYER
        # 10 neurons, one per clothing class. Softmax turns the 10 raw
        # scores into 10 probabilities that add up to 1.
        layers.Dense(10, activation="softmax", name="output_layer"),
    ],
    name="simple_cnn",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

print("""
HOW THE SHAPE CHANGES AS THE IMAGE FLOWS THROUGH

  Input        (28, 28, 1)      the picture
  Conv2D 32    (26, 26, 32)     32 feature maps. Why 26 and not 28? A 3x3
                                window cannot be centred on the outermost
                                pixels, so it loses 1 pixel on each side.
  MaxPool 2x2  (13, 13, 32)     halved. 26/2 = 13
  Flatten      (5408,)          13 * 13 * 32 = 5408
  Dense 64     (64,)            the decision layer
  Dense 10     (10,)            10 probabilities
""")

print_header("PRACTICE 2: TRAIN")
print(f"  Epochs {EPOCHS}, batch size {BATCH_SIZE}, validation_split 0.2")
print("  (short run - the mini project trains a bigger CNN for longer)\n")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1,
)


# ======================================================================
# PRACTICE 3 - EVALUATE
# ======================================================================
print_header("PRACTICE 3: EVALUATE")

train_acc = history.history["accuracy"][-1]
train_loss = history.history["loss"][-1]
val_acc = history.history["val_accuracy"][-1]
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"  Training accuracy   : {train_acc:.4f}  ({train_acc*100:.2f}%)")
print(f"  Validation accuracy : {val_acc:.4f}  ({val_acc*100:.2f}%)")
print(f"  Test accuracy       : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Training loss       : {train_loss:.4f}")
print(f"  Test loss           : {test_loss:.4f}")
print(f"\n  Train - Test gap    : {(train_acc - test_acc):.4f}")
print("""
  ACCURACY vs LOSS - the difference beginners always ask about:

    Accuracy = "out of 100 images, how many did I name correctly?"
               Easy to read, but it does not care HOW sure the model was.

    Loss     = "how wrong were my probabilities?"
               Being 51% sure and right scores much worse than being
               99% sure and right. This is the number the optimizer
               actually tries to make smaller. Accuracy is just the
               human-friendly report card.
""")

# --- Predictions on sample images -------------------------------------
predictions = model.predict(X_test[:12], verbose=0)
predicted_labels = predictions.argmax(axis=1)

print("First 12 test predictions:")
print(f"  {'#':<4} {'Predicted':<15} {'Actual':<15} {'Confidence':<12} {'Result'}")
print("  " + "-" * 58)
for i in range(12):
    pred, actual = predicted_labels[i], y_test[i]
    conf = predictions[i][pred]
    print(f"  {i:<4} {CLASS_NAMES[pred]:<15} {CLASS_NAMES[actual]:<15} "
          f"{conf:<12.4f} {'CORRECT' if pred == actual else 'WRONG'}")

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for i, ax in enumerate(axes.flat):
    pred, actual = predicted_labels[i], y_test[i]
    correct = pred == actual
    ax.imshow(X_test[i].squeeze(), cmap="gray")
    ax.set_title(f"Pred: {CLASS_NAMES[pred]}\nTrue: {CLASS_NAMES[actual]}",
                 fontsize=9, fontweight="bold",
                 color="#15803d" if correct else "#dc2626")
    ax.axis("off")
fig.suptitle("Practice 3 - Predictions (green = correct, red = wrong)",
             fontsize=15, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "practice_predictions.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"\nSaved: {p}")


# ======================================================================
# BONUS - WHAT DOES A CONVOLUTION ACTUALLY DO? (no training involved)
# ======================================================================
print_header("BONUS: SEEING A CONVOLUTION BY HAND")

# Three classic 3x3 filters, hand-written. A real CNN LEARNS these
# numbers; here we set them ourselves just to see the effect.
filters = {
    "Vertical edge detector": np.array([[-1, 0, 1],
                                        [-1, 0, 1],
                                        [-1, 0, 1]], dtype="float32"),
    "Horizontal edge detector": np.array([[-1, -1, -1],
                                          [ 0,  0,  0],
                                          [ 1,  1,  1]], dtype="float32"),
    "Blur": np.ones((3, 3), dtype="float32") / 9.0,
}

image = X_test[0].squeeze()          # one 28x28 picture

def convolve(img, kernel):
    """Slide the kernel over the image, one pixel at a time."""
    h, w = img.shape
    k = kernel.shape[0]
    out = np.zeros((h - k + 1, w - k + 1), dtype="float32")
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            patch = img[r:r + k, c:c + k]     # the 3x3 window
            out[r, c] = np.sum(patch * kernel)  # multiply, then add up
    return out

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
axes[0].imshow(image, cmap="gray")
axes[0].set_title(f"Original\n({CLASS_NAMES[y_test[0]]})", fontweight="bold")
axes[0].axis("off")
for ax, (name, kernel) in zip(axes[1:], filters.items()):
    ax.imshow(convolve(image, kernel), cmap="gray")
    ax.set_title(name, fontweight="bold", fontsize=10)
    ax.axis("off")
fig.suptitle("One filter = one pattern detector. This is all a convolution is.",
             fontsize=14, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "convolution_by_hand.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"Saved: {p}")
print("""
  Each filter is 9 numbers. Slide it over the picture, multiply the 9
  pixels under it by the 9 filter numbers, add them up, write down the
  answer. Move one pixel right. Repeat.

  A vertical-edge filter has negative numbers on the left and positive
  on the right, so it outputs a big number exactly where dark meets
  light vertically - i.e. at a vertical edge.

  In a real CNN nobody writes these 9 numbers. The network starts with
  random numbers and BACKPROPAGATION tunes them until they detect
  whatever turns out to be useful for telling a sneaker from a sandal.
""")

print_header("PRACTICE COMPLETE")
print(f"""
  Simple CNN test accuracy : {test_acc*100:.2f}%  (after only {EPOCHS} epochs)

  Generated files:
    images/practice_samples.png
    images/practice_predictions.png
    images/convolution_by_hand.png

  Next: python 2_cnn_fashion_mnist.py
""")
print("=" * 70 + "\n")
