"""
Day 12 - Mini Project: Your First Artificial Neural Network
============================================================
Fashion MNIST clothing classifier, built with TensorFlow / Keras.

THE TASK
--------
Given a 28x28 greyscale photo of a piece of clothing, decide which of
10 categories it belongs to (T-shirt, trouser, pullover, ...).

THE PIPELINE
------------
    1. Load the dataset
    2. Explore it (shapes, labels, sample images)
    3. Normalize pixel values from 0-255 to 0-1
    4. Build a simple ANN
    5. Train it
    6. Evaluate it on data it has never seen
    7. Plot accuracy and loss curves
    8. Predict on a few test images and show predicted vs actual

Run with:
    python 4_ann_fashion_mnist.py

Everything it produces lands in ./images/ and ./outputs/.
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
EPOCHS = 15
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

keras.utils.set_random_seed(SEED)

# Fashion MNIST labels are integers 0-9. These are what they mean.
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ======================================================================
# STEP 1: LOAD THE DATASET
# ======================================================================
print_header("STEP 1: LOADING FASHION MNIST")

# Keras downloads this once (~30 MB) and caches it in ~/.keras/datasets.
# It comes pre-split into a training set and a test set.
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("Loaded successfully.")
print(f"  Training images : {X_train_full.shape}   labels: {y_train_full.shape}")
print(f"  Test images     : {X_test.shape}   labels: {y_test.shape}")
print("""
Reading the shape (60000, 28, 28):
    60000 -> number of images
    28    -> pixel height
    28    -> pixel width
No 4th number, because these are greyscale. A colour dataset would be
(60000, 28, 28, 3) - the 3 being red, green, blue.

Why a separate test set? Because measuring accuracy on data the model
trained on tells you how well it MEMORISED, not how well it LEARNED.
""")


# ======================================================================
# STEP 2: EXPLORE THE DATASET
# ======================================================================
print_header("STEP 2: EXPLORING THE DATASET")

print(f"Pixel value range : {X_train_full.min()} to {X_train_full.max()}")
print(f"Pixel data type   : {X_train_full.dtype}  (uint8 = 8-bit integer, 0-255)")
print(f"Number of classes : {len(np.unique(y_train_full))}")

print("\nClass distribution in the training set:")
print(f"  {'Label':<7} {'Name':<15} {'Count':<8} {'Percent':<8}")
print("  " + "-" * 42)
for label in range(10):
    count = int((y_train_full == label).sum())
    print(f"  {label:<7} {CLASS_NAMES[label]:<15} {count:<8} {count / len(y_train_full) * 100:.1f}%")
print("\n  Perfectly balanced - 6000 of each class. That means plain accuracy")
print("  is a fair metric here; on an imbalanced dataset it would not be.")

# --- Save a grid of sample images -------------------------------------
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_full[i], cmap="gray")
    ax.set_title(f"{CLASS_NAMES[y_train_full[i]]}\n(label {y_train_full[i]})", fontsize=10)
    ax.axis("off")
fig.suptitle("Fashion MNIST - Sample Training Images", fontsize=16, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "sample_images.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"\nSaved: {p}")

# --- One image per class ----------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for label, ax in enumerate(axes.flat):
    first = np.where(y_train_full == label)[0][0]
    ax.imshow(X_train_full[first], cmap="gray")
    ax.set_title(f"{label}: {CLASS_NAMES[label]}", fontsize=11)
    ax.axis("off")
fig.suptitle("One Example of Each Class", fontsize=16, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "one_per_class.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"Saved: {p}")

# --- What a single image looks like as raw numbers --------------------
print("\nTop-left 8x8 corner of the first training image, as raw pixel values:")
print(X_train_full[0][:8, :8])
print("\n  0 = black, 255 = white. An image is literally just a grid of numbers,")
print("  and that grid is what we feed the network.")


# ======================================================================
# STEP 3: NORMALIZE THE PIXEL VALUES
# ======================================================================
print_header("STEP 3: NORMALIZING PIXEL VALUES")

X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print(f"New pixel range : {X_train_full.min():.1f} to {X_train_full.max():.1f}")
print(f"New data type   : {X_train_full.dtype}")
print("""
WHY divide by 255?

  1. Gradient descent behaves badly with large inputs. Weights multiplied
     by values in the hundreds produce huge activations and huge gradients,
     so the optimizer overshoots and the loss bounces around or explodes.

  2. Keras initialises weights assuming inputs are roughly in the -1..1
     range. Feed it 0-255 and those defaults are badly calibrated.

  3. It converges faster. Same model, same epochs - normalized input
     typically reaches several percentage points higher accuracy.

  Crucially: the test set is divided by the SAME 255. Whatever transform
  you apply to training data must be applied identically at inference.
""")


# ======================================================================
# STEP 4: BUILD THE ANN
# ======================================================================
print_header("STEP 4: BUILDING THE ARTIFICIAL NEURAL NETWORK")

model = keras.Sequential(
    [
        # INPUT - declares that each sample is a 28x28 grid.
        layers.Input(shape=(28, 28), name="input_layer"),

        # FLATTEN - unrolls the 28x28 grid into one long vector of 784.
        # A Dense layer can only accept a flat vector, not a 2D grid.
        # This layer has 0 parameters - it only reshapes.
        # (Note: this throws away spatial structure. A CNN keeps it, which
        #  is why CNNs beat plain ANNs on images. But today is about ANNs.)
        layers.Flatten(name="flatten"),

        # HIDDEN LAYER 1 - 128 neurons. Each one looks at all 784 pixels.
        # Params: 784 * 128 + 128 = 100,480
        layers.Dense(128, activation="relu", name="hidden_layer_1"),

        # HIDDEN LAYER 2 - 64 neurons, compressing the representation.
        # Params: 128 * 64 + 64 = 8,256
        layers.Dense(64, activation="relu", name="hidden_layer_2"),

        # OUTPUT - 10 neurons, one per clothing class.
        # Softmax turns 10 raw scores into 10 probabilities summing to 1.
        # Params: 64 * 10 + 10 = 650
        layers.Dense(10, activation="softmax", name="output_layer"),
    ],
    name="fashion_mnist_ann",
)

model.compile(
    # Adam: adaptive learning rate per weight. Sensible default in 2024+.
    optimizer=keras.optimizers.Adam(learning_rate=0.001),

    # sparse_categorical_crossentropy: use this when labels are plain
    # integers (0-9). If labels were one-hot vectors you would use
    # categorical_crossentropy instead. Same maths, different label format.
    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"],
)

model.summary()

print(f"""
LAYER-BY-LAYER BREAKDOWN

  Flatten        (28,28) -> (784,)     0 params      just reshaping
  Dense 128 relu (784,)  -> (128,)     {784*128+128:,} params  784*128 + 128
  Dense  64 relu (128,)  -> (64,)      {128*64+64:,} params    128*64 + 64
  Dense  10 soft (64,)   -> (10,)      {64*10+10:,} params      64*10 + 10
  ------------------------------------------------------------
  TOTAL                                {model.count_params():,} params

  Nearly 92% of the parameters live in the first Dense layer. That is
  typical - the layer touching the raw high-dimensional input is always
  the heaviest.
""")


# ======================================================================
# STEP 5: TRAIN THE MODEL
# ======================================================================
print_header("STEP 5: TRAINING")

print(f"""
Settings:
  Epochs           : {EPOCHS}      (full passes over the training data)
  Batch size       : {BATCH_SIZE}      (samples per weight update)
  Validation split : {VALIDATION_SPLIT}     ({int(60000 * VALIDATION_SPLIT):,} images held back from training)

The validation set is carved out of the TRAINING data, not the test data.
It lets us watch for overfitting during training while keeping the test
set genuinely untouched until the very end.

What happens in one training step:
  1. Forward pass  - push a batch through, get predictions
  2. Loss          - compare predictions to true labels
  3. Backward pass - compute how much each weight contributed to the error
  4. Update        - nudge every weight in the direction that reduces loss
""")

history = model.fit(
    X_train_full, y_train_full,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=1,
)

final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]
final_train_loss = history.history["loss"][-1]
final_val_loss = history.history["val_loss"][-1]

print("\nTraining finished.")
print(f"  Final training accuracy   : {final_train_acc:.4f}  ({final_train_acc*100:.2f}%)")
print(f"  Final validation accuracy : {final_val_acc:.4f}  ({final_val_acc*100:.2f}%)")
print(f"  Final training loss       : {final_train_loss:.4f}")
print(f"  Final validation loss     : {final_val_loss:.4f}")

gap = final_train_acc - final_val_acc
print(f"\n  Train - Val accuracy gap  : {gap:.4f}")
if gap > 0.05:
    print("  A gap above ~5% means the model is starting to memorise the")
    print("  training set. Dropout or early stopping would help.")
else:
    print("  A small gap means the model is generalising well, not memorising.")


# ======================================================================
# STEP 6: EVALUATE ON THE TEST SET
# ======================================================================
print_header("STEP 6: EVALUATING ON THE TEST SET")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"  Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Test loss     : {test_loss:.4f}")
print(f"\n  These 10,000 images were never seen during training - not even as")
print("  validation data. This is the honest number.")
print(f"\n  Random guessing would score 10%. We scored {test_acc*100:.2f}%.")


# ======================================================================
# STEP 7: PLOT THE TRAINING CURVES
# ======================================================================
print_header("STEP 7: PLOTTING TRAINING CURVES")

epochs_range = range(1, EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(epochs_range, history.history["accuracy"], "o-",
         color="#0ea5e9", linewidth=2, markersize=4, label="Training accuracy")
ax1.plot(epochs_range, history.history["val_accuracy"], "s--",
         color="#f97316", linewidth=2, markersize=4, label="Validation accuracy")
ax1.axhline(test_acc, color="#22c55e", linestyle=":", linewidth=2,
            label=f"Test accuracy ({test_acc:.4f})")
ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

ax2.plot(epochs_range, history.history["loss"], "o-",
         color="#0ea5e9", linewidth=2, markersize=4, label="Training loss")
ax2.plot(epochs_range, history.history["val_loss"], "s--",
         color="#f97316", linewidth=2, markersize=4, label="Validation loss")
ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)

fig.suptitle("Fashion MNIST ANN - Training History", fontsize=16, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "training_history.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"Saved: {p}")
print("""
How to read these curves:
  Both rising / falling together    -> healthy learning
  Train keeps improving, val flat   -> overfitting (memorising)
  Val loss turning upward           -> stop training, that is the point
                                       where the model starts to get worse
""")


# ======================================================================
# STEP 8: MAKE PREDICTIONS
# ======================================================================
print_header("STEP 8: MAKING PREDICTIONS ON TEST IMAGES")

# predict() returns, for each image, 10 probabilities - one per class.
predictions = model.predict(X_test, verbose=0)
predicted_labels = predictions.argmax(axis=1)

print(f"Predictions array shape: {predictions.shape}")
print("  -> 10,000 images, 10 probabilities each.\n")

print("First test image, raw probability breakdown:")
for i, prob in enumerate(predictions[0]):
    marker = "  <-- highest" if i == predicted_labels[0] else ""
    bar = "#" * int(prob * 40)
    print(f"  {i} {CLASS_NAMES[i]:<15} {prob:.6f} {bar}{marker}")
print(f"\n  Predicted : {CLASS_NAMES[predicted_labels[0]]}")
print(f"  Actual    : {CLASS_NAMES[y_test[0]]}")
print(f"  Correct   : {'YES' if predicted_labels[0] == y_test[0] else 'NO'}")

# --- Text table of the first 15 predictions ---------------------------
print("\nFirst 15 test predictions:")
print(f"  {'#':<4} {'Predicted':<15} {'Actual':<15} {'Confidence':<12} {'Result':<8}")
print("  " + "-" * 60)
for i in range(15):
    pred, actual = predicted_labels[i], y_test[i]
    conf = predictions[i][pred]
    ok = "CORRECT" if pred == actual else "WRONG"
    print(f"  {i:<4} {CLASS_NAMES[pred]:<15} {CLASS_NAMES[actual]:<15} "
          f"{conf:<12.4f} {ok:<8}")

# --- Visual grid of predictions ---------------------------------------
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    pred, actual = predicted_labels[i], y_test[i]
    conf = predictions[i][pred]
    correct = pred == actual
    ax.imshow(X_test[i], cmap="gray")
    ax.set_title(
        f"Predicted: {CLASS_NAMES[pred]}\nActual: {CLASS_NAMES[actual]}\nConfidence: {conf:.1%}",
        fontsize=9,
        color="#15803d" if correct else "#dc2626",
        fontweight="bold",
    )
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#15803d" if correct else "#dc2626")
        spine.set_linewidth(3)
fig.suptitle("Sample Predictions  (green = correct, red = wrong)",
             fontsize=16, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "sample_predictions.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"\nSaved: {p}")

# --- Show some actual mistakes ----------------------------------------
wrong_idx = np.where(predicted_labels != y_test)[0]
print(f"\nThe model got {len(wrong_idx):,} of {len(y_test):,} test images wrong "
      f"({len(wrong_idx)/len(y_test)*100:.2f}%).")

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for ax, idx in zip(axes.flat, wrong_idx[:10]):
    pred, actual = predicted_labels[idx], y_test[idx]
    ax.imshow(X_test[idx], cmap="gray")
    ax.set_title(f"Predicted: {CLASS_NAMES[pred]}\nActual: {CLASS_NAMES[actual]}",
                 fontsize=9, color="#dc2626", fontweight="bold")
    ax.axis("off")
fig.suptitle("Misclassified Examples - Where the Model Struggles",
             fontsize=16, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "misclassified.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"Saved: {p}")


# ======================================================================
# STEP 9: PER-CLASS ACCURACY AND CONFUSION MATRIX
# ======================================================================
print_header("STEP 9: PER-CLASS PERFORMANCE")

print(f"  {'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
print("  " + "-" * 48)
per_class = []
for label in range(10):
    mask = y_test == label
    correct = int((predicted_labels[mask] == label).sum())
    total = int(mask.sum())
    acc = correct / total
    per_class.append(acc)
    print(f"  {CLASS_NAMES[label]:<15} {correct:<10} {total:<10} {acc:<10.4f}")

best_c = int(np.argmax(per_class))
worst_c = int(np.argmin(per_class))
print(f"\n  Easiest class : {CLASS_NAMES[best_c]} ({per_class[best_c]:.2%})")
print(f"  Hardest class : {CLASS_NAMES[worst_c]} ({per_class[worst_c]:.2%})")
print("\n  Shirt, Coat and Pullover are usually the hardest - they are all")
print("  upper-body garments with near-identical silhouettes at 28x28.")

# --- Confusion matrix, built with numpy (no sklearn needed) -----------
cm = np.zeros((10, 10), dtype=int)
for actual, pred in zip(y_test, predicted_labels):
    cm[actual, pred] += 1

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted label", fontweight="bold")
ax.set_ylabel("Actual label", fontweight="bold")
ax.set_title("Confusion Matrix - Fashion MNIST ANN", fontsize=14, fontweight="bold")
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
fig.colorbar(im, ax=ax, label="Number of images")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "confusion_matrix.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"\nSaved: {p}")
print("  Diagonal = correct. Anything off the diagonal is a confusion.")


# ======================================================================
# STEP 10: SAVE THE MODEL AND A RESULTS SUMMARY
# ======================================================================
print_header("STEP 10: SAVING ARTIFACTS")

model_path = os.path.join(OUTPUTS_DIR, "fashion_mnist_ann.keras")
model.save(model_path)
print(f"Model saved: {model_path}")

summary_path = os.path.join(OUTPUTS_DIR, "results_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Day 12 - Fashion MNIST ANN - Results Summary\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"TensorFlow version : {tf.__version__}\n")
    f.write(f"Keras version      : {keras.__version__}\n\n")
    f.write("ARCHITECTURE\n")
    f.write("  Flatten(28x28) -> Dense(128, relu) -> Dense(64, relu) -> Dense(10, softmax)\n")
    f.write(f"  Total parameters : {model.count_params():,}\n\n")
    f.write("TRAINING CONFIGURATION\n")
    f.write("  Optimizer        : Adam (lr=0.001)\n")
    f.write("  Loss             : sparse_categorical_crossentropy\n")
    f.write(f"  Epochs           : {EPOCHS}\n")
    f.write(f"  Batch size       : {BATCH_SIZE}\n")
    f.write(f"  Validation split : {VALIDATION_SPLIT}\n\n")
    f.write("FINAL RESULTS\n")
    f.write(f"  Training accuracy   : {final_train_acc:.4f}  ({final_train_acc*100:.2f}%)\n")
    f.write(f"  Validation accuracy : {final_val_acc:.4f}  ({final_val_acc*100:.2f}%)\n")
    f.write(f"  Test accuracy       : {test_acc:.4f}  ({test_acc*100:.2f}%)\n")
    f.write(f"  Training loss       : {final_train_loss:.4f}\n")
    f.write(f"  Validation loss     : {final_val_loss:.4f}\n")
    f.write(f"  Test loss           : {test_loss:.4f}\n\n")
    f.write("PER-CLASS TEST ACCURACY\n")
    for label in range(10):
        f.write(f"  {CLASS_NAMES[label]:<15} {per_class[label]:.4f}\n")
    f.write(f"\n  Best  : {CLASS_NAMES[best_c]} ({per_class[best_c]:.4f})\n")
    f.write(f"  Worst : {CLASS_NAMES[worst_c]} ({per_class[worst_c]:.4f})\n")
print(f"Summary saved: {summary_path}")


# ======================================================================
# DONE
# ======================================================================
print_header("MINI PROJECT COMPLETE")
print(f"""
  Training accuracy   : {final_train_acc*100:.2f}%
  Validation accuracy : {final_val_acc*100:.2f}%
  Test accuracy       : {test_acc*100:.2f}%

  Generated files:
    images/sample_images.png
    images/one_per_class.png
    images/training_history.png
    images/sample_predictions.png
    images/misclassified.png
    images/confusion_matrix.png
    outputs/fashion_mnist_ann.keras
    outputs/results_summary.txt
""")
print("=" * 70 + "\n")
