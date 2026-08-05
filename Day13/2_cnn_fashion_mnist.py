"""
Day 13 - Mini Project: Fashion MNIST Image Classifier using a CNN
==================================================================
Same dataset as Day 12, completely different architecture.

    Day 12 (ANN) : Flatten the picture into 784 unrelated numbers, then
                   let Dense layers figure it out. Got 86.31%.
    Day 13 (CNN) : Keep the picture a picture. Slide small filters over
                   it to find edges, corners and textures FIRST, and only
                   then hand the findings to Dense layers.

THE PIPELINE
------------
    1.  Load the dataset
    2.  Explore it and show sample images
    3.  Normalize + add the channel dimension
    4.  Show what data augmentation does (concept)
    5.  Build the CNN
    6.  Train it
    7.  Evaluate on data it has never seen
    8.  Plot accuracy and loss curves
    9.  Predict, and show predicted vs actual labels
    10. Per-class accuracy + confusion matrix
    11. 10 correct and 10 incorrect predictions
    12. Look inside: what the filters and feature maps actually see
    13. Save the model and a results summary

Run with:
    python 2_cnn_fashion_mnist.py

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
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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
# STEP 1: LOAD THE DATASET
# ======================================================================
print_header("STEP 1: LOADING FASHION MNIST")

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("Loaded successfully.")
print(f"  Training images : {X_train_full.shape}   labels: {y_train_full.shape}")
print(f"  Test images     : {X_test.shape}   labels: {y_test.shape}")
print("""
  60,000 images to learn from, 10,000 kept in a sealed box to mark us at
  the end. The test set is never trained on and never validated on.
""")


# ======================================================================
# STEP 2: EXPLORE + SAMPLE IMAGES
# ======================================================================
print_header("STEP 2: EXPLORING THE DATASET")

print(f"  Pixel value range : {X_train_full.min()} to {X_train_full.max()}")
print(f"  Number of classes : {len(np.unique(y_train_full))}")
print("\n  Class distribution in the training set:")
print(f"    {'Label':<7} {'Name':<15} {'Count':<8} {'Percent'}")
print("    " + "-" * 40)
for label in range(10):
    count = int((y_train_full == label).sum())
    print(f"    {label:<7} {CLASS_NAMES[label]:<15} {count:<8} "
          f"{count / len(y_train_full) * 100:.1f}%")
print("\n  Perfectly balanced, 6,000 each - so plain accuracy is a fair metric.")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
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

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
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


# ======================================================================
# STEP 3: NORMALIZE AND RESHAPE FOR THE CNN
# ======================================================================
print_header("STEP 3: NORMALIZING AND RESHAPING")

X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
print(f"  Pixels rescaled to : {X_train_full.min():.1f} - {X_train_full.max():.1f}")

# Conv2D needs (height, width, channels). Greyscale = 1 channel.
X_train_full = np.expand_dims(X_train_full, -1)
X_test = np.expand_dims(X_test, -1)
print(f"  Training shape now : {X_train_full.shape}")
print(f"  Test shape now     : {X_test.shape}")
print("""
  THE ONE EXTRA STEP vs THE DAY 12 ANN.
  An ANN flattens the picture immediately, so it never needed a channel
  dimension. A Conv2D layer works ON the picture, so it must be told how
  many colour channels there are:  1 = greyscale, 3 = RGB.
""")


# ======================================================================
# STEP 4: DATA AUGMENTATION (CONCEPT)
# ======================================================================
print_header("STEP 4: DATA AUGMENTATION - THE CONCEPT")

print("""
  The problem: 60,000 images sounds like a lot, but the model can still
  memorise them. Data augmentation makes free extra training data by
  showing slightly changed copies of the images it already has -
  flipped, shifted, rotated, zoomed.

  A shirt flipped left-to-right is still a shirt. So the model learns
  "shirt-ness" instead of memorising one exact arrangement of pixels.

  NOTE: augmentation layers are ACTIVE during training and AUTOMATICALLY
  SWITCHED OFF at prediction time. Keras handles that for you.

  Careful with the choice: horizontal flip is safe for clothes, but would
  be a disaster for handwritten digits, where a mirrored 2 is not a 2.
""")

augment_demo = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
], name="augmentation_demo")

sample = X_train_full[7:8]
fig, axes = plt.subplots(1, 6, figsize=(16, 3.2))
axes[0].imshow(sample[0].squeeze(), cmap="gray")
axes[0].set_title(f"ORIGINAL\n{CLASS_NAMES[y_train_full[7]]}", fontweight="bold", fontsize=10)
axes[0].axis("off")
for ax in axes[1:]:
    ax.imshow(augment_demo(sample, training=True)[0].numpy().squeeze(), cmap="gray")
    ax.set_title("augmented copy", fontsize=10)
    ax.axis("off")
fig.suptitle("Data Augmentation - one image, five free variations",
             fontsize=15, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "data_augmentation.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"Saved: {p}")
print("""
  In this project the overfitting is handled with Dropout instead, which
  is cheaper on CPU. Augmentation is the next thing to reach for if the
  train/validation gap stays wide.
""")


# ======================================================================
# STEP 5: BUILD THE CNN
# ======================================================================
print_header("STEP 5: BUILDING THE CNN")

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1), name="input_layer"),

        # ---- BLOCK 1 : find simple patterns -------------------------
        # 32 filters, each 3x3. padding="same" keeps the size at 28x28
        # so we do not lose the border pixels.
        # Params: (3*3*1 + 1) * 32 = 320
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_1"),
        # Shrink by half: 28x28 -> 14x14. Keeps the strongest signal in
        # each 2x2 block, throws away the exact position. That is what
        # makes a CNN forgiving about a sleeve being 2 pixels to the left.
        layers.MaxPooling2D((2, 2), name="pool_1"),

        # ---- BLOCK 2 : combine them into bigger patterns ------------
        # 64 filters now. Each one looks at all 32 maps from block 1.
        # Params: (3*3*32 + 1) * 64 = 18,496
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_2"),
        layers.MaxPooling2D((2, 2), name="pool_2"),   # 14x14 -> 7x7

        # ---- BLOCK 3 : the most abstract patterns -------------------
        # Params: (3*3*64 + 1) * 64 = 36,928
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_3"),

        # ---- HEAD : turn found patterns into a decision -------------
        # 7*7*64 = 3136 numbers in one long line.
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_1"),
        # Dropout randomly switches off 30% of these neurons on each
        # training step, so no single neuron can become a crutch. It is
        # only active during training.
        layers.Dropout(0.3, name="dropout"),
        layers.Dense(10, activation="softmax", name="output_layer"),
    ],
    name="fashion_mnist_cnn",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

print(f"""
THE JOURNEY OF ONE IMAGE THROUGH THIS NETWORK

  Input                (28, 28, 1)     the raw picture
  Conv2D 32 + ReLU     (28, 28, 32)    32 maps of "where is my pattern?"
  MaxPool 2x2          (14, 14, 32)    half the size, same 32 maps
  Conv2D 64 + ReLU     (14, 14, 64)    64 richer patterns
  MaxPool 2x2          ( 7,  7, 64)    half again
  Conv2D 64 + ReLU     ( 7,  7, 64)    most abstract patterns
  Flatten              (3136,)         7 * 7 * 64
  Dense 128 + ReLU     (128,)          the thinking layer
  Dropout 0.3          (128,)          anti-memorisation, training only
  Dense 10 + Softmax   (10,)           10 probabilities, summing to 1

  Total parameters : {model.count_params():,}

  Compare with the Day 12 ANN: 109,386 parameters for 86.31%. The
  convolution layers here hold only ~56,000 parameters between them,
  yet they do the heavy lifting - because a 3x3 filter is REUSED at
  every position in the image instead of learning a separate weight
  for every pixel. That reuse is the whole trick.
""")


# ======================================================================
# STEP 6: TRAIN
# ======================================================================
print_header("STEP 6: TRAINING")

print(f"""
  Epochs           : {EPOCHS}      (full passes over the training data)
  Batch size       : {BATCH_SIZE}      (images per weight update)
  Validation split : {VALIDATION_SPLIT}     ({int(60000 * VALIDATION_SPLIT):,} images held back)

  EPOCH  = one complete pass over all the training images.
  BATCH  = how many images the model looks at before adjusting weights.
           Small batch = noisy but frequent updates. Big batch = smoother
           but rarer updates. 64 is a sensible middle.
           60,000 * 0.8 = 48,000 images / 64 = 750 updates per epoch.
""")

history = model.fit(
    X_train_full, y_train_full,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=2,          # one clean line per epoch
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
    print("  Above ~5% means the model is memorising. More dropout, data")
    print("  augmentation or early stopping would help.")
else:
    print("  A small gap means it is generalising, not memorising.")


# ======================================================================
# STEP 7: EVALUATE ON THE TEST SET
# ======================================================================
print_header("STEP 7: EVALUATING ON THE TEST SET")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Test loss     : {test_loss:.4f}")
print(f"\n  10,000 images the model has never seen. This is the honest number.")
print(f"  Random guessing scores 10%. The Day 12 ANN scored 86.31%.")


# ======================================================================
# STEP 8: TRAINING CURVES
# ======================================================================
print_header("STEP 8: PLOTTING TRAINING CURVES")

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

fig.suptitle("Fashion MNIST CNN - Training History", fontsize=16, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "training_history.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"Saved: {p}")
print("""
  How to read them:
    both curves improving together   -> healthy learning
    train improves, validation flat  -> overfitting (memorising)
    validation LOSS turning upward   -> stop training, it is now getting
                                        worse on data it has not seen
""")


# ======================================================================
# STEP 9: PREDICTIONS
# ======================================================================
print_header("STEP 9: PREDICTIONS ON SAMPLE IMAGES")

predictions = model.predict(X_test, verbose=0)
predicted_labels = predictions.argmax(axis=1)
confidences = predictions.max(axis=1)

print(f"Predictions array shape: {predictions.shape}  -> 10,000 images x 10 probabilities\n")

print("First test image, full probability breakdown:")
for i, prob in enumerate(predictions[0]):
    marker = "  <-- highest" if i == predicted_labels[0] else ""
    print(f"  {i} {CLASS_NAMES[i]:<15} {prob:.6f} {'#' * int(prob * 40)}{marker}")
print(f"\n  Predicted : {CLASS_NAMES[predicted_labels[0]]}")
print(f"  Actual    : {CLASS_NAMES[y_test[0]]}")
print(f"  Correct   : {'YES' if predicted_labels[0] == y_test[0] else 'NO'}")

print("\nFirst 15 test predictions:")
print(f"  {'#':<4} {'Predicted':<15} {'Actual':<15} {'Confidence':<12} {'Result'}")
print("  " + "-" * 60)
for i in range(15):
    pred, actual = predicted_labels[i], y_test[i]
    print(f"  {i:<4} {CLASS_NAMES[pred]:<15} {CLASS_NAMES[actual]:<15} "
          f"{confidences[i]:<12.4f} {'CORRECT' if pred == actual else 'WRONG'}")

fig, axes = plt.subplots(3, 5, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    pred, actual = predicted_labels[i], y_test[i]
    correct = pred == actual
    ax.imshow(X_test[i].squeeze(), cmap="gray")
    ax.set_title(f"Predicted: {CLASS_NAMES[pred]}\nActual: {CLASS_NAMES[actual]}\n"
                 f"Confidence: {confidences[i]:.1%}",
                 fontsize=9, fontweight="bold",
                 color="#15803d" if correct else "#dc2626")
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


# ======================================================================
# STEP 10: PER-CLASS ACCURACY + CONFUSION MATRIX
# ======================================================================
print_header("STEP 10: PER-CLASS PERFORMANCE")

print(f"  {'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy'}")
print("  " + "-" * 46)
per_class = []
for label in range(10):
    mask = y_test == label
    correct = int((predicted_labels[mask] == label).sum())
    total = int(mask.sum())
    per_class.append(correct / total)
    print(f"  {CLASS_NAMES[label]:<15} {correct:<10} {total:<10} {correct / total:.4f}")

best_c = int(np.argmax(per_class))
worst_c = int(np.argmin(per_class))
print(f"\n  Easiest class : {CLASS_NAMES[best_c]} ({per_class[best_c]:.2%})")
print(f"  Hardest class : {CLASS_NAMES[worst_c]} ({per_class[worst_c]:.2%})")

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
ax.set_title("Confusion Matrix - Fashion MNIST CNN", fontsize=14, fontweight="bold")
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
print("  Read a row: 'of all the real Shirts, where did they get sent?'")
print("  The diagonal is correct. Everything off it is a confusion.")

print("\n  Top 5 confusions:")
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
for actual, pred in np.dstack(np.unravel_index(
        np.argsort(cm_off.ravel())[::-1][:5], cm_off.shape))[0]:
    print(f"    {cm_off[actual, pred]:>4}  real {CLASS_NAMES[actual]:<12} "
          f"called {CLASS_NAMES[pred]}")


# ======================================================================
# STEP 11: 10 CORRECT AND 10 INCORRECT
# ======================================================================
print_header("STEP 11: 10 CORRECT AND 10 INCORRECT PREDICTIONS")

correct_idx = np.where(predicted_labels == y_test)[0]
wrong_idx = np.where(predicted_labels != y_test)[0]
print(f"  Correct : {len(correct_idx):,} of {len(y_test):,}")
print(f"  Wrong   : {len(wrong_idx):,} of {len(y_test):,} "
      f"({len(wrong_idx)/len(y_test)*100:.2f}%)")

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for ax, idx in zip(axes.flat, correct_idx[:10]):
    ax.imshow(X_test[idx].squeeze(), cmap="gray")
    ax.set_title(f"Predicted: {CLASS_NAMES[predicted_labels[idx]]}\n"
                 f"Actual: {CLASS_NAMES[y_test[idx]]}\n"
                 f"Confidence: {confidences[idx]:.1%}",
                 fontsize=9, color="#15803d", fontweight="bold")
    ax.axis("off")
fig.suptitle("10 CORRECTLY Classified Images", fontsize=16, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "correct_predictions.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"\nSaved: {p}")

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for ax, idx in zip(axes.flat, wrong_idx[:10]):
    ax.imshow(X_test[idx].squeeze(), cmap="gray")
    ax.set_title(f"Predicted: {CLASS_NAMES[predicted_labels[idx]]}\n"
                 f"Actual: {CLASS_NAMES[y_test[idx]]}\n"
                 f"Confidence: {confidences[idx]:.1%}",
                 fontsize=9, color="#dc2626", fontweight="bold")
    ax.axis("off")
fig.suptitle("10 INCORRECTLY Classified Images", fontsize=16, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "incorrect_predictions.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"Saved: {p}")


# ======================================================================
# STEP 12: LOOK INSIDE THE CNN
# ======================================================================
print_header("STEP 12: WHAT THE CNN ACTUALLY LEARNED")

# --- The 32 learned filters of conv_1 ---------------------------------
conv1_weights = model.get_layer("conv_1").get_weights()[0]   # (3, 3, 1, 32)
fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(conv1_weights[:, :, 0, i], cmap="viridis")
    ax.set_title(f"f{i}", fontsize=8)
    ax.axis("off")
fig.suptitle("The 32 learned 3x3 filters of the first conv layer\n"
             "(nobody wrote these numbers - training discovered them)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "learned_filters.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"Saved: {p}")

# --- Feature maps for one image ---------------------------------------
feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(n).output for n in ["conv_1", "pool_1", "conv_2", "pool_2"]],
)
maps = feature_extractor.predict(X_test[:1], verbose=0)

fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(4, 9)
ax0 = fig.add_subplot(gs[:2, 0])
ax0.imshow(X_test[0].squeeze(), cmap="gray")
ax0.set_title(f"INPUT\n{CLASS_NAMES[y_test[0]]}", fontsize=10, fontweight="bold")
ax0.axis("off")
for row, (name, fmap) in enumerate(zip(["conv_1", "pool_1", "conv_2", "pool_2"], maps)):
    for col in range(8):
        ax = fig.add_subplot(gs[row, col + 1])
        ax.imshow(fmap[0, :, :, col], cmap="viridis")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(name)
            ax.set_title(f"{name}  {fmap.shape[1]}x{fmap.shape[2]}",
                         fontsize=9, fontweight="bold", loc="left")
fig.suptitle("Feature maps - the same image seen by 8 different filters at each stage",
             fontsize=15, fontweight="bold")
fig.tight_layout()
p = os.path.join(IMAGES_DIR, "feature_maps.png")
fig.savefig(p, dpi=120)
plt.close(fig)
print(f"Saved: {p}")
print("""
  Early layers keep the outline recognisable - they are finding edges.
  Deeper layers look like abstract blobs, because they are no longer
  drawing the object, they are recording "this kind of pattern was found
  roughly here". That progression is exactly what a CNN is for.
""")


# ======================================================================
# STEP 13: SAVE
# ======================================================================
print_header("STEP 13: SAVING ARTIFACTS")

model_path = os.path.join(OUTPUTS_DIR, "fashion_mnist_cnn.keras")
model.save(model_path)
print(f"Model saved: {model_path}")

summary_path = os.path.join(OUTPUTS_DIR, "results_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Day 13 - Fashion MNIST CNN - Results Summary\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"TensorFlow version : {tf.__version__}\n")
    f.write(f"Keras version      : {keras.__version__}\n\n")
    f.write("ARCHITECTURE\n")
    f.write("  Conv2D(32,3x3,relu,same) -> MaxPool(2x2)\n")
    f.write("  Conv2D(64,3x3,relu,same) -> MaxPool(2x2)\n")
    f.write("  Conv2D(64,3x3,relu,same)\n")
    f.write("  Flatten -> Dense(128,relu) -> Dropout(0.3) -> Dense(10,softmax)\n")
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
    f.write(f"  Test loss           : {test_loss:.4f}\n")
    f.write(f"  Correct / wrong     : {len(correct_idx):,} / {len(wrong_idx):,}\n\n")
    f.write("PER-CLASS TEST ACCURACY\n")
    for label in range(10):
        f.write(f"  {CLASS_NAMES[label]:<15} {per_class[label]:.4f}\n")
    f.write(f"\n  Best  : {CLASS_NAMES[best_c]} ({per_class[best_c]:.4f})\n")
    f.write(f"  Worst : {CLASS_NAMES[worst_c]} ({per_class[worst_c]:.4f})\n\n")
    f.write("CONFUSION MATRIX (rows = actual, cols = predicted)\n")
    for i in range(10):
        f.write("  " + " ".join(f"{v:>5}" for v in cm[i]) + f"   <- {CLASS_NAMES[i]}\n")
    f.write("\nCOMPARISON WITH DAY 12 ANN\n")
    f.write("  Day 12 ANN test accuracy : 86.31%\n")
    f.write(f"  Day 13 CNN test accuracy : {test_acc*100:.2f}%\n")
    f.write(f"  Improvement              : {(test_acc*100 - 86.31):+.2f} points\n")
print(f"Summary saved: {summary_path}")

np.savez_compressed(
    os.path.join(OUTPUTS_DIR, "history.npz"),
    **{k: np.array(v) for k, v in history.history.items()},
)
print(f"History saved: {os.path.join(OUTPUTS_DIR, 'history.npz')}")


# ======================================================================
# DONE
# ======================================================================
print_header("MINI PROJECT COMPLETE")
print(f"""
  Training accuracy   : {final_train_acc*100:.2f}%
  Validation accuracy : {final_val_acc*100:.2f}%
  Test accuracy       : {test_acc*100:.2f}%   (Day 12 ANN: 86.31%)
  Improvement over the ANN : {(test_acc*100 - 86.31):+.2f} points

  Generated files:
    images/sample_images.png
    images/one_per_class.png
    images/data_augmentation.png
    images/training_history.png
    images/sample_predictions.png
    images/correct_predictions.png
    images/incorrect_predictions.png
    images/confusion_matrix.png
    images/learned_filters.png
    images/feature_maps.png
    outputs/fashion_mnist_cnn.keras
    outputs/results_summary.txt
    outputs/history.npz
""")
print("=" * 70 + "\n")
