"""
Day 14 - MINI PROJECT: Cats vs Dogs classifier using Transfer Learning
=====================================================================

The whole plan in one paragraph:

  MobileNetV2 has already looked at 1.4 million photos and learned what
  edges, fur, ears and eyes look like. We keep all of that, freeze it so
  it cannot be damaged, bolt a tiny 2-class decision layer on top, and
  train ONLY that layer. Then we gently un-freeze the last few layers and
  polish them with a very small learning rate. Two short phases, and we
  end up far above what a from-scratch CNN could reach in the same time.

Steps in this file:
   1. Load Cats vs Dogs from TensorFlow Datasets
   2. Explore the data (how many, how big, what do they look like)
   3. Preprocess: resize -> batch -> prefetch, split 80/20
   4. Show what data augmentation does
   5. Build the model: frozen MobileNetV2 + my own head
   6. PHASE 1 - train the head only (feature extraction)
   7. PHASE 2 - unfreeze the top of MobileNetV2 and fine-tune
   8. Evaluate on the validation set
   9. Plot accuracy and loss curves
  10. Show sample predictions (right ones and wrong ones)
  11. Confusion matrix + precision / recall / F1
  12. Save everything to outputs/

Run it with:   python 2_cats_vs_dogs_transfer_learning.py

Optional environment variables (handy while experimenting):
  D14_EPOCHS1=5   epochs for phase 1
  D14_EPOCHS2=3   epochs for phase 2
  D14_SMOKE=1     tiny run on a few batches, just to check nothing crashes
"""

import os
import time
import json
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
HERE = Path(__file__).parent
IMAGES = HERE / "images"
OUTPUTS = HERE / "outputs"
IMAGES.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

keras.utils.set_random_seed(42)

IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS_PHASE1 = int(os.environ.get("D14_EPOCHS1", 5))
EPOCHS_PHASE2 = int(os.environ.get("D14_EPOCHS2", 3))
FINE_TUNE_FROM = 100  # unfreeze MobileNetV2 layers from index 100 onwards
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5  # 100x smaller - this matters, see the README
SMOKE = os.environ.get("D14_SMOKE") == "1"

CLASS_NAMES = ["cat", "dog"]


def banner(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


t_start = time.time()

# ==========================================================================
# STEP 1 - Load the dataset
# ==========================================================================
banner("STEP 1 - Load Cats vs Dogs from TensorFlow Datasets")

(train_raw, val_raw), info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    with_info=True,
    as_supervised=True,
)

n_total = info.splits["train"].num_examples
n_train = int(train_raw.cardinality())
n_val = int(val_raw.cardinality())

print(f"Total images      : {n_total:,}")
print(f"Training images   : {n_train:,}")
print(f"Validation images : {n_val:,}")
print(f"Classes           : {info.features['label'].names}")
print(f"Downloaded to     : {info.data_dir}")

# ==========================================================================
# STEP 2 - Explore
# ==========================================================================
banner("STEP 2 - Explore the data")

shapes = []
labels_seen = []
for image, label in train_raw.take(200):
    shapes.append(tuple(image.shape[:2]))
    labels_seen.append(int(label))

heights = np.array([s[0] for s in shapes])
widths = np.array([s[1] for s in shapes])
print(f"Sample of 200 training images:")
print(f"  height : min={heights.min()}  max={heights.max()}  mean={heights.mean():.0f}")
print(f"  width  : min={widths.min()}   max={widths.max()}  mean={widths.mean():.0f}")
print(f"  cats={labels_seen.count(0)}  dogs={labels_seen.count(1)}  (roughly balanced)")
print("Every picture is a different size -> resizing is not optional.")

plt.figure(figsize=(13, 6))
for i, (image, label) in enumerate(train_raw.take(10)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image.numpy())
    plt.title(f"{CLASS_NAMES[int(label)]}  {image.shape[0]}x{image.shape[1]}", fontsize=10)
    plt.axis("off")
plt.suptitle("Cats vs Dogs - 10 original images (all different sizes)", fontsize=14)
plt.tight_layout()
plt.savefig(IMAGES / "sample_images.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"Saved: images/sample_images.png")

# ==========================================================================
# STEP 3 - Preprocess and build the input pipeline
# ==========================================================================
banner("STEP 3 - Preprocess: resize, batch, prefetch")

AUTOTUNE = tf.data.AUTOTUNE


def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train_ds = (
    train_raw.map(preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
val_ds = (
    val_raw.map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

if SMOKE:
    print("*** SMOKE MODE: using only a few batches ***")
    train_ds = train_ds.take(4)
    val_ds = val_ds.take(4)

print(f"Image size    : {IMG_SIZE} x {IMG_SIZE} x 3")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Train batches : {len(train_ds)}")
print(f"Val batches   : {len(val_ds)}")

# ==========================================================================
# STEP 4 - Data augmentation
# ==========================================================================
banner("STEP 4 - Data augmentation")

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

print("RandomFlip('horizontal') - a mirrored dog is still a dog")
print("RandomRotation(0.1)      - +/- 10% of a full turn")
print("RandomZoom(0.1)          - zoom in or out by up to 10%")
print("These only run while TRAINING. Keras switches them off when predicting.")

plt.figure(figsize=(13, 3))
for images, _ in train_ds.take(1):
    first = images[0]
    plt.subplot(1, 6, 1)
    plt.imshow(first.numpy().astype("uint8"))
    plt.title("original", fontsize=10)
    plt.axis("off")
    for i in range(5):
        aug = data_augmentation(tf.expand_dims(first, 0), training=True)[0]
        plt.subplot(1, 6, i + 2)
        plt.imshow(tf.clip_by_value(aug, 0, 255).numpy().astype("uint8"))
        plt.title(f"augmented {i + 1}", fontsize=10)
        plt.axis("off")
plt.suptitle("Data augmentation - one image, five free variations", fontsize=13)
plt.tight_layout()
plt.savefig(IMAGES / "data_augmentation.png", dpi=110, bbox_inches="tight")
plt.close()
print("Saved: images/data_augmentation.png")

# ==========================================================================
# STEP 5 - Build the model
# ==========================================================================
banner("STEP 5 - Build the model: frozen MobileNetV2 + my own head")

base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,      # throw away the original 1000-class output layer
    weights="imagenet",     # keep everything it learned from ImageNet
)
base_model.trainable = False  # FREEZE

print(f"MobileNetV2 : {len(base_model.layers)} layers, {base_model.count_params():,} parameters - all frozen")

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1, name="rescale_to_minus1_1")(x)
x = base_model(x, training=False)                                  # (5, 5, 1280)
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)       # (1280,)
x = layers.Dropout(0.2, name="dropout")(x)
outputs = layers.Dense(1, activation="sigmoid", name="cat_or_dog")(x)

model = keras.Model(inputs, outputs, name="cats_vs_dogs_transfer")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE1),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

total_params = model.count_params()
trainable_params = sum(int(tf.size(w)) for w in model.trainable_weights)
print(f"\nTotal parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}  ({100 * trainable_params / total_params:.2f}%)")
print("We are training almost nothing - and borrowing almost everything.")

# ==========================================================================
# STEP 6 - PHASE 1: feature extraction
# ==========================================================================
banner(f"STEP 6 - PHASE 1: train the head only ({EPOCHS_PHASE1} epochs, lr={LR_PHASE1})")

loss0, acc0 = model.evaluate(val_ds, verbose=0)
print(f"Before any training: val_accuracy = {acc0 * 100:.2f}%  (a coin flip, as expected)")

t0 = time.time()
history1 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE1,
    validation_data=val_ds,
    verbose=1,
)
phase1_time = time.time() - t0
print(f"\nPhase 1 took {phase1_time / 60:.1f} minutes")

# ==========================================================================
# STEP 7 - PHASE 2: fine-tuning
# ==========================================================================
banner(f"STEP 7 - PHASE 2: fine-tune the top of MobileNetV2 ({EPOCHS_PHASE2} epochs, lr={LR_PHASE2})")

base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_FROM]:
    layer.trainable = False

# IMPORTANT: BatchNormalization layers must stay frozen. They carry running
# averages from ImageNet; letting them update on a small batch wrecks accuracy.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Re-compile so the new learning rate and the new frozen/unfrozen split apply.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE2),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

trainable_params2 = sum(int(tf.size(w)) for w in model.trainable_weights)
unfrozen_layers = sum(1 for l in base_model.layers if l.trainable)
print(f"Unfrozen layers      : {unfrozen_layers} of {len(base_model.layers)} (from index {FINE_TUNE_FROM})")
print(f"Trainable parameters : {trainable_params2:,}  ({100 * trainable_params2 / total_params:.2f}%)")
print(f"Learning rate        : {LR_PHASE2} - {int(LR_PHASE1 / LR_PHASE2)}x smaller than phase 1")

t0 = time.time()
history2 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
    initial_epoch=len(history1.epoch),  # continue the epoch numbering
    validation_data=val_ds,
    verbose=1,
)
phase2_time = time.time() - t0
print(f"\nPhase 2 took {phase2_time / 60:.1f} minutes")

# Stitch the two histories together for plotting
hist = {k: list(history1.history[k]) + list(history2.history[k]) for k in history1.history}

# ==========================================================================
# STEP 8 - Evaluate
# ==========================================================================
banner("STEP 8 - Evaluate on the validation set")

train_loss, train_acc = model.evaluate(train_ds, verbose=0)
val_loss, val_acc = model.evaluate(val_ds, verbose=0)

print(f"Training   accuracy : {train_acc * 100:.2f}%   loss: {train_loss:.4f}")
print(f"Validation accuracy : {val_acc * 100:.2f}%   loss: {val_loss:.4f}")
print(f"Train/val gap       : {(train_acc - val_acc) * 100:+.2f} points")

target = 0.93
minimum = 0.90
print(f"\nMinimum target (90%) : {'PASS' if val_acc >= minimum else 'FAIL'}")
print(f"Stretch target (93%)  : {'PASS' if val_acc >= target else 'FAIL'}")

# ==========================================================================
# STEP 9 - Plot the curves
# ==========================================================================
banner("STEP 9 - Plot accuracy and loss curves")

epochs_range = range(1, len(hist["accuracy"]) + 1)
switch = EPOCHS_PHASE1 + 0.5

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, hist["accuracy"], "o-", label="Training accuracy")
axes[0].plot(epochs_range, hist["val_accuracy"], "s-", label="Validation accuracy")
axes[0].axvline(switch, color="red", ls="--", alpha=0.7)
# y in axes-fraction coords so the label never collides with the legend
axes[0].text(switch + 0.05, 0.5, " fine-tuning starts", color="red", fontsize=9,
             transform=axes[0].get_xaxis_transform())
axes[0].set_title("Model Accuracy", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, hist["loss"], "o-", label="Training loss")
axes[1].plot(epochs_range, hist["val_loss"], "s-", label="Validation loss")
axes[1].axvline(switch, color="red", ls="--", alpha=0.7)
axes[1].text(switch + 0.05, 0.5, " fine-tuning starts", color="red", fontsize=9,
             transform=axes[1].get_xaxis_transform())
axes[1].set_title("Model Loss", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.3)

plt.suptitle(
    f"Cats vs Dogs - Transfer Learning with MobileNetV2  (final val accuracy {val_acc * 100:.2f}%)",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(IMAGES / "training_history.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: images/training_history.png")

# ==========================================================================
# STEP 10 - Predictions
# ==========================================================================
banner("STEP 10 - Predictions on validation images")

all_probs = []
all_true = []
# Keep the first 60 batches (1,920 images) so the "got it wrong" grid has
# enough failures to fill 10 slots - at 98.6% accuracy, mistakes are rare.
kept_images = []
for i, (images, labels) in enumerate(val_ds):
    probs = model.predict(images, verbose=0).flatten()
    all_probs.append(probs)
    all_true.append(labels.numpy())
    if i < 60:
        kept_images.append(images.numpy().astype("uint8"))

all_probs = np.concatenate(all_probs)
all_true = np.concatenate(all_true)
all_pred = (all_probs >= 0.5).astype(int)
kept_images = np.concatenate(kept_images) if kept_images else np.zeros((0, IMG_SIZE, IMG_SIZE, 3), "uint8")
n_kept = len(kept_images)

correct_mask = all_pred == all_true
print(f"Predicted on {len(all_true):,} validation images")
print(f"Correct : {int(correct_mask.sum()):,}")
print(f"Wrong   : {int((~correct_mask).sum()):,}")
print(f"Accuracy: {correct_mask.mean() * 100:.2f}%")


def confidence(p):
    """How sure was the model, as a 0-100% number."""
    return p * 100 if p >= 0.5 else (1 - p) * 100


def grid(indices, filename, title):
    n = min(len(indices), 10)
    if n == 0:
        print(f"  (nothing to plot for {filename})")
        return
    plt.figure(figsize=(15, 7))
    for j, idx in enumerate(indices[:n]):
        plt.subplot(2, 5, j + 1)
        plt.imshow(kept_images[idx])
        ok = all_pred[idx] == all_true[idx]
        plt.title(
            f"pred: {CLASS_NAMES[all_pred[idx]]}  ({confidence(all_probs[idx]):.1f}%)\n"
            f"true: {CLASS_NAMES[all_true[idx]]}",
            fontsize=10,
            color="green" if ok else "red",
        )
        plt.axis("off")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(IMAGES / filename, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved: images/{filename}")


rng = np.random.default_rng(42)
sample_idx = rng.choice(n_kept, size=min(10, n_kept), replace=False)
grid(sample_idx, "sample_predictions.png", "Sample predictions (green = correct, red = wrong)")

correct_idx = np.where(correct_mask[:n_kept])[0]
wrong_idx = np.where(~correct_mask[:n_kept])[0]
grid(correct_idx, "correct_predictions.png", "Images the model got RIGHT")
# Show the most confidently wrong ones - those are the interesting failures
wrong_sorted = wrong_idx[np.argsort([-confidence(all_probs[i]) for i in wrong_idx])]
grid(wrong_sorted, "incorrect_predictions.png", "Images the model got WRONG (most confident mistakes first)")

# ==========================================================================
# STEP 11 - Confusion matrix and per-class scores
# ==========================================================================
banner("STEP 11 - Confusion matrix")

cm = tf.math.confusion_matrix(all_true, all_pred, num_classes=2).numpy()
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

print("                 predicted cat   predicted dog")
print(f"  actual cat     {tn:>10,}     {fp:>11,}")
print(f"  actual dog     {fn:>10,}     {tp:>11,}")

cat_acc = tn / (tn + fp) * 100
dog_acc = tp / (tp + fn) * 100
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print(f"\nCat accuracy : {cat_acc:.2f}%   ({tn:,} of {tn + fp:,})")
print(f"Dog accuracy : {dog_acc:.2f}%   ({tp:,} of {tp + fn:,})")
print(f"Precision (dog) : {precision:.4f}")
print(f"Recall    (dog) : {recall:.4f}")
print(f"F1 score  (dog) : {f1:.4f}")

plt.figure(figsize=(6.5, 5.5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Cats vs Dogs", fontsize=13, fontweight="bold")
plt.colorbar()
plt.xticks([0, 1], CLASS_NAMES)
plt.yticks([0, 1], CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(
            j, i, f"{cm[i, j]:,}",
            ha="center", va="center", fontsize=18,
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )
plt.tight_layout()
plt.savefig(IMAGES / "confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: images/confusion_matrix.png")

# ==========================================================================
# STEP 12 - Save everything
# ==========================================================================
banner("STEP 12 - Save the model and results")

model.save(OUTPUTS / "cats_vs_dogs_mobilenetv2.keras")
np.savez(OUTPUTS / "history.npz", **{k: np.array(v) for k, v in hist.items()})
np.savez_compressed(
    OUTPUTS / "predictions.npz",
    probs=all_probs.astype("float32"),
    true=all_true.astype("int8"),
    pred=all_pred.astype("int8"),
)

total_time = time.time() - t_start
summary = {
    "model": "MobileNetV2 (ImageNet) + GlobalAvgPool + Dropout(0.2) + Dense(1, sigmoid)",
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "train_images": int(n_train),
    "val_images": int(n_val),
    "epochs_phase1": EPOCHS_PHASE1,
    "epochs_phase2": EPOCHS_PHASE2,
    "lr_phase1": LR_PHASE1,
    "lr_phase2": LR_PHASE2,
    "fine_tune_from_layer": FINE_TUNE_FROM,
    "total_params": int(total_params),
    "trainable_params_phase1": int(trainable_params),
    "trainable_params_phase2": int(trainable_params2),
    "train_accuracy": float(train_acc),
    "train_loss": float(train_loss),
    "val_accuracy": float(val_acc),
    "val_loss": float(val_loss),
    "val_accuracy_after_phase1": float(history1.history["val_accuracy"][-1]),
    "best_val_accuracy": float(max(hist["val_accuracy"])),
    "correct": int(correct_mask.sum()),
    "wrong": int((~correct_mask).sum()),
    "cat_accuracy": float(cat_acc / 100),
    "dog_accuracy": float(dog_acc / 100),
    "precision_dog": float(precision),
    "recall_dog": float(recall),
    "f1_dog": float(f1),
    "confusion_matrix": cm.tolist(),
    "phase1_minutes": round(phase1_time / 60, 2),
    "phase2_minutes": round(phase2_time / 60, 2),
    "total_minutes": round(total_time / 60, 2),
    "history": {k: [float(x) for x in v] for k, v in hist.items()},
}
(OUTPUTS / "results_summary.json").write_text(json.dumps(summary, indent=2))

with open(OUTPUTS / "results_summary.txt", "w", encoding="utf-8") as f:
    f.write("Day 14 - Cats vs Dogs, Transfer Learning with MobileNetV2\n")
    f.write("=" * 58 + "\n\n")
    f.write(f"Training images       : {n_train:,}\n")
    f.write(f"Validation images     : {n_val:,}\n")
    f.write(f"Image size            : {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"Total parameters      : {total_params:,}\n")
    f.write(f"Trainable (phase 1)   : {trainable_params:,}\n")
    f.write(f"Trainable (phase 2)   : {trainable_params2:,}\n\n")
    f.write(f"Val accuracy after phase 1 : {history1.history['val_accuracy'][-1] * 100:.2f}%\n")
    f.write(f"Val accuracy final         : {val_acc * 100:.2f}%\n")
    f.write(f"Train accuracy final       : {train_acc * 100:.2f}%\n")
    f.write(f"Correct / wrong            : {int(correct_mask.sum()):,} / {int((~correct_mask).sum()):,}\n")
    f.write(f"Cat accuracy               : {cat_acc:.2f}%\n")
    f.write(f"Dog accuracy               : {dog_acc:.2f}%\n")
    f.write(f"F1 (dog)                   : {f1:.4f}\n")
    f.write(f"Total run time             : {total_time / 60:.1f} minutes\n\n")
    f.write("Per-epoch history\n")
    f.write("epoch  train_acc  val_acc  train_loss  val_loss\n")
    for i in range(len(hist["accuracy"])):
        f.write(
            f"{i + 1:5d}  {hist['accuracy'][i]:9.4f}  {hist['val_accuracy'][i]:7.4f}"
            f"  {hist['loss'][i]:10.4f}  {hist['val_loss'][i]:8.4f}\n"
        )

print("Saved: outputs/cats_vs_dogs_mobilenetv2.keras")
print("Saved: outputs/history.npz")
print("Saved: outputs/predictions.npz")
print("Saved: outputs/results_summary.json")
print("Saved: outputs/results_summary.txt")

banner(f"DONE - final validation accuracy {val_acc * 100:.2f}%  (total {total_time / 60:.1f} minutes)")
