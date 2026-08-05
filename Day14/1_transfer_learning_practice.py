"""
Day 14 - Practice Script: Transfer Learning basics
==================================================

Two practices, exactly as the assignment asks:

  PRACTICE 1 - Load a pre-trained MobileNetV2
               Explore its architecture
               Freeze the base model
               Add my own classification head

  PRACTICE 2 - Load Cats vs Dogs from TensorFlow Datasets (TFDS)
               Preprocess and resize the images
               Split into training and validation sets

Run it with:   python 1_transfer_learning_practice.py

Everything it prints is also explained in the README.
"""

import os
from pathlib import Path

# Quieten TensorFlow's start-up chatter (0=all, 1=no INFO, 2=no WARNING)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save figures to disk instead of opening windows
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Save every picture next to this file, no matter where the script is run from
HERE = Path(__file__).parent
IMAGES = HERE / "images"
IMAGES.mkdir(exist_ok=True)

keras.utils.set_random_seed(42)  # same random numbers every run

IMG_SIZE = 160  # MobileNetV2 likes square images; 160x160 is a standard choice
BATCH_SIZE = 32


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ==========================================================================
# PRACTICE 1 - Load MobileNetV2, explore it, freeze it, add a new head
# ==========================================================================

banner("PRACTICE 1.1 - Load the pre-trained MobileNetV2")

# weights="imagenet"    -> download the weights learned from 1.4 million photos
# include_top=False     -> LEAVE OFF the original 1000-class output layer,
#                          because we only want 2 classes (cat / dog)
# input_shape           -> the picture size we will feed it
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

print(f"Loaded MobileNetV2")
print(f"  Input shape  : {base_model.input_shape}")
print(f"  Output shape : {base_model.output_shape}")
print(f"  Layers       : {len(base_model.layers)}")
print(f"  Parameters   : {base_model.count_params():,}")

print(
    "\nWhat the output shape means:\n"
    f"  A {IMG_SIZE}x{IMG_SIZE} picture (76,800 numbers) goes in.\n"
    f"  {base_model.output_shape[1]}x{base_model.output_shape[2]}x{base_model.output_shape[3]} numbers come out.\n"
    "  Those numbers are not pixels any more - they are a description of\n"
    "  WHAT IS IN the picture (fur, ears, eyes, whiskers, ...).\n"
    "  That description is the thing we are borrowing."
)

banner("PRACTICE 1.2 - Explore the architecture")

# The full summary is ~150 layers long, so print the first and last few
print("\nFirst 8 layers:")
for i, layer in enumerate(base_model.layers[:8]):
    print(f"  {i:3d}  {layer.name:30s} {str(layer.output.shape):24s} {layer.__class__.__name__}")

print("\nLast 8 layers:")
for i, layer in enumerate(base_model.layers[-8:], start=len(base_model.layers) - 8):
    print(f"  {i:3d}  {layer.name:30s} {str(layer.output.shape):24s} {layer.__class__.__name__}")

# Count how many of each kind of layer there is
kinds = {}
for layer in base_model.layers:
    kinds[layer.__class__.__name__] = kinds.get(layer.__class__.__name__, 0) + 1

print("\nLayer types inside MobileNetV2:")
for name, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
    print(f"  {count:4d} x {name}")

print(
    "\nNotice the DepthwiseConv2D layers - that is MobileNetV2's trick.\n"
    "A normal convolution mixes 'where' and 'which colour channel' in one go.\n"
    "A depthwise-separable convolution does those as two cheap steps instead\n"
    "of one expensive step, which is why this model is small enough for phones."
)

# Show how the picture shrinks as it travels through the network
print("\nHow the image shrinks on its way through:")
seen = set()
for layer in base_model.layers:
    shape = layer.output.shape
    if len(shape) == 4 and shape[1] is not None and shape[1] not in seen:
        seen.add(shape[1])
        print(f"  {layer.name:30s} -> {shape[1]:3d} x {shape[2]:3d} x {shape[3]:4d}")

banner("PRACTICE 1.3 - Freeze the base model")

print(f"BEFORE freezing: trainable params = {base_model.count_params():,}")

base_model.trainable = False  # <- this one line is the whole idea of freezing

trainable = sum(int(tf.size(w)) for w in base_model.trainable_weights)
frozen = sum(int(tf.size(w)) for w in base_model.non_trainable_weights)
print(f"AFTER  freezing: trainable params = {trainable:,}")
print(f"                 frozen params    = {frozen:,}")
print(
    "\nFreezing means: 'do not change these numbers during training'.\n"
    "MobileNetV2 already knows how to see. We do not want to ruin that\n"
    "knowledge with our small dataset - we only want to use it."
)

banner("PRACTICE 1.4 - Add my own classification head")

# A tiny data-augmentation block. It only runs during training; Keras
# switches it off automatically when predicting.
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
x = data_augmentation(inputs)
# MobileNetV2 was trained on pixels scaled to -1..1, so feed it the same thing.
x = layers.Rescaling(1.0 / 127.5, offset=-1, name="rescale_to_minus1_1")(x)
x = base_model(x, training=False)  # (5, 5, 1280) - the borrowed features
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)  # -> (1280,)
x = layers.Dropout(0.2, name="dropout")(x)
outputs = layers.Dense(1, activation="sigmoid", name="cat_or_dog")(x)  # -> 1 number

model = keras.Model(inputs, outputs, name="mobilenetv2_cats_vs_dogs")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

total = model.count_params()
trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
print(f"\nTotal parameters     : {total:,}")
print(f"Trainable parameters : {trainable:,}  ({100 * trainable / total:.2f}%)")
print(f"Frozen parameters    : {total - trainable:,}  ({100 * (total - trainable) / total:.2f}%)")
print(
    "\nThat percentage is the headline of the whole day:\n"
    "we are only training a tiny fraction of the model, and borrowing the rest."
)

print(
    "\nWhy GlobalAveragePooling2D and not Flatten?\n"
    "  Flatten would turn (5,5,1280) into 32,000 numbers, and a Dense layer\n"
    "  after that would need 32,001 weights per neuron.\n"
    "  GlobalAveragePooling2D averages each of the 1280 feature maps down to\n"
    "  ONE number -> 1280 numbers, so the head stays tiny (1,281 weights)."
)


# ==========================================================================
# PRACTICE 2 - Load Cats vs Dogs from TFDS, preprocess, split
# ==========================================================================

banner("PRACTICE 2.1 - Load the Cats vs Dogs dataset from TFDS")

import tensorflow_datasets as tfds

# The dataset ships with ONE split called "train". We slice it ourselves.
#   train[:80%]  -> first 80% of the images  -> training set
#   train[80%:]  -> last  20% of the images  -> validation set
(train_raw, val_raw), info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    with_info=True,
    as_supervised=True,  # give me (image, label) pairs, not a dictionary
)

class_names = info.features["label"].names  # ['cat', 'dog']
n_total = info.splits["train"].num_examples
n_train = int(train_raw.cardinality())
n_val = int(val_raw.cardinality())

print(f"Dataset      : {info.name}")
print(f"Total images : {n_total:,}")
print(f"Classes      : {class_names}")
print(f"Training     : {n_train:,} images ({100 * n_train / n_total:.0f}%)")
print(f"Validation   : {n_val:,} images ({100 * n_val / n_total:.0f}%)")
print(
    "\nNote: the original Kaggle dataset has 25,000 pictures. TFDS drops 1,738\n"
    "of them because those files are corrupted, which is why we get 23,262."
)

print("\nThe raw images are all DIFFERENT sizes. First 5:")
for image, label in train_raw.take(5):
    print(f"  {str(image.shape):20s} label={int(label)} ({class_names[int(label)]})")
print("  -> this is exactly why we must resize before training.")

banner("PRACTICE 2.2 - Preprocess and resize")


def preprocess(image, label):
    """Make every picture the same size, and make the numbers decimals."""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # -> (160, 160, 3)
    return image, label


AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    train_raw.map(preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(1000)  # mix the cats and dogs up
    .batch(BATCH_SIZE)  # 32 pictures at a time
    .prefetch(AUTOTUNE)  # load the next batch while the CPU trains on this one
)
val_ds = (
    val_raw.map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

for images, labels in train_ds.take(1):
    print(f"One batch of images : {images.shape}  dtype={images.dtype}")
    print(f"One batch of labels : {labels.shape}  dtype={labels.dtype}")
    print(f"Pixel range         : {float(tf.reduce_min(images)):.1f} .. {float(tf.reduce_max(images)):.1f}")
    print(f"Batches per epoch   : train={len(train_ds)}, validation={len(val_ds)}")

print(
    "\nWe do NOT divide by 255 here, because the Rescaling layer inside the\n"
    "model already turns 0..255 into -1..1. Doing it twice would be a bug."
)

# --- a picture of some training images --------------------------------------
plt.figure(figsize=(12, 6))
for images, labels in train_ds.take(1):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])], fontsize=11)
        plt.axis("off")
plt.suptitle("Cats vs Dogs - 10 resized training images (160x160)", fontsize=14)
plt.tight_layout()
plt.savefig(IMAGES / "practice_samples.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"\nSaved: {IMAGES / 'practice_samples.png'}")

# --- a picture of what augmentation does ------------------------------------
plt.figure(figsize=(12, 3))
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
plt.suptitle("Data augmentation - free extra training images", fontsize=13)
plt.tight_layout()
plt.savefig(IMAGES / "practice_augmentation.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"Saved: {IMAGES / 'practice_augmentation.png'}")

banner("PRACTICE 2.3 - Sanity check: is the untrained head any good?")

# The head is randomly initialised, so it should be around 50% - a coin flip.
loss, acc = model.evaluate(val_ds.take(20), verbose=0)
print(f"Untrained head on 640 validation images: accuracy = {acc * 100:.2f}%, loss = {loss:.4f}")
print("About 50% is exactly right - it has not learned anything yet.")
print("\nThe mini project (2_cats_vs_dogs_transfer_learning.py) trains it properly.")

banner("PRACTICE COMPLETE")
