"""
Day 12 - Practice 1: Install, Verify and Import TensorFlow / Keras
==================================================================

GOAL
----
Prove that TensorFlow is actually installed and working on this machine
before we try to build anything with it. Most "my model won't run" problems
on Day 1 of deep learning are really installation problems.

WHAT THIS SCRIPT CHECKS
-----------------------
1. TensorFlow imports without error, and what version it is
2. Keras is reachable (in TF 2.x, Keras ships INSIDE TensorFlow)
3. What hardware TF can see (CPU / GPU)
4. A real tensor computation actually runs and gives the right answer

INSTALL COMMAND (only needed once)
---------------------------------
    pip install tensorflow

Run with:
    python 1_tensorflow_setup.py
"""

import sys
import platform


def print_header(title):
    """Small helper so the console output is readable instead of a wall of text."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ----------------------------------------------------------------------
# STEP 1: The environment itself
# ----------------------------------------------------------------------
print_header("STEP 1: Python Environment")
print(f"Python version   : {sys.version.split()[0]}")
print(f"Platform         : {platform.system()} {platform.release()}")
print(f"Interpreter path : {sys.executable}")
print("\nWhy this matters: TensorFlow is compiled for specific Python versions.")
print("If `pip install tensorflow` fails, a mismatched Python version is the")
print("most common reason.")


# ----------------------------------------------------------------------
# STEP 2: Import TensorFlow
# ----------------------------------------------------------------------
print_header("STEP 2: Importing TensorFlow")

try:
    import tensorflow as tf
    print("SUCCESS - TensorFlow imported.")
    print(f"TensorFlow version : {tf.__version__}")
except ImportError as e:
    print("FAILED - TensorFlow is not installed.")
    print(f"Error: {e}")
    print("\nFix it with:  pip install tensorflow")
    sys.exit(1)


# ----------------------------------------------------------------------
# STEP 3: Import Keras
# ----------------------------------------------------------------------
print_header("STEP 3: Importing Keras")

from tensorflow import keras
from tensorflow.keras import layers

print("SUCCESS - Keras imported.")
print(f"Keras version : {keras.__version__}")
print("\nWhat is Keras?")
print("  TensorFlow is the engine (fast math on tensors, autodiff, GPU support).")
print("  Keras is the steering wheel - the friendly API you actually build")
print("  models with. Since TF 2.0 they ship together, so `pip install")
print("  tensorflow` gives you both.")


# ----------------------------------------------------------------------
# STEP 4: What hardware can TensorFlow see?
# ----------------------------------------------------------------------
print_header("STEP 4: Available Devices")

cpus = tf.config.list_physical_devices("CPU")
gpus = tf.config.list_physical_devices("GPU")

print(f"CPUs detected : {len(cpus)}")
for d in cpus:
    print(f"   - {d.name}")

print(f"GPUs detected : {len(gpus)}")
if gpus:
    for d in gpus:
        print(f"   - {d.name}")
    print("\nGPU found - training will be noticeably faster.")
else:
    print("   (none)")
    print("\nNo GPU found. That is completely fine for today - the Fashion MNIST")
    print("ANN in this folder trains in well under a minute on CPU.")


# ----------------------------------------------------------------------
# STEP 5: Prove the engine actually computes
# ----------------------------------------------------------------------
print_header("STEP 5: Running a Real Tensor Computation")

# A tensor is just an n-dimensional array that TF can differentiate through.
a = tf.constant([[1.0, 2.0],
                 [3.0, 4.0]])
b = tf.constant([[5.0, 6.0],
                 [7.0, 8.0]])

print("Tensor a:")
print(a.numpy())
print("\nTensor b:")
print(b.numpy())

print(f"\nShape of a  : {a.shape}")
print(f"Dtype of a  : {a.dtype}")

added = tf.add(a, b)
print("\na + b (element-wise addition):")
print(added.numpy())

matmul = tf.matmul(a, b)
print("\na @ b (matrix multiplication - this is what a Dense layer does):")
print(matmul.numpy())
print("Check by hand: row1 x col1 = 1*5 + 2*7 = 19  <- matches [0][0]")

# Automatic differentiation - the single feature that makes training possible.
print("\nAutomatic differentiation check:")
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2          # y = x^2
grad = tape.gradient(y, x)   # dy/dx = 2x = 6 when x = 3
print(f"   y = x^2, at x = 3.0  ->  dy/dx = {grad.numpy()}  (expected 6.0)")
print("   This is how a neural network learns: TF tracks every operation and")
print("   computes the gradient of the loss with respect to every weight.")


# ----------------------------------------------------------------------
# STEP 6: Build a throwaway model to confirm Keras works end to end
# ----------------------------------------------------------------------
print_header("STEP 6: Building a Throwaway Keras Model")

smoke_test = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(3, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
smoke_test.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Model built and compiled successfully.")
print(f"Total trainable parameters: {smoke_test.count_params()}")


# ----------------------------------------------------------------------
# DONE
# ----------------------------------------------------------------------
print_header("VERIFICATION COMPLETE")
print("TensorFlow  : OK")
print("Keras       : OK")
print("Tensor math : OK")
print("Gradients   : OK")
print("Model build : OK")
print("\nEnvironment is ready. Next: 2_simple_neural_network.py")
print("=" * 70 + "\n")
