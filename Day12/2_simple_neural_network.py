"""
Day 12 - Practice 2: Build a Simple Neural Network (Input -> Hidden -> Output)
==============================================================================

GOAL
----
Build the smallest network that still counts as a real neural network:
    ONE input layer, ONE hidden layer, ONE output layer.
Then print model.summary() and explain, line by line, what every number means.

THE PROBLEM WE PRETEND TO SOLVE
-------------------------------
Say we have 4 measurements about a flower (sepal length, sepal width,
petal length, petal width) and we want to predict which of 3 species it is.
   Input  : 4 numbers
   Output : 3 probabilities (one per species)

Run with:
    python 2_simple_neural_network.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ----------------------------------------------------------------------
# The architecture, spelled out
# ----------------------------------------------------------------------
N_FEATURES = 4      # size of the input layer
N_HIDDEN = 8        # neurons in the hidden layer
N_CLASSES = 3       # size of the output layer

print_header("THE ARCHITECTURE WE ARE BUILDING")
print(f"""
    INPUT LAYER            HIDDEN LAYER           OUTPUT LAYER
    {N_FEATURES} features      ->     {N_HIDDEN} neurons      ->      {N_CLASSES} neurons
    (raw numbers)          (ReLU)                 (Softmax)

    Every neuron in one layer connects to EVERY neuron in the next.
    That is what "Dense" (also called "fully connected") means.
""")


# ----------------------------------------------------------------------
# Build it
# ----------------------------------------------------------------------
print_header("BUILDING THE MODEL")

model = keras.Sequential(
    [
        # LAYER 0 - the Input layer.
        # This holds NO weights. It only tells Keras the shape of one sample.
        # shape=(4,) means "each sample is a vector of 4 numbers".
        # The batch dimension is left out on purpose - Keras adds it as None.
        layers.Input(shape=(N_FEATURES,), name="input_layer"),

        # LAYER 1 - the Hidden layer.
        # 8 neurons. Each neuron computes:  output = relu(w . x + b)
        # where w is a weight vector of length 4 and b is one bias number.
        layers.Dense(N_HIDDEN, activation="relu", name="hidden_layer"),

        # LAYER 2 - the Output layer.
        # 3 neurons, one per class. Softmax turns the 3 raw scores into
        # 3 probabilities that add up to exactly 1.0.
        layers.Dense(N_CLASSES, activation="softmax", name="output_layer"),
    ],
    name="simple_ann",
)

# Compiling attaches the three things needed for training:
#   optimizer -> HOW weights get updated
#   loss      -> WHAT we are trying to minimise
#   metrics   -> what we want reported (not used for training)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Model built and compiled.")


# ----------------------------------------------------------------------
# The summary
# ----------------------------------------------------------------------
print_header("MODEL SUMMARY")
model.summary()


# ----------------------------------------------------------------------
# Explaining the summary, layer by layer
# ----------------------------------------------------------------------
print_header("EXPLAINING EVERY LAYER")

print("""
--------------------------------------------------------------------
INPUT LAYER  -  shape (None, 4)  -  0 parameters
--------------------------------------------------------------------
  What it does : Nothing computational. It is a doorway that declares
                 "each sample arriving here is 4 numbers long".

  Why 0 params : There is nothing to learn. No weights, no biases.
                 It does not even appear as a row in newer Keras
                 summaries - only its shape shows up.

  What is None : The batch size. It is None because the model does not
                 care whether you feed it 1 sample or 10,000 at a time.
                 It is decided at run time, not at build time.
""")

print(f"""
--------------------------------------------------------------------
HIDDEN LAYER  -  Dense({N_HIDDEN}, activation='relu')  -  output (None, {N_HIDDEN})
--------------------------------------------------------------------
  What it does : This is where the actual learning happens. Each of the
                 {N_HIDDEN} neurons looks at ALL {N_FEATURES} inputs, multiplies each by
                 its own learned weight, adds them up, adds a bias, and
                 passes the result through ReLU.

  The maths    : z = (w1*x1 + w2*x2 + w3*x3 + w4*x4) + b
                 a = relu(z) = max(0, z)

  Parameter count:
                 weights = inputs x neurons = {N_FEATURES} x {N_HIDDEN} = {N_FEATURES * N_HIDDEN}
                 biases  = one per neuron   =     {N_HIDDEN}
                 TOTAL   = {N_FEATURES * N_HIDDEN} + {N_HIDDEN} = {N_FEATURES * N_HIDDEN + N_HIDDEN}

  Why ReLU     : Without an activation function, stacking Dense layers is
                 pointless - a chain of linear operations collapses into a
                 single linear operation. ReLU bends the line, which is
                 what lets the network learn non-linear patterns.
""")

print(f"""
--------------------------------------------------------------------
OUTPUT LAYER  -  Dense({N_CLASSES}, activation='softmax')  -  output (None, {N_CLASSES})
--------------------------------------------------------------------
  What it does : Produces the final answer. {N_CLASSES} neurons because we have
                 {N_CLASSES} classes. The neuron count of the output layer is
                 dictated by the problem, not by choice.

  Parameter count:
                 weights = {N_HIDDEN} x {N_CLASSES} = {N_HIDDEN * N_CLASSES}
                 biases  =     {N_CLASSES}
                 TOTAL   = {N_HIDDEN * N_CLASSES} + {N_CLASSES} = {N_HIDDEN * N_CLASSES + N_CLASSES}

  Why Softmax  : It converts {N_CLASSES} arbitrary real numbers into {N_CLASSES} probabilities
                 that sum to 1.0, e.g. [0.02, 0.91, 0.07]. Now the output
                 is readable as "91% confident it is class 1".

                 softmax(z_i) = e^(z_i) / sum_j e^(z_j)
""")

total = N_FEATURES * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_CLASSES + N_CLASSES
print(f"""
--------------------------------------------------------------------
TOTAL PARAMETERS
--------------------------------------------------------------------
  Hidden layer : {N_FEATURES * N_HIDDEN + N_HIDDEN}
  Output layer : {N_HIDDEN * N_CLASSES + N_CLASSES}
  ------------------
  TOTAL        : {total}   (Keras reports: {model.count_params()})

  "Trainable params"     -> numbers that gradient descent will change.
  "Non-trainable params" -> frozen numbers (0 here; you see these when
                            using pre-trained models or BatchNorm stats).
""")


# ----------------------------------------------------------------------
# Look at the actual weight tensors
# ----------------------------------------------------------------------
print_header("THE ACTUAL WEIGHT SHAPES")

for layer in model.layers:
    print(f"\nLayer: {layer.name}")
    print(f"  Output shape : {layer.output.shape}")
    print(f"  Parameters   : {layer.count_params()}")
    for w in layer.weights:
        kind = "weights" if "kernel" in w.name else "biases "
        print(f"    {kind} {str(w.shape):<12} -> {int(tf.size(w))} numbers")


# ----------------------------------------------------------------------
# Push one fake sample through it
# ----------------------------------------------------------------------
print_header("A FORWARD PASS ON ONE FAKE SAMPLE")

sample = tf.constant([[5.1, 3.5, 1.4, 0.2]])   # shape (1, 4)
prediction = model.predict(sample, verbose=0)

print(f"Input          : {sample.numpy()[0]}")
print(f"Output         : {prediction[0]}")
print(f"Sums to        : {prediction[0].sum():.6f}  (softmax always sums to 1)")
print(f"Predicted class: {prediction[0].argmax()}")
print("\nThe prediction is meaningless right now - the weights are still")
print("random because we never called model.fit(). But the plumbing works.")

print("\n" + "=" * 70)
print("  Next: 3_activation_experiments.py")
print("=" * 70 + "\n")
