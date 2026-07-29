# app.py - Gradio version for Hugging Face
import gradio as gr
import pickle
import numpy as np
from sklearn.datasets import load_iris
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model and scaler
model_path = os.path.join(script_dir, 'logreg_iris.pkl')
scaler_path = os.path.join(script_dir, 'scaler_iris.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

iris = load_iris()
target_names = iris.target_names

def predict_species(sepal_len, sepal_wid, petal_len, petal_wid):
    """Predict Iris species based on measurements."""
    input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    
    # Create result message
    result = f"**Predicted species:** {target_names[pred]}\n\n"
    result += "**Probabilities:**\n"
    for name, p in zip(target_names, prob):
        result += f"- {name}: {p:.2%}\n"
    return result

# Create the Gradio interface
inputs = [
    gr.Slider(4.0, 8.0, value=5.8, label="Sepal length (cm)"),
    gr.Slider(2.0, 4.5, value=3.0, label="Sepal width (cm)"),
    gr.Slider(1.0, 7.0, value=4.0, label="Petal length (cm)"),
    gr.Slider(0.1, 2.5, value=1.2, label="Petal width (cm)"),
]

outputs = gr.Markdown(label="Prediction Result")

title = "🌸 Iris Flower Species Predictor"
description = "Enter the sepal and petal measurements to predict the Iris flower species."

demo = gr.Interface(
    fn=predict_species,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description
)

if __name__ == "__main__":
    demo.launch()