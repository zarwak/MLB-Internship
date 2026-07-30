'''4. Final Prediction Pipeline (4_final_pipeline.py)
Combines everything into a reusable pipeline 
(including preprocessing like StandardScaler).'''

# Scalable pipeline with StandardScaler

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create pipeline (Scaling + Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),          # Crucial for Logistic Regression
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Hyperparameter grid (note the 'classifier__' prefix)
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs']
}

# GridSearch
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("="*50)
print("FINAL PIPELINE WITH STANDARDIZATION & TUNING")
print("="*50)
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Save the model for the app
with open('breast_cancer_pipeline.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)
print("\n✅ Model saved as 'breast_cancer_pipeline.pkl'")