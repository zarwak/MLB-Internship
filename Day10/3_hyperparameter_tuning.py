'''Hyperparameter Tuning Script (3_hyperparameter_tuning.py)
Uses GridSearchCV to find the best hyperparameters.'''

#GridSearchCV implementation

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define hyperparameter grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],          # Inverse of regularization strength
    'solver': ['liblinear', 'lbfgs'],       # Optimization algorithms
    'penalty': ['l2']                       # Regularization type (liblinear supports l1/l2, but l2 is safest)
}

# Note: 'liblinear' works with l2, 'lbfgs' works with l2. We'll just use l2 for simplicity.
# Actually, let's make a robust grid:
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2']  # Both solvers support l2
}

# GridSearch with 5-fold cross-validation
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("="*50)
print("HYPERPARAMETER TUNING RESULTS")
print("="*50)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Evaluate tuned model on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nTUNED MODEL PERFORMANCE ON TEST SET")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")