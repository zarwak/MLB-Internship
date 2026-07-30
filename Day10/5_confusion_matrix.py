'''Confusion Matrix Comparison (5_confusion_matrix.py)
Generates a side-by-side heatmap comparison.'''

# Side-by-side comparison

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load and split
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Baseline
baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
cm_base = confusion_matrix(y_test, y_pred_base)

# Tuned (best parameters from previous - adjust if yours differ)
tuned = LogisticRegression(C=1, solver='liblinear', max_iter=1000, random_state=42)
tuned.fit(X_train, y_train)
y_pred_tuned = tuned.predict(X_test)
cm_tuned = confusion_matrix(y_test, y_pred_tuned)

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Baseline Model')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Tuned Model (C=1, solver=liblinear)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('comparison_cm.png')
print("✅ Comparison confusion matrix saved as 'comparison_cm.png'")