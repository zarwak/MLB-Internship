# iris_classification_project/iris_classifier.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def evaluate_model(name, y_test, y_pred, target_names):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\n{name} Evaluation:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    return acc, prec, rec, f1

def main():
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scaling for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------- Logistic Regression ----------
    logreg = LogisticRegression(multi_class='ovr', max_iter=200, random_state=42)
    logreg.fit(X_train_scaled, y_train)
    y_pred_lr = logreg.predict(X_test_scaled)
    lr_metrics = evaluate_model("Logistic Regression", y_test, y_pred_lr, target_names)

    # ---------- Decision Tree ----------
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)   # no scaling needed
    y_pred_dt = dt.predict(X_test)
    dt_metrics = evaluate_model("Decision Tree", y_test, y_pred_dt, target_names)

    # ---------- Confusion Matrices ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name, y_pred in zip(axes, ['Logistic Regression', 'Decision Tree'], [y_pred_lr, y_pred_dt]):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_title(f'Confusion Matrix – {name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('../confusion_matrix.png')   # overwrites previous
    plt.show()

    # ---------- Save models and scaler for later use (Streamlit) ----------
    with open('logreg_iris.pkl', 'wb') as f:
        pickle.dump(logreg, f)
    with open('dt_iris.pkl', 'wb') as f:
        pickle.dump(dt, f)
    with open('scaler_iris.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("\nModels and scaler saved as .pkl files.")

    # ---------- Classification reports ----------
    print("\n--- Logistic Regression Report ---")
    print(classification_report(y_test, y_pred_lr, target_names=target_names))
    print("\n--- Decision Tree Report ---")
    print(classification_report(y_test, y_pred_dt, target_names=target_names))

    # ---------- Sample predictions ----------
    print("\nSample Predictions (Logistic Regression):")
    for i in range(5):
        pred = logreg.predict(X_test_scaled[i].reshape(1, -1))[0]
        actual = y_test[i]
        print(f"Sample {i}: Predicted = {target_names[pred]}, Actual = {target_names[actual]}")

if __name__ == "__main__":
    main()