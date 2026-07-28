"""
Day 8 — Step 2: the Linear Regression model.

Preprocessing is done in data_preprocessing.py; this file is only about the model:
train it, score it, and draw the graphs. Like the preprocessing file, it's both a
runnable script and a toolbox the other scripts import.

    python linear_regression_model.py
"""

import matplotlib

# Use the non-interactive backend so saving PNGs works even with no window/display
# (needed on Streamlit Cloud and in any headless run).
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import FOLDER, prepare_everything

PLOT_STYLE = "seaborn-v0_8-whitegrid"
BLUE = "#2563eb"
ORANGE = "#f97316"


def train_model(X_train, y_train):
    """
    Fit the line. Linear Regression finds the one straight-line equation

        score = intercept + w1*feature1 + w2*feature2 + ...

    whose total squared error across all training students is as small as possible.
    .fit() is where the actual learning happens.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Predict on the unseen test students and score the result four ways."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mse,
        # RMSE is just sqrt(MSE) — it drags the number back into "marks" units
        # so it can be read the same way as MAE.
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, y_pred),
    }
    return y_pred, metrics


def comparison_table(y_test, y_pred):
    """Side-by-side Actual vs Predicted, plus how far off each prediction was."""
    table = pd.DataFrame({
        "Actual": y_test.values.round(2),
        "Predicted": y_pred.round(2),
    })
    table["Error"] = (table["Actual"] - table["Predicted"]).round(2)
    table["Abs_Error"] = table["Error"].abs()
    return table


def coefficient_table(model, feature_names):
    """
    What the model learned, in plain numbers.

    Because every feature was standardised, these coefficients are directly comparable:
    a bigger absolute value means that feature moves the predicted score more.
    """
    coefs = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_.round(3),
    })
    coefs["Impact"] = coefs["Coefficient"].abs()
    return coefs.sort_values("Impact", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(y_test, y_pred, path=FOLDER / "actual_vs_predicted.png"):
    """
    The headline chart. Each dot is one test student: actual score across, predicted up.
    The dashed line is "perfect prediction" — the closer the dots hug it, the better.
    """
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_test, y_pred, s=70, color=BLUE, alpha=0.7,
               edgecolor="white", linewidth=0.8, label="Test students")

    lo = min(y_test.min(), y_pred.min()) - 3
    hi = max(y_test.max(), y_pred.max()) + 3
    ax.plot([lo, hi], [lo, hi], "--", color=ORANGE, linewidth=2,
            label="Perfect prediction")

    ax.set_xlabel("Actual Average Score", fontsize=12)
    ax.set_ylabel("Predicted Average Score", fontsize=12)
    ax.set_title("Actual vs Predicted Student Average Scores",
                 fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_residuals(y_pred, y_test, path=FOLDER / "residuals_plot.png"):
    """
    Residual = actual - predicted. Plotted against the prediction, these should look
    like a shapeless cloud around zero. A curve or a funnel shape would mean a straight
    line was the wrong model for this data.
    """
    residuals = y_test.values - y_pred

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, s=60, color=BLUE, alpha=0.7,
               edgecolor="white", linewidth=0.8)
    ax.axhline(0, color=ORANGE, linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Average Score", fontsize=12)
    ax.set_ylabel("Residual (Actual - Predicted)", fontsize=12)
    ax.set_title("Residual Plot - are the errors random?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(coefs, path=FOLDER / "feature_importance.png"):
    """Horizontal bars of the learned coefficients: green pushes scores up, red pulls down."""
    plt.style.use(PLOT_STYLE)
    data = coefs.sort_values("Coefficient")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#dc2626" if c < 0 else "#16a34a" for c in data["Coefficient"]]
    ax.barh(data["Feature"], data["Coefficient"], color=colors, alpha=0.85)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_xlabel("Coefficient (effect on predicted score)", fontsize=12)
    ax.set_title("What the model thinks matters", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_error_distribution(y_test, y_pred, path=FOLDER / "error_distribution.png"):
    """Histogram of the errors — it should be centred on 0 and roughly bell-shaped."""
    errors = y_test.values - y_pred

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=12, color=BLUE, alpha=0.8, edgecolor="white")
    ax.axvline(0, color=ORANGE, linestyle="--", linewidth=2, label="No error")
    ax.set_xlabel("Prediction Error (Actual - Predicted)", fontsize=12)
    ax.set_ylabel("Number of students", fontsize=12)
    ax.set_title("Distribution of Prediction Errors", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_all_plots(y_test, y_pred, coefs):
    return [
        plot_actual_vs_predicted(y_test, y_pred),
        plot_residuals(y_pred, y_test),
        plot_feature_importance(coefs),
        plot_error_distribution(y_test, y_pred),
    ]


def main():
    print("=" * 62)
    print("DAY 8 - LINEAR REGRESSION MODEL")
    print("=" * 62)

    data = prepare_everything()
    model = train_model(data["X_train"], data["y_train"])
    y_pred, metrics = evaluate_model(model, data["X_test"], data["y_test"])

    print(f"\nTrained on {len(data['X_train'])} students, "
          f"tested on {len(data['X_test'])} unseen students.")
    print(f"\nIntercept: {model.intercept_:.3f}")
    print("\nLearned coefficients (biggest effect first):")
    coefs = coefficient_table(model, data["X_train"].columns)
    print(coefs[["Feature", "Coefficient"]].to_string(index=False))

    print("\nEvaluation metrics on the test set:")
    print(f"  MAE  : {metrics['MAE']:.3f}")
    print(f"  MSE  : {metrics['MSE']:.3f}")
    print(f"  RMSE : {metrics['RMSE']:.3f}")
    print(f"  R2   : {metrics['R2']:.4f}  ({metrics['R2'] * 100:.2f}% of the variation explained)")

    print("\nGraphs saved:")
    for path in save_all_plots(data["y_test"], y_pred, coefs):
        print(f"  - {Path(path).name}")


if __name__ == "__main__":
    main()
