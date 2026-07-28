"""
Day 8 — Mini Project: Student Score Prediction System.

This is the whole thing end to end, in one run:

    load -> preprocess -> train -> predict -> evaluate -> compare -> plot

It reuses the functions from data_preprocessing.py and linear_regression_model.py
rather than copy-pasting them, so there's only ever one version of each step.

    python student_score_prediction.py
"""

from pathlib import Path

import pandas as pd

from data_preprocessing import (
    CLEAN_FILE,
    RAW_FILE,
    build_features,
    clean_data,
    load_data,
    split_and_scale,
)
from linear_regression_model import (
    coefficient_table,
    comparison_table,
    evaluate_model,
    save_all_plots,
    train_model,
)

LINE = "=" * 68


def header(step, title):
    print(f"\n{LINE}\nSTEP {step}: {title}\n{LINE}")


def predict_new_student(model, scaler, feature_columns, student):
    """
    Score a brand-new student who was never in the CSV.

    The catch: the new student's data has to go through the *exact* same transformations
    the training data went through — same one-hot columns in the same order, same scaler.
    reindex(..., fill_value=0) adds back any dummy column this student doesn't trigger
    (e.g. Program_SE = 0 for an AI student), which keeps the column order lined up.
    """
    row = pd.DataFrame([student])
    row = pd.get_dummies(row, columns=["Gender", "Program"], drop_first=False)
    row = row.reindex(columns=feature_columns, fill_value=0).astype(float)
    # Keep it as a DataFrame with the same column names the model was trained on,
    # otherwise sklearn warns that the feature names went missing.
    row_scaled = pd.DataFrame(scaler.transform(row), columns=feature_columns)
    return float(model.predict(row_scaled)[0])


def main():
    print(LINE)
    print("DAY 8 MINI PROJECT - STUDENT SCORE PREDICTION SYSTEM".center(68))
    print(LINE)

    # ---------------------------------------------------------------- STEP 1
    header(1, "LOAD THE DATASET")
    raw = load_data(RAW_FILE)
    print(f"File     : {RAW_FILE.name}")
    print(f"Shape    : {raw.shape[0]} rows x {raw.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(raw.head().to_string(index=False))

    # ---------------------------------------------------------------- STEP 2
    header(2, "PREPROCESS THE DATA")
    print("Problems found in the raw file:")
    missing = raw.isnull().sum()
    for col, n in missing[missing > 0].items():
        print(f"   - {col}: {n} missing values")
    print(f"   - {raw.duplicated().sum()} duplicate rows")

    df, report = clean_data(raw)
    print("\nWhat I did about it:")
    print(f"   1. Dropped {report['duplicates_found']} duplicate rows")
    print("   2. Filled missing numbers with the column median")
    print("   3. Dropped Student_ID and Name (unique labels, no predictive value)")
    print("   4. Created the target column: Average_Score")
    print(f"\nClean dataset: {report['rows_after']} rows, "
          f"{report['missing_after']} missing values left")

    X, y = build_features(df)
    print(f"\n5. One-hot encoded Gender and Program -> {X.shape[1]} feature columns:")
    print("   " + ", ".join(X.columns))

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"\n6. Train/test split 80/20 -> {len(X_train)} train, {len(X_test)} test")
    print("7. Scaled the features with StandardScaler (fitted on the train set only)")

    df.to_csv(CLEAN_FILE, index=False)
    print(f"\nCleaned data saved to {CLEAN_FILE.name}")

    # ---------------------------------------------------------------- STEP 3
    header(3, "TRAIN THE LINEAR REGRESSION MODEL")
    model = train_model(X_train, y_train)
    coefs = coefficient_table(model, X_train.columns)
    print(f"Intercept: {model.intercept_:.3f}")
    print("\nCoefficients, strongest effect first:")
    print(coefs[["Feature", "Coefficient"]].to_string(index=False))
    print("\n(Features were standardised, so these numbers are directly comparable:")
    print(" a +1 standard-deviation change in the feature moves the predicted score")
    print(" by roughly the coefficient, in marks.)")

    # ---------------------------------------------------------------- STEP 4
    header(4, "PREDICT ON THE UNSEEN TEST STUDENTS")
    y_pred, metrics = evaluate_model(model, X_test, y_test)
    print(f"Predicted average scores for all {len(y_pred)} test students.")
    print(f"Predicted range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    print(f"Actual range   : {y_test.min():.2f} to {y_test.max():.2f}")

    # ---------------------------------------------------------------- STEP 5
    header(5, "EVALUATION METRICS")
    print(f"MAE  (Mean Absolute Error)     : {metrics['MAE']:.3f}")
    print("       -> on average the prediction misses by this many marks")
    print(f"MSE  (Mean Squared Error)      : {metrics['MSE']:.3f}")
    print("       -> errors squared before averaging, so big misses hurt more")
    print(f"RMSE (Root Mean Squared Error) : {metrics['RMSE']:.3f}")
    print("       -> MSE back in marks, comparable to MAE")
    print(f"R2   (R-squared)               : {metrics['R2']:.4f}")
    print(f"       -> the model explains {metrics['R2'] * 100:.2f}% of the variation in scores")

    # ---------------------------------------------------------------- STEP 6
    header(6, "ACTUAL vs PREDICTED COMPARISON TABLE")
    table = comparison_table(y_test, y_pred)
    print(table.head(15).to_string(index=False))
    print(f"\n... showing 15 of {len(table)} test students.")
    print(f"Best  prediction: off by {table['Abs_Error'].min():.2f} marks")
    print(f"Worst prediction: off by {table['Abs_Error'].max():.2f} marks")
    within_5 = (table["Abs_Error"] <= 5).mean() * 100
    print(f"{within_5:.1f}% of predictions land within 5 marks of the real score.")

    # ---------------------------------------------------------------- STEP 7
    header(7, "GRAPHS")
    for path in save_all_plots(y_test, y_pred, coefs):
        print(f"   saved: {Path(path).name}")

    # ---------------------------------------------------------------- STEP 8
    header(8, "PREDICT FOR A BRAND-NEW STUDENT")
    new_student = {
        "Age": 21,
        "Study_Hours": 6.0,
        "Attendance": 92.0,
        "Previous_Score": 78.0,
        "Sleep_Hours": 7.5,
        "Internet_Access": 1,
        "Gender": "Female",
        "Program": "AI",
    }
    for key, value in new_student.items():
        print(f"   {key:<16}: {value}")
    score = predict_new_student(model, scaler, X_train.columns, new_student)
    print(f"\n   PREDICTED AVERAGE SCORE: {score:.2f} / 100")
    print(f"   (give or take about {metrics['RMSE']:.1f} marks, based on the RMSE)")

    print(f"\n{LINE}\nDone.\n{LINE}")


if __name__ == "__main__":
    main()
