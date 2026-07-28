"""
Day 8 — Step 1: Data Preprocessing.

A model can only learn from numbers, and only from *clean* numbers. This file is the
"prepare the data" half of the project. Every other script (the model script, the mini
project, the Streamlit app) imports its functions from here, so the preprocessing rules
live in exactly one place and can never drift apart.

Run it on its own and it also writes cleaned_students_scores.csv so I can eyeball
the result in Excel:

    python data_preprocessing.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths are built from THIS file's location, not from wherever the terminal happens to
# be. That way the scripts work whether I run them from inside Day8, from the repo root,
# or from Streamlit Cloud.
FOLDER = Path(__file__).parent
RAW_FILE = FOLDER / "students_scores.csv"
CLEAN_FILE = FOLDER / "cleaned_students_scores.csv"

SUBJECTS = ["Python", "Mathematics", "Statistics", "Machine_Learning"]

# The columns the model is allowed to learn from.
# Note what's NOT here: the four subject scores. Predicting the average FROM the subjects
# would be arithmetic, not machine learning — the model would just rediscover "divide by 4"
# and score a fake perfect 100%. So I predict from behaviour and background instead.
NUMERIC_FEATURES = [
    "Age",
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Sleep_Hours",
    "Internet_Access",
]
CATEGORICAL_FEATURES = ["Gender", "Program"]

TARGET = "Average_Score"


def load_data(path=RAW_FILE):
    """Read the CSV off disk into a pandas DataFrame."""
    return pd.read_csv(path)


def clean_data(df, verbose=False):
    """
    The actual cleaning, in the order that matters:

    1. drop duplicate rows      — same student counted twice would bias the model
    2. fill missing numbers     — with the median, which ignores outliers
    3. drop ID / Name           — unique labels, zero predictive value
    4. build the target column  — Average_Score, the thing we want to predict
    """
    report = {}

    # --- 1. duplicates ---------------------------------------------------------
    report["duplicates_found"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # --- 2. missing values -----------------------------------------------------
    # Median over mean: if one student logged 40 study hours by mistake, the mean gets
    # dragged upwards but the median (the middle value) barely moves.
    report["missing_before"] = df.isnull().sum().to_dict()
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # --- 3. useless columns ----------------------------------------------------
    # Student_ID and Name are different for literally every row. A model that "learns"
    # from them memorises students instead of learning a rule.
    df = df.drop(columns=[c for c in ["Student_ID", "Name"] if c in df.columns])

    # --- 4. the target ---------------------------------------------------------
    # axis=1 averages ACROSS the four subject columns for each student (a row average),
    # not down a single column.
    df[TARGET] = df[SUBJECTS].mean(axis=1).round(2)

    report["rows_after"] = len(df)
    report["missing_after"] = int(df.isnull().sum().sum())

    if verbose:
        print(f"Duplicate rows removed : {report['duplicates_found']}")
        print(f"Missing values left    : {report['missing_after']}")
        print(f"Rows ready for training: {report['rows_after']}")

    return df, report


def build_features(df):
    """
    Turn the cleaned table into X (inputs) and y (answer).

    The only tricky part is the text columns. 'Gender' and 'Program' hold words, and
    sklearn can't do maths on the word "Female". One-hot encoding turns each category
    into its own 0/1 column.

    drop_first=True deletes one column per category on purpose. If Gender_Male is 0,
    the student is obviously female — keeping both columns adds no information and makes
    the coefficients unstable (that's the "dummy variable trap"). The dropped category
    becomes the baseline everything else is compared against.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True)
    # get_dummies makes True/False columns; convert to 1/0 so everything is numeric.
    X = X.astype(float)

    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split first, scale second. That order is not optional.

    Train-test split: the model studies 80% of the students and is then examined on the
    20% it has never seen. Grading it on data it memorised would be like giving a student
    the exam paper as homework — the score would look great and mean nothing.

    Scaling: Age runs 18-25 while Previous_Score runs 35-100. StandardScaler rewrites
    every column to mean 0, std 1, so the model compares them on equal footing and the
    coefficients become directly comparable.

    fit_transform on train, plain transform on test: the scaler learns the average and
    spread from the TRAINING data only. If it peeked at the test set, information from
    the exam would leak into the study material — that's data leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def prepare_everything(path=RAW_FILE):
    """One call that runs the whole pipeline — used by the model script and the app."""
    df = load_data(path)
    clean_df, report = clean_data(df)
    X, y = build_features(clean_df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    return {
        "clean_df": clean_df,
        "report": report,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


def main():
    print("=" * 62)
    print("DAY 8 - DATA PREPROCESSING")
    print("=" * 62)

    df = load_data()
    print(f"\nRaw data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nMissing values in the raw file:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() or "  none")
    print(f"\nDuplicate rows in the raw file: {df.duplicated().sum()}")

    clean_df, report = clean_data(df, verbose=False)
    print("\n--- after cleaning ---")
    print(f"Duplicate rows removed : {report['duplicates_found']}")
    print(f"Missing values left    : {report['missing_after']}")
    print(f"Rows ready for training: {report['rows_after']}")
    print(f"Target column created  : {TARGET}")

    X, y = build_features(clean_df)
    print(f"\nFeatures after one-hot encoding ({X.shape[1]} columns):")
    for col in X.columns:
        print(f"   - {col}")

    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
    print(f"\nTrain set: {X_train.shape[0]} students (80%)")
    print(f"Test set : {X_test.shape[0]} students (20%)")

    clean_df.to_csv(CLEAN_FILE, index=False)
    print(f"\nCleaned data saved to: {CLEAN_FILE.name}")


if __name__ == "__main__":
    main()
