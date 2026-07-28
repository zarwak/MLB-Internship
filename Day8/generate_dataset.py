"""
Day 8 — Step 0: build the dummy dataset.

Real student data is private, so I generated my own. The scores here aren't random
noise: I picked a hidden formula (study hours, attendance, previous score and sleep
all push the marks up) and then added random noise on top. That way a Linear
Regression model actually has a real pattern to find — and I already know what the
"right answer" roughly looks like, which makes it easy to tell if the model works.

I also deliberately dirty the data at the end (a few missing values, a few duplicate
rows), so the preprocessing step in data_preprocessing.py has something real to fix.

Run:  python generate_dataset.py
Out:  students_scores.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

# A fixed seed means the "random" numbers come out the same every time I run this.
# Without it, my dataset (and every metric in the README) would change on every run.
RNG = np.random.default_rng(42)

N_STUDENTS = 200
OUTPUT_FILE = Path(__file__).parent / "students_scores.csv"

FIRST_NAMES = [
    "Ali", "Sara", "Ahmed", "Fatima", "Usman", "Ayesha", "Hassan", "Zainab",
    "Bilal", "Maryam", "Hamza", "Noor", "Talha", "Iqra", "Danish", "Hira",
    "Omar", "Laiba", "Abdullah", "Mehwish", "Zara", "Saad", "Areeba", "Faizan",
]
LAST_NAMES = [
    "Khan", "Ahmed", "Raza", "Noor", "Ali", "Malik", "Tariq", "Iqbal",
    "Siddiqui", "Javed", "Aslam", "Shah", "Farooq", "Butt", "Chaudhry", "Qureshi",
]

PROGRAMS = ["AI", "DS", "SE", "CS"]
# How much each program nudges the final average, on top of everything else.
PROGRAM_EFFECT = {"AI": 1.5, "DS": 0.0, "SE": 2.0, "CS": -1.0}

# Each subject sits slightly above/below a student's overall level.
SUBJECT_OFFSET = {
    "Python": 2.0,
    "Mathematics": -1.5,
    "Statistics": -0.5,
    "Machine_Learning": 0.0,
}


def make_students(n):
    """Create the input columns — the facts about each student."""
    return pd.DataFrame({
        "Student_ID": [f"S{i:03d}" for i in range(1, n + 1)],
        "Name": [
            f"{RNG.choice(FIRST_NAMES)} {RNG.choice(LAST_NAMES)}" for _ in range(n)
        ],
        "Age": RNG.integers(18, 26, size=n),
        "Gender": RNG.choice(["Male", "Female"], size=n),
        "Program": RNG.choice(PROGRAMS, size=n, p=[0.3, 0.25, 0.25, 0.2]),
        # .round(1) keeps the CSV readable instead of 3.8471029384 hours of study.
        "Study_Hours": RNG.normal(4.0, 1.5, size=n).clip(0.5, 10).round(1),
        "Attendance": RNG.normal(85, 10, size=n).clip(50, 100).round(1),
        "Previous_Score": RNG.normal(70, 12, size=n).clip(35, 100).round(1),
        "Sleep_Hours": RNG.normal(7.0, 1.2, size=n).clip(4, 10).round(1),
        # 1 = has reliable internet at home, 0 = doesn't.
        "Internet_Access": RNG.choice([1, 0], size=n, p=[0.8, 0.2]),
    })


def add_subject_scores(df):
    """
    Apply the hidden formula to get each student's true ability, then spread that
    ability across the four subjects. This is the pattern the model has to rediscover.
    """
    true_average = (
        8.0
        + 2.0 * df["Study_Hours"]
        + 0.22 * df["Attendance"]
        + 0.42 * df["Previous_Score"]
        + 1.2 * df["Sleep_Hours"]
        + 3.0 * df["Internet_Access"]
        + df["Program"].map(PROGRAM_EFFECT)
        # Noise = everything I'm not measuring (mood, luck, a hard exam paper).
        # Without it the model would score a perfect R2 = 1.0, which never happens in real life.
        + RNG.normal(0, 3.0, size=len(df))
    )

    for subject, offset in SUBJECT_OFFSET.items():
        marks = true_average + offset + RNG.normal(0, 2.5, size=len(df))
        df[subject] = marks.clip(0, 100).round().astype(int)

    return df


def dirty_the_data(df):
    """
    Poke a few holes in the clean data on purpose, so preprocessing has real work to do:
    missing values to fill, and duplicate rows to drop.
    """
    missing_attendance = RNG.choice(df.index, size=6, replace=False)
    missing_study = RNG.choice(df.index, size=5, replace=False)
    df.loc[missing_attendance, "Attendance"] = np.nan
    df.loc[missing_study, "Study_Hours"] = np.nan

    # Copy 4 existing rows and stick them back on the end — classic data-entry duplicates.
    duplicates = df.sample(4, random_state=7)
    df = pd.concat([df, duplicates], ignore_index=True)
    return df


def main():
    df = make_students(N_STUDENTS)
    df = add_subject_scores(df)
    df = dirty_the_data(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Dummy dataset created:", OUTPUT_FILE)
    print(f"Rows: {len(df)}  (that's {N_STUDENTS} students + 4 duplicate rows)")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing values planted on purpose:")
    print(df.isnull().sum()[lambda s: s > 0].to_string())
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
