"""
Day 7 - Data Cleaning
Cleans students_performance.csv and saves cleaned_student_performance.csv
"""

import pandas as pd

# --- 1. Load raw data ---
df = pd.read_csv("students_performance.csv")
print(f"Raw shape: {df.shape}")

# --- 2. Check missing values ---
print("\nMissing values per column:")
print(df.isnull().sum())

# --- 3. Handle missing values (fill numeric columns with their column mean) ---
numeric_cols = ["Python", "Mathematics", "Statistics", "Machine_Learning", "Attendance"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# --- 4. Remove duplicates ---
dupes = df.duplicated().sum()
print(f"\nDuplicate rows found: {dupes}")
df = df.drop_duplicates().reset_index(drop=True)

# --- 5. Rename columns (lowercase, short names) ---
df = df.rename(columns={
    "Student_ID": "student_id",
    "Name": "name",
    "Age": "age",
    "Program": "program",
    "Python": "python",
    "Mathematics": "math",
    "Statistics": "stats",
    "Machine_Learning": "ml",
    "Attendance": "attendance",
})

# --- 6. Change data types (scores are whole marks out of 100 -> int) ---
score_cols = ["python", "math", "stats", "ml", "attendance"]
df[score_cols] = df[score_cols].round(0).astype(int)

# --- 7. Create new column: Average_Score ---
subject_cols = ["python", "math", "stats", "ml"]
df["Average_Score"] = df[subject_cols].mean(axis=1).round(2)


# --- 8. Create new column: Performance category ---
def performance(score):
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Average"
    else:
        return "Needs Improvement"


df["Performance"] = df["Average_Score"].apply(performance)

# --- 9. Sort & filter demo ---
df_sorted = df.sort_values("Average_Score", ascending=False)
print("\nTop 3 students by Average_Score:")
print(df_sorted[["name", "Average_Score"]].head(3).to_string(index=False))

ai_high_scorers = df[(df["program"] == "AI") & (df["Average_Score"] > 80)]
print(f"\nAI-program students scoring above 80: {len(ai_high_scorers)}")

# --- 10. Save cleaned data ---
df.to_csv("cleaned_student_performance.csv", index=False)
print(f"\nCleaned shape: {df.shape}")
print("Saved cleaned_student_performance.csv")
