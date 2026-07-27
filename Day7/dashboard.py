"""
Day 7 - Mini Project: Student Performance Dashboard
Run with: streamlit run dashboard.py
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "cleaned_student_performance.csv"))
subject_cols = ["python", "math", "stats", "ml"]
subject_labels = {"python": "Python", "math": "Mathematics", "stats": "Statistics", "ml": "Machine Learning"}

subject_avg = df[subject_cols].mean().rename(subject_labels)
top5 = df.nlargest(5, "Average_Score")[["name", "program", "Average_Score", "Performance"]]
needs_improvement = df[df["Performance"] == "Needs Improvement"][["name", "program", "Average_Score"]]
best_subject = subject_avg.idxmax()

st.title("Student Performance Dashboard")
st.caption("Day 7 mini project - MLB Internship")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", len(df))
col2.metric("Class Average", f"{df['Average_Score'].mean():.1f}")
col3.metric("Highest-Avg Subject", best_subject, f"{subject_avg.max():.1f}")
col4.metric("Need Improvement", len(needs_improvement))

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Average Score per Subject")
    fig, ax = plt.subplots()
    subject_avg.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_ylabel("Average Score")
    ax.set_xticklabels(subject_avg.index, rotation=0)
    st.pyplot(fig)

with right:
    st.subheader("Performance Category Split")
    st.image(os.path.join(BASE_DIR, "pie_performance.png"))

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Top 5 Students")
    st.dataframe(top5, hide_index=True, use_container_width=True)

with right:
    st.subheader("Students Needing Improvement")
    if needs_improvement.empty:
        st.write("None - everyone is scoring 70+.")
    else:
        st.dataframe(needs_improvement, hide_index=True, use_container_width=True)

st.divider()
st.subheader("Marks Distribution Across Subjects")
st.image(os.path.join(BASE_DIR, "boxplot_subjects.png"))

st.divider()
st.subheader("Full Cleaned Dataset")
program_filter = st.selectbox("Filter by program", ["All"] + sorted(df["program"].unique().tolist()))
table = df if program_filter == "All" else df[df["program"] == program_filter]
st.dataframe(table, hide_index=True, use_container_width=True)
