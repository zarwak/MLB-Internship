"""
Day 7 - Mini Project: Student Performance Dashboard (Gradio version)
Run with: python gradio_app.py
"""

import os

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "cleaned_student_performance.csv"))

subject_cols = ["python", "math", "stats", "ml"]
subject_labels = {"python": "Python", "math": "Mathematics", "stats": "Statistics", "ml": "Machine Learning"}

subject_avg = df[subject_cols].mean().rename(subject_labels)
top5 = df.nlargest(5, "Average_Score")[["name", "program", "Average_Score", "Performance"]]
needs_improvement = df[df["Performance"] == "Needs Improvement"][["name", "program", "Average_Score"]]
best_subject = subject_avg.idxmax()

summary_md = f"""
| Total Students | Class Average | Highest-Avg Subject | Need Improvement |
|---|---|---|---|
| {len(df)} | {df['Average_Score'].mean():.1f} | {best_subject} ({subject_avg.max():.1f}) | {len(needs_improvement)} |
"""

needs_improvement_display = (
    needs_improvement if not needs_improvement.empty
    else pd.DataFrame({"message": ["None - everyone is scoring 70+."]})
)


def subject_bar_chart():
    fig, ax = plt.subplots()
    subject_avg.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_ylabel("Average Score")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def filter_table(program):
    return df if program == "All" else df[df["program"] == program]


with gr.Blocks(title="Student Performance Dashboard") as demo:
    gr.Markdown("# Student Performance Dashboard\nDay 7 mini project - MLB Internship")
    gr.Markdown(summary_md)

    with gr.Row():
        gr.Plot(value=subject_bar_chart(), label="Average Score per Subject")
        gr.Image(value=os.path.join(BASE_DIR, "pie_performance.png"), label="Performance Category Split")

    with gr.Row():
        gr.Dataframe(value=top5, label="Top 5 Students")
        gr.Dataframe(value=needs_improvement_display, label="Students Needing Improvement")

    gr.Image(value=os.path.join(BASE_DIR, "boxplot_subjects.png"), label="Marks Distribution Across Subjects")

    gr.Markdown("### Full Cleaned Dataset")
    program_dropdown = gr.Dropdown(
        choices=["All"] + sorted(df["program"].unique().tolist()), value="All", label="Filter by program"
    )
    table = gr.Dataframe(value=df)
    program_dropdown.change(fn=filter_table, inputs=program_dropdown, outputs=table)


if __name__ == "__main__":
    demo.launch(server_port=7860)
