"""
Day 7 - Data Visualization
Reads cleaned_student_performance.csv and saves 5 charts as PNG files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_student_performance.csv")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 1. Bar Chart - Average score per student
plt.figure()
plt.bar(df["student_id"], df["Average_Score"], color="skyblue")
plt.title("Average Score per Student")
plt.xlabel("Student ID")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bar_chart_avg_score.png")
plt.close()

# 2. Histogram - Average Score distribution
plt.figure()
plt.hist(df["Average_Score"], bins=10, edgecolor="black", alpha=0.7, color="green")
plt.title("Distribution of Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("histogram_avg_score.png")
plt.close()

# 3. Scatter Plot - Python vs Machine Learning marks
plt.figure()
plt.scatter(df["python"], df["ml"], alpha=0.7, c="red")
plt.title("Python vs Machine Learning Marks")
plt.xlabel("Python")
plt.ylabel("Machine Learning")
plt.tight_layout()
plt.savefig("scatter_python_ml.png")
plt.close()

# 4. Pie Chart - Performance categories
perf_counts = df["Performance"].value_counts()
plt.figure()
plt.pie(perf_counts, labels=perf_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Performance Category Distribution")
plt.axis("equal")
plt.savefig("pie_performance.png")
plt.close()

# 5. Box Plot - Marks across all subjects
df_melt = df.melt(
    id_vars=["student_id", "name"],
    value_vars=["python", "math", "stats", "ml"],
    var_name="Subject",
    value_name="Marks",
)
plt.figure()
sns.boxplot(x="Subject", y="Marks", data=df_melt)
plt.title("Marks Distribution Across Subjects")
plt.tight_layout()
plt.savefig("boxplot_subjects.png")
plt.close()

print("All 5 charts saved: bar_chart_avg_score.png, histogram_avg_score.png, "
      "scatter_python_ml.png, pie_performance.png, boxplot_subjects.png")
