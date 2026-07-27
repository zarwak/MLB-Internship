# Day 7 — Data Cleaning & Visualization

Day 7 was about turning a messy CSV into something a data analyst would actually hand over:
clean it with Pandas, then make Matplotlib/Seaborn charts that answer real questions about
the class. The day ends with a mini project — a small Streamlit dashboard summarising the
whole class at a glance.

---

## What's in this folder

```
Day7/
├── students_performance.csv           # raw dataset (20 students, 1 duplicate row, 1 missing mark)
├── data_cleaning.py                   # Task 1: cleans the raw data
├── cleaned_student_performance.csv    # output of data_cleaning.py
├── data_visualization.py              # Task 2: builds the 5 required charts
├── bar_chart_avg_score.png            # Average score per student
├── histogram_avg_score.png            # Average Score distribution
├── scatter_python_ml.png              # Python vs Machine Learning marks
├── pie_performance.png                # Performance category split
├── boxplot_subjects.png               # Marks spread across all 4 subjects
├── dashboard.py                       # Mini project: Streamlit dashboard
└── README.md                          # this file
```

> **Run order:** run `data_cleaning.py` first — it writes `cleaned_student_performance.csv`,
> which `data_visualization.py` and `dashboard.py` both read. Run everything from inside the
> `Day7` folder so the relative CSV paths resolve.

```bash
python data_cleaning.py
python data_visualization.py
streamlit run dashboard.py
```

---

## Data cleaning steps

All of this lives in `data_cleaning.py`, run against the raw `students_performance.csv`:

1. **Check missing values** — `df.isnull().sum()` first, so I know exactly what's broken
   before touching anything. Turned up 1 missing mark (Sara Ahmed's Statistics score).
2. **Handle missing values** — filled the missing numeric marks with their column mean
   (`fillna(df[numeric_cols].mean())`), rather than dropping the row and losing a whole
   student's other scores.
3. **Remove duplicates** — `drop_duplicates()`. The raw file had Ali Khan's row entered twice.
4. **Rename columns** — from `Student_ID`, `Machine_Learning`, etc. to short lowercase names
   (`student_id`, `ml`, ...) that are faster to type in the rest of the script.
5. **Change data types** — rounded the mean-filled marks back to whole numbers (`.astype(int)`)
   so a filled-in score doesn't show up as `82.0` next to everyone else's plain integers.
6. **Create `Average_Score`** — mean of the four subject columns, `axis=1` (average **across**
   a row, not down a column — same gotcha from Day 6).
7. **Create `Performance`** — bucketed from `Average_Score`: Excellent (≥90), Good (80–89),
   Average (70–79), Needs Improvement (<70).
8. **Sort & filter** — sorted by `Average_Score` descending to find the top performers, and
   filtered `program == 'AI' and Average_Score > 80` as a filter-logic demo.
9. Saved the result as `cleaned_student_performance.csv`.

**Bug I caught while doing this:** my first pass only *checked* for missing values and never
actually filled them before computing `Average_Score`. Pandas' `.mean()` silently skips `NaN`
by default, so the average still came out looking "normal" — it just quietly used 3 subjects
instead of 4 for that one student, and the duplicate row slipped through too since I'd
checked for duplicates without calling `drop_duplicates()`. Nothing errored, so it was easy to
miss. Lesson: after `isnull().sum()` / `duplicated().sum()` says something's wrong, actually
call the fix (`fillna`, `drop_duplicates`) — checking isn't cleaning.

---

## Visualizations created

Built in `data_visualization.py` from the cleaned CSV, using Matplotlib + Seaborn:

| Chart | File | What it shows |
|---|---|---|
| Bar chart | `bar_chart_avg_score.png` | Every student's average score, side by side |
| Histogram | `histogram_avg_score.png` | How average scores are distributed across the class |
| Scatter plot | `scatter_python_ml.png` | Python marks vs Machine Learning marks per student |
| Pie chart | `pie_performance.png` | Share of students in each Performance category |
| Box plot | `boxplot_subjects.png` | Spread (median, quartiles, outliers) of marks per subject |

---

## Mini project — Student Performance Dashboard

`dashboard.py` is a small Streamlit app built on top of the cleaned data. It shows, at a glance:

- **Total students** in the class
- **Average score per subject** (bar chart)
- **Top 5 students** by average score
- **Students needing improvement** (average < 70)
- **Subject with the highest average**
- The performance split (pie chart) and marks spread (box plot), plus a filterable data table

Run it with:

```bash
streamlit run dashboard.py
```

---

## 3 key insights

1. **Machine Learning has the highest class average (82.6), Python the lowest (78.9).** A
   3.7-point gap across only 4 subjects for the same students is worth a closer look —
   possibly the ML assignments overlap more with what students already practiced in Python.
2. **Python and Machine Learning marks are almost perfectly correlated (r ≈ 0.98).** Students
   who do well in Python do well in ML almost without exception (see `scatter_python_ml.png`)
   — makes sense since ML work in this course is Python-heavy, but it's a strong enough
   signal that a student struggling in Python is an early warning sign for ML too.
3. **The "Needs Improvement" group isn't evenly spread across programs.** 4 students fell
   below a 70 average, and 3 of the 4 (Fatima Noor, Hassan Tariq, Abdullah) are in the DS
   program — which also has the lowest program average overall (74.2, vs 86.0 for SE and
   81.3 for AI). If this holds beyond one class of 20, it's worth checking whether DS
   students need different support than AI/SE students.

---

## Still to do

Everything above (cleaning, charts, dashboard) is done. Two deliverables are left, and both
have to happen on my own machine/account rather than from a script — a screen recording
captures the actual desktop, and an Ngrok tunnel needs my own authtoken and has to stay
running for evaluation:

**Screen recording:** `Win + G` (Xbox Game Bar, built into Windows) → record clicking through
`streamlit run dashboard.py` in the browser → save the clip into this folder.

**Ngrok:**
```bash
pip install pyngrok
streamlit run dashboard.py          # starts on localhost:8501
ngrok config add-authtoken <your-authtoken-from-ngrok.com>
ngrok http 8501                     # copy the "Forwarding" URL it prints
```
