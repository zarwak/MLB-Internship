#  Day 6 — NumPy & Pandas

Day 6 was about moving from plain Python to the two libraries every data/ML project starts with:
**NumPy** (fast numbers) and **Pandas** (labelled tables). The day ends with a mini project —
analysing a class of 20 students' marks.

---

##  What's in this folder

```
Day6/
├── numpy_learning.ipynb              # NumPy concepts, explained step by step
├── numpy_coding_practice.ipynb       # NumPy exercises I solved
├── pandas_learning.ipynb             # Pandas concepts, explained step by step
├── pandas_coding_practice.ipynb      # Pandas exercises I solved
├── students.csv                      # sample data (auto-created by pandas_learning.ipynb)
├── student_performance.csv           # the mini-project dataset (20 students)
├── Student_Performance_Analysis.ipynb#  mini project 
├── student_analysis_processed.csv    # output: dataset + Avg_Score + Above_Avg
└── README.md                         # this file
```

>  **Run order:** open `pandas_learning.ipynb` first — one of its cells writes `students.csv`,
> which `pandas_coding_practice.ipynb` then reads. All paths are relative, so run the notebooks
> from inside the `Day6` folder.

---

##  What I learned about NumPy

NumPy gives Python a new data type — the **ndarray** — where every element is the same type and
stored together in memory. That one change fixes two problems with lists:

- **Maths just works.** `[1,2,3] * 2` repeats a list, but `np.array([1,2,3]) * 2` doubles the
  numbers. The operation is applied to every element at once — this is called **vectorisation**,
  and it means I stop writing `for` loops for arithmetic.
- **It's much faster.** Doubling a million numbers with a Python loop vs NumPy showed roughly a
  **10–50× speed difference**, because NumPy runs the loop in C instead of Python.

The pieces I now feel comfortable with:

| Topic | What clicked |
|---|---|
| **Creating arrays** | `np.array()` from a list; `zeros`, `ones`, `full`, `eye` for placeholders; `arange` (step) vs `linspace` (how many, and the end **is** included); `np.random` for practice data |
| **Inspecting** | `.shape`, `.ndim`, `.size`, `.dtype` — I check these before doing anything else |
| **Indexing & slicing** | `start:stop:step` like lists, but 2D uses **one bracket with a comma**: `a[row, col]`. `a[:, 2]` = a whole column |
| **Boolean masking** | `a[a > 70]` — writing a condition to *select* data instead of looping and `if`-ing. This turned out to be the single most useful trick |
| **Operations** | Element-wise `+ - * /`, **broadcasting** (a single number stretches to the whole array), and `*` vs `@` — `*` multiplies element by element, `@` is real matrix multiplication |
| **Math functions** | `sqrt`, `exp`, `log`, trig (in **radians**, so convert with `np.radians`), `round/floor/ceil`, and stats: `mean`, `median`, `std`, `var` |
| **`axis`** | `axis=0` collapses **down** the rows (per column), `axis=1` goes **across** (per row). I needed this constantly |
| **Reshaping** | `reshape(r, c)` keeps the data and changes the shape — only the element count must match. `-1` means "you work it out". `.T` transposes, `.flatten()` returns to 1D |

**The gotcha I'll remember:** a slice is a **view**, not a copy. Editing `a[0:3]` also edits `a`.
Use `.copy()` when you want an independent array.

---

##  What I learned about Pandas

Pandas is NumPy **with labels on top**. There are only two objects to learn:

- **Series** = one column + an index (the row labels).
- **DataFrame** = a whole table = several Series sharing one index.

The index is the whole point: pandas keeps names attached to the data, so rows never get silently
mixed up when you sort or filter.

| Topic | What clicked |
|---|---|
| **Reading CSVs** | `pd.read_csv('file.csv')` is one line. Useful options: `usecols`, `nrows`, `index_col`, `na_values`. Saving back: `to_csv('file.csv', index=False)` |
| **Exploring** | The 3-step routine for *any* new dataset: `.head()` (what does it look like) → `.info()` (types + missing values) → `.describe()` (what are the numbers doing). Plus `.shape`, `.dtypes`, `.isnull().sum()`, `.value_counts()` |
| **Selecting columns** | `df['score']` → Series; `df[['name','score']]` → DataFrame (**double brackets**) |
| **Selecting rows** | `.loc[]` works by **label** and **includes** the end of a slice; `.iloc[]` works by **position** and **excludes** it. Mixing these up was my most common error |
| **Filtering** | Write the condition first (it produces True/False), then put it in `df[...]`. Combine with `&` / `|`, and **each condition needs its own brackets**. Also `.isin()`, `.between()`, `~` to invert, `.str.contains()` |
| **Statistics** | `.mean()`, `.median()`, `.std()`, `.corr()`, and `.groupby('col')['x'].mean()` — split into groups, calculate, combine |
| **Row-wise vs column-wise** | `df[subjects].mean()` gives one average **per subject**; `df[subjects].mean(axis=1)` gives one average **per student**. Same `axis` idea as NumPy |

**What surprised me:** how much shorter pandas is. Finding the top 5 students is
`df.nlargest(5, 'Avg_Score')` — one line instead of sorting and looping myself.

---


##  Challenges I faced

**1. 2D slicing didn't behave like I expected.**
I wrote `arr2[0][0]` and it worked, so I assumed chained brackets were fine — but `arr2[:, 0]` is
the correct way to get a **column**, and `arr2[0][0]` only works by accident because `arr2[0]` is
already a row. I also tried `arr2[::3]` on a 2-row array and got back one row instead of every 3rd
number — because in 2D, the first slice applies to **rows**, not elements.
 *Fix:* always think `arr[rows, columns]`, and use `.shape` to confirm what came back.

**2. `reshape` kept throwing "cannot reshape array of size X into shape Y".**
 *Fix:* the element count must match exactly — 12 items can be 3×4 or 2×6, never 5×3. Checking
`.size` first (and using `-1` to let NumPy do the arithmetic) solved it.

**3. Filtering with `and` crashed.**
`df[df['score'] > 70 and df['course'] == 'ML']` raises *"truth value of a Series is ambiguous"*,
because Python's `and` wants a single True/False, not a whole column of them.
 *Fix:* use `&` / `|`, and wrap **each** condition in brackets — `&` binds tighter than `>`, so
without brackets it evaluates in the wrong order.

**4. Row average vs column average.**
To find each student's average I first wrote `df[subject_cols].mean()`, which gave me 4 numbers
(one per subject) instead of 20 (one per student).
 *Fix:* `df[subject_cols].mean(axis=1)` — `axis=1` works across the row.

** 5. syntax problem due to lack of familiarity.**
during slicing and indexing and applying filters, i had syntax problems
*Fix:* practice and coding


---

##  Requirements met

- [x] NumPy: arrays, arithmetic, min/max/mean/sum, reshaping, indexing & slicing
- [x] Pandas: Series/DataFrames, reading CSVs, exploring, selecting, filtering, statistics
- [x] Mini project: loaded the dataset, showed basic info, averaged each subject
- [x] Found the top 5 students and everyone scoring below average
- [x] Displayed the total number of students
- [x] Saved the processed dataset as `student_analysis_processed.csv`
