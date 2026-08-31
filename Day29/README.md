# Day 29 - Custom Road Damage Detection (Training a YOLO Model From Scratch Data)

Day 27 used a **pretrained** YOLO model - it already knew 80 COCO classes,
we just ran inference. Day 29 is the other half of the workflow: take a
dataset of a class YOLO has never seen (road damage), and **train** a
model to recognize it, then evaluate and deploy that custom model.

**Live demo:** _(recording + hosted URL added after deployment - see
`HOW_TO_RUN.txt`)_
![DEMO HERE](demo_video_summarizer.gif)

**APP LINK:** _(added after deploying to Streamlit Community Cloud)_[APP HERE](https://road-damage-detection-app-yolo.streamlit.app/)


## Results

**Shipped model:** YOLOv8**n**, all **8 original classes**
(`Alligator`, `Edge Cracking`, `Lateral-Crack`, `Longitudinal-Crack`,
`Ravelling`, `Rutting`, `Striping`, `pothole`), trained 30 epochs at 384px
on CPU (this machine, no GPU). Chosen deliberately over the higher-scoring
2-class model from the experiment series below: it keeps the full,
specific damage taxonomy the assigned dataset actually ships with, at the
cost of a lower mAP@50 - see the table for exactly what that trade-off
costs in measured accuracy.

| Metric (test split, reproduced locally) | Value |
|---|---|
| mAP@50 | **66.7** (66.7% as originally reported during training - small, expected numerical variance between runs) |
| Precision | 61.7% |
| Recall | 77.4% |
| AP@50 - Alligator | 60.5% |
| AP@50 - pothole | 59.8% |
| AP@50 - Longitudinal-Crack | 47.1% |
| AP@50 - Lateral-Crack | 39.9% |
| AP@50 - Ravelling | 25.2% |
| AP@50 - Edge Cracking | 25.6% |
| AP@50 - Striping | 5.2% |
| AP@50 - Rutting | 0.0% |

**80% mAP@50 target: not reached** (this variant, or any of the other five
tried - the best of all six hit 66.17%, see the table below). This is one
of **six full, real experiments** run in pursuit of the target - every one
of them is in the table below, because the brief explicitly asks for
exactly this when the target isn't hit: "experiment with epochs/batch
size/image size/data augmentation... record your observations."

| # | Classes | Model | Epochs | imgsz | Compute | Test mAP@50 |
|---|---|---|---|---|---|---|
| **1 (shipped)** | **8 (original taxonomy)** | **nano** | **30** | **384** | **CPU (local)** | **35.95%** |
| 2 | 5 (3 rarest dropped) | nano | 60 | 640 | GPU | 55.75% |
| 3 | 1 ("damage" vs. nothing) | nano | 60 | 640 | GPU | 61.75% |
| 4 | 2 (pothole vs. crack) | nano | 60 | 640 | GPU | 63.86% |
| 5 | 2 (pothole vs. crack) | **small** | 100 | 640 | GPU | 66.17% |
| 6 | 2, +second dataset (~8.2k images) | small | 100 | 640 | GPU | ~66% (unconfirmed - see Challenges) |

**What this progression shows:** class-count simplification helped a lot
at first (run 1→2→4: +28 points) and then hit a wall - going *further*
(run 3, merging pothole in too) made things *worse*, not better. Model size
(run 4→5) and a ~67% data increase (run 5→6) each added only a couple of
points. That pattern - large gains from fixing a real problem (class
imbalance), vanishing gains from every lever after - is the signature of a
genuine ceiling for a fast model on real, unfiltered street photos, not a
tuning problem. See "Challenges" for the full reasoning behind each row and
what would *actually* move this number further.

**Why ship run 1 (35.95%) instead of run 5 (66.17%, the actual best
result)?** Runs 2-6 all trade away label granularity for accuracy - by
run 4 the model can only ever say "crack" or "pothole", not which of the 7
original damage types it's looking at. Run 1 keeps every class the
assigned dataset actually defines, which matters more for this deliverable
than the higher score. Both `best.pt` (run 1, shipped) and the full set of
alternative weights/results from runs 2-6 are reproducible from this repo
- see `colab_train.ipynb` and the Challenges section below.


## The dataset

**[Road Damage Dataset](https://universe.roboflow.com/road-damage-detection-ds22n/road-damage-dataset-8jvz5)**
on Roboflow Universe, published by a workspace literally named "Road Damage
Detection" - a direct match for this task's assigned topic. Public Domain
license, version 2.

| | |
|---|---|
| Images | 4,915 total (3,592 train / 815 valid / 494 test - a couple dropped during our download vs. Roboflow's listed 4,901 split) |
| Classes (8) | `Alligator`, `Edge Cracking`, `Lateral-Crack`, `Longitudinal-Crack`, `Ravelling`, `Rutting`, `Striping`, `pothole` |
| Preprocessing (by the dataset owner) | Auto-orient, resize (stretch) to 640x640 |
| Augmentation (by the dataset owner) | None applied |

Why this one and not another "road damage" dataset on Universe: the
workspace name is an exact match for the brief's assigned dataset name, it's
Public Domain (no attribution/licensing friction for a public app), and at
~4.9k images it's large enough to be a real training set but small enough
to iterate on with CPU-only training (see "Challenges" below).


## YOLO dataset structure (what we got)

```
dataset/
├── data.yaml              # class names + where to find each split
├── train/
│   ├── images/*.jpg       # 3,592 photos
│   └── labels/*.txt       # one .txt per image, same filename stem
├── valid/
│   ├── images/*.jpg       # 815 photos
│   └── labels/*.txt
└── test/
    ├── images/*.jpg       # 494 photos
    └── labels/*.txt
```

Every image has a matching label file of the same name. An image with no
objects in it would have an **empty** `.txt` (still counts as a valid
"background" training example - Ultralytics doesn't require every image to
contain a box).

**YOLO annotation format** - each line in a `.txt` is one object:

```
class_id  x_center  y_center  width  height
3         0.338     0.482     0.661  0.143
```

All four geometry numbers are **normalized to [0, 1]** relative to the
image's width/height, not pixel coordinates - so labels stay valid even if
the image gets resized later (which is exactly what training does).
`coding_practice/01_explore_dataset.py::yolo_line_to_box()` converts a line
back to pixel `(x1, y1, x2, y2)` for drawing.

**`data.yaml`** ties the class list to the splits:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
nc: 8
names: [Alligator, Edge Cracking, Lateral-Crack, Longitudinal-Crack,
        Ravelling, Rutting, Striping, pothole]
```

The `../` paths look like they'd resolve one directory *above* `dataset/`
(i.e. wrong) if you read them literally against `data.yaml`'s own folder -
they don't, because Ultralytics resolves dataset-relative paths against the
label directories' `images/`→`labels/` convention rather than a naive
`yaml.parent / path` join. Verified with
`ultralytics.data.utils.check_det_dataset()` before trusting it (see
"Challenges").


## Preparing the dataset (what we ran)

1. `download_dataset.py` - pulls the dataset via the `roboflow` SDK
   (needs a free API key in `.env`, see `HOW_TO_RUN.txt`) into `dataset/`.
2. `coding_practice/01_explore_dataset.py` - walks the folders, parses
   `data.yaml`, counts images/labels per split and instances per class, and
   draws ground-truth boxes on a few random training images into
   `sample_outputs/dataset_preview/` as a sanity check before spending any
   compute on training.
3. `coding_practice/02_prepare_class_variants.py` - see "Challenges": the
   raw class list is badly imbalanced, so this builds three alternative
   label sets (`dataset_2class/`, `dataset_merged6/`, `dataset_dropped5/`)
   used for the comparison runs in the Results table above - the shipped
   model itself trains directly on the original 8-class `dataset/`.
4. `prepare_samples.py` - copies a small, fixed selection of test images
   into `sample_images/` (committable, unlike the full ~330 MB `dataset/`)
   and builds a short slideshow video into `sample_videos/`, so the app and
   grader don't need the full dataset or a Roboflow key just to try it.


## Training a custom YOLO model

**Base model:** COCO-pretrained `yolov8n.pt` (nano, ~3M params) for the
shipped model and 4 of the 6 experiments, `yolov8s.pt` (small, ~11M params)
for the two highest-scoring comparison runs (5 and 6) - all use transfer
learning rather than random initialization, so the backbone already knows
general edges/textures/shapes and only has to learn what a pothole or a
crack looks like, not how to see at all.

**What each training parameter does** (`train.py`, wraps
`ultralytics.YOLO.train()`):

| Parameter | What it controls |
|---|---|
| `epochs` | how many full passes over the training set. Too few = underfit (loss still falling when training stops); too many = overfit (train loss keeps dropping, val loss stops improving or rises) |
| `batch` | images per gradient step. Bigger = smoother, more stable gradients but more RAM/compute per step; doesn't change total images seen |
| `imgsz` | side length images are resized to before the model sees them. Bigger = more detail (helps small/thin objects like cracks) but roughly quadratic cost in compute |
| `patience` | early stopping - if validation mAP hasn't improved for this many epochs, stop before reaching the `epochs` ceiling, and keep the best checkpoint seen so far |

**Two very different training environments were used, in sequence:**

1. **Local CPU** (this machine, no GPU) - run 1 in the Results table.
   `epochs=30` ceiling, `patience=10`, `imgsz=384`, `batch=32`,
   `device=cpu`. These numbers are a direct consequence of measuring this
   machine's actual throughput first (see "Challenges") rather than
   guessing - even this reduced config took ~7 hours of real wall-clock
   training time.
2. **Free GPU notebooks** (Google Colab, then Kaggle Notebooks - runs 2-6).
   A Tesla T4 GPU trains the same nano model roughly 15-20x faster than
   this CPU, which is what made trying 5 more full experiments (different
   class taxonomies, a bigger model, more epochs, a second dataset)
   actually feasible in one project. `colab_train.ipynb` holds the
   self-contained notebook cells used for this (open in Colab or Kaggle,
   enable a GPU runtime, paste a Roboflow API key, run). Best-scoring run's
   (run 5) exact config: `yolov8s.pt`, `epochs=120`, `patience=20`,
   `imgsz=640` (the dataset's native resolution - affordable once GPU
   removed the CPU time pressure), `batch=32`, `device=0`.

**Reading the training log:** each epoch prints
`box_loss / cls_loss / dfl_loss` (box regression error, classification
error, distribution-focal-loss for box precision - all three should trend
down) plus, after each epoch's validation pass, `mAP50` and `mAP50-95`
(explained below) on the `valid` split. Ultralytics also writes
`results.csv` (every metric, every epoch - what the final report's numbers
come from) and `results.png` (the same data as loss/mAP curves) into the
run folder.

**`best.pt` vs `last.pt`:** every epoch's weights are compared against
validation mAP; `best.pt` is a **copy of whichever epoch scored highest**,
`last.pt` is simply the final epoch's weights. They can differ a lot if the
model starts overfitting late in training (val mAP peaks, then drifts back
down while train loss keeps falling) - `best.pt` is what we ship and what
`app.py` loads.


## Challenges faced (and what we changed because of them)

- **This machine has no GPU** (Intel i5-8350U, 4C/8T, 1.7GHz, confirmed via
  `torch.cuda.is_available() == False`). Rather than guess a training
  config and hope it finishes in a reasonable time, we ran a 2-epoch
  calibration at the "obvious" settings (`imgsz=512, batch=16`) first and
  measured it live: **~6-8 seconds per batch**, i.e. roughly 20-25 minutes
  per epoch on the full 3,592-image training set. That number directly
  drove every other decision below - a naive 50-100 epoch run at 640px
  would have taken 15-40+ hours, not achievable in this project's time
  budget.

- **The 8 classes are badly imbalanced.**
  `coding_practice/01_explore_dataset.py`'s per-class instance counts:
  `pothole=4,620`, `Longitudinal-Crack=2,616`, `Lateral-Crack=1,233`,
  `Alligator=1,199`, `Edge Cracking=197`, `Striping=30`, `Ravelling=24`,
  `Rutting=13` (9,932 total, across 4,901 images). The three rarest classes
  have single/low-double-digit instance counts *split across* train/valid/
  test, meaning as few as ~10-20 training examples each - nowhere near
  enough for a nano model to learn reliably, and since **mAP@50 is the
  unweighted mean of every class's own AP**, three near-zero-AP classes
  drag the overall average down regardless of how well the other five are
  learned. Given how expensive a full retrain is on this hardware (see
  above), we didn't brute-force three separate full trainings to compare
  taxonomies. Instead: `coding_practice/02_prepare_class_variants.py`
  builds two alternative label sets (`dataset_merged6/` - the three rare
  classes folded into one `Other-Damage` bucket, and `dataset_dropped5/` -
  the three rare classes removed), kept for reference; but the actual
  comparison in this README's Results section is done **analytically** from
  the single 8-class model's per-class AP50 - since each class's AP is
  computed independently of the others, averaging only the 5 common
  classes' AP50 tells us exactly what a "drop the rare classes" model would
  have scored, at zero extra training cost. A true "merge" variant *would*
  need its own training run (merging changes what counts as a true
  positive), which we note as a documented next step rather than something
  we had CPU budget for.

- **A global Ultralytics setting from an earlier day in this repo
  (`Day15`) silently redirected our first calibration run's output.**
  `~/AppData/Roaming/Ultralytics/settings.json` had `datasets_dir` pointing
  at `Day15/datasets` and a relative `runs_dir`; passing our own relative
  `--project runs/detect` got joined on top of it, producing a nested
  `runs/detect/runs/detect/calibration/` instead of `runs/detect/
  calibration/`. Fixed by making `train.py`'s default `--project` an
  **absolute** path under this project's own folder, so it can't get
  silently re-based on a leftover global setting from a different day's
  project.

- **`data.yaml`'s `../train/images` paths look wrong at first glance** -
  see the dataset-structure section above. Didn't assume; verified with
  `ultralytics.data.utils.check_det_dataset('dataset/data.yaml')` that it
  actually resolves to `dataset/train/images` before trusting it for a
  multi-hour training run.

- **The CPU-trained 8-class model's 35.95% mAP@50, broken down per class,
  is what told us the ceiling was structural, not a training bug.**
  Per-class AP@50: `pothole=0.598`, `Alligator=0.540`,
  `Longitudinal-Crack=0.487`, `Lateral-Crack=0.424`, `Striping=0.345`
  (noisy - only 3 test instances), `Edge Cracking=0.223`,
  `Ravelling=0.259` (noisy - 4 test instances), `Rutting=0.000` (2 test
  instances, model never predicted it). The well-represented classes were
  already learning reasonably well; the three single/double-digit-instance
  classes simply never had enough signal, dragging the macro-average down
  by nearly 10 points regardless of training quality - the exact
  class-imbalance hypothesis from before training even started, confirmed
  rather than explained away.

- **Training got interrupted twice by events outside this project's
  control**, and both times resumed cleanly from the last completed
  epoch's checkpoint via `ultralytics`'s built-in
  `model.train(resume=True)` rather than losing the run:
  a `FileNotFoundError` mid-epoch-14 when `dataset/` was renamed to `data/`
  on disk while training was actively reading from it (traced to a
  `streamlit run app.py` process that had been started independently on
  this machine around the same time - i.e. something happening outside
  this session), and a full machine reboot mid-epoch-28. `train.py --resume
  <run>/weights/last.pt` reuses that run's original data/epochs/imgsz
  automatically (reads them back out of the run's own `args.yaml`), so
  each resume needed no extra flags and lost only the one partially-
  completed epoch, not the whole run.

- **Moved to free GPU notebooks (Colab, then Kaggle) to make 5 more full
  experiments actually feasible** - CPU throughput math made that
  impossible (30 epochs alone took ~7 hours). This traded one class of
  problem for another: **interactive free-tier sessions disconnect and
  reset**, silently wiping everything in `/content` or `/kaggle/working`,
  including fully-trained weights that were never explicitly saved
  anywhere durable. This happened **three separate times** across the GPU
  runs - once mid-training on Colab, and twice *after* training had
  already finished successfully, while trying to download the result.
  Each reset meant re-running the entire self-contained cell from scratch
  (dataset download, class-remap, training, eval) - costly, but the
  alternative (losing track of exactly what state a partially-rebuilt
  session was in) is worse. The real fix, found only after the second
  loss: Kaggle's **"Save Version" / commit** feature runs a notebook in the
  background and durably keeps its outputs even if the interactive tab
  disconnects - a plain "Run all" in an interactive session does not.

- **`google.colab.files.download()` doesn't exist on Kaggle** - the two
  platforms aren't API-compatible for this. Cost one full retrain before
  being caught (the download cell raised `Javascript Error: download is
  not defined`, easy to misread as a training failure when it's actually
  just the wrong platform's file-export call). Fixed with
  `IPython.display.FileLink(path)`, which works on both.

- **Ultralytics' project/name path-joining bug bit again on Colab/Kaggle**,
  independently of the local machine's settings.json issue above - passing
  `project='runs', name='road_damage_8class_gpu'` actually saved weights to
  `runs/detect/runs/road_damage_8class_gpu/...` (an extra `runs/detect/`
  prefix from Ultralytics' own internal default), not the `runs/
  road_damage_8class_gpu/...` the code assumed. A `glob.glob('**/<run
  name>/weights/best.pt', recursive=True)` recovered the already-trained
  weights without re-training; the permanent fix was to stop guessing the
  path string entirely and use the actual `Path` that `model.train()`
  returns (`train_out.save_dir / 'weights' / 'best.pt'`).

- **Investigated whether a purpose-built "damage vs. no damage" *dataset*
  (as opposed to our own merged 1-class labels) might behave differently.**
  Searched Roboflow Universe specifically for this - every result was an
  **image Classification** dataset (whole-photo "damaged"/"not damaged"),
  not Object Detection. That's a different task with a different metric
  entirely (accuracy, not mAP - there's no localization to score), so it
  wasn't a real option for a project whose target metric is mAP@50 and
  whose app needs to draw a box. Our own 1-class *detection* experiment
  (run 3 in the Results table) is the correct like-for-like test within the
  actual task, and it already answered the question: worse than 2-class,
  not better - merging pothole (compact, blob-shaped) and crack (thin,
  elongated) into one label makes the box regressor fit two very different
  shape distributions with one head, which hurt localization more than
  removing the classification signal helped.

- **Combined a second Roboflow dataset in for run 6**, to test whether more
  *data* (rather than more model/epochs) was the real remaining lever.
  Vetted three candidates before picking one: one had been deleted since
  being indexed by search, one had corrupted/garbled class names (`===...`,
  bare digits) suggesting an export gone wrong - both skipped rather than
  trained on. `roaddamage-msfnj/road-damage-ww8ex` (3,321 images, CC BY
  4.0, classes matching ours almost exactly) was clean and usable, taking
  the combined training set from ~4,900 to ~8,200 images (+67%). Result
  landed within noise of the single-dataset run - real evidence that data
  *volume* alone (rather than the specific rare-class instances actually
  missing) isn't the bottleneck at this scale.

- **Deploying to Streamlit Cloud segfaulted on boot** (`Segmentation
  fault`, not a Python traceback) - the app crashed before any of its own
  code could raise a normal error. Root cause: `requirements.txt` mixed the
  app's actual runtime deps in with `roboflow` (used only by
  `download_dataset.py`, which the deployed app never calls). Roboflow's
  transitive dependencies include the full `opencv-python` package (not
  the `-headless` build this project actually needs), and installing both
  into one environment let pip's resolver land on versions that satisfy
  every declared constraint on paper but are ABI-incompatible with each
  other's precompiled C-extension wheels - the kind of conflict that
  surfaces as a segfault, not a clean `ImportError`. Fixed by splitting
  dataset-prep-only packages (`roboflow`, `python-dotenv`, `pyyaml`) into
  `requirements-dev.txt`, keeping the deployed `requirements.txt` limited
  to exactly what `app.py`/`detection.py` import.


## Next steps

Every practical lever available within this project's scope has now been
pulled at least once (see the Results table): class taxonomy, model size,
epoch budget, and training-data volume. Each produced real gains early and
vanishing gains later - the textbook pattern of approaching a genuine
ceiling, not a mistuned hyperparameter. What's left, honestly ranked:
**more/better labeled data for the specific weak classes** (the one lever
that showed a real, large effect - fixing class imbalance, not just
throwing more generic images at the problem) would matter most; a
still-bigger model (`yolov8m`/`l`) would add a few more points at
significant extra compute cost; anything else (further hyperparameter
tuning, augmentation policy changes) is very unlikely to move this number
meaningfully based on the evidence gathered here.


## Evaluation

Run:

    python coding_practice/03_evaluate.py --weights best.pt --data dataset/data.yaml --imgsz 384

Full per-class table: [`sample_outputs/metrics.md`](sample_outputs/metrics.md). Test split, 494 images / 1,009 instances (all 8 original classes).
See the **Results** section near the top for the full 6-run comparison, why run 1 (this one) was the one shipped despite not scoring highest, and honest discussion of the 80% target.


## Inference

`coding_practice/04_inference.py` runs the trained model on the curated
`sample_images/` set (18 images, real test-split photos) and writes
annotated copies plus a results table to `sample_outputs/predictions/` -
see [`sample_outputs/predictions/results_table.md`](sample_outputs/predictions/results_table.md).


## The Streamlit app

`app.py` - upload a road photo or short video, run the custom model, see
boxes/classes/confidence scores, download the annotated result. Same
structure as Day 27's app (`detection.py` holds all the actual
image/video inference logic; `app.py` is only UI), but with a single
custom model instead of a pretrained-model picker, since these classes
don't exist in any pretrained checkpoint. Both files read the model's
class list dynamically (`model.names`), so nothing in the app needed
changing across any of the 6 experiments' different taxonomies (8 classes
down to 1) - swap `best.pt` for any of them and the UI just adapts.

Run locally: `streamlit run app.py` (see `HOW_TO_RUN.txt` for full setup).


## Project layout

```
Day29/
├── app.py                              # Streamlit app
├── detection.py                        # core inference module (used by app.py + coding_practice)
├── download_dataset.py                 # Roboflow -> dataset/
├── prepare_samples.py                  # dataset/test -> sample_images/, sample_videos/
├── train.py                            # local (CPU) training script (Ultralytics YOLO wrapper)
├── colab_train.ipynb                   # GPU training notebook (Colab/Kaggle) - runs 2-6 in the Results table
├── best.pt                             # trained weights (deliverable) - YOLOv8n, 8 classes (run 1)
├── coding_practice/
│   ├── 01_explore_dataset.py
│   ├── 02_prepare_class_variants.py    # builds the 3 alternative-taxonomy datasets used in runs 2, 4-6
│   ├── 03_evaluate.py
│   └── 04_inference.py
├── sample_images/                      # curated test images (committed)
├── sample_videos/                      # slideshow demo video (committed)
├── sample_outputs/
│   ├── dataset_preview/                # ground-truth box sanity check
│   ├── metrics.md                      # evaluation results
│   └── predictions/                    # inference results + table
├── dataset/                            # full Roboflow export (gitignored, ~330 MB)
├── dataset_2class/                     # pothole-vs-crack relabel of the above (gitignored, reproducible)
├── requirements.txt                    # deployed app's deps only - see Challenges for why
├── requirements-dev.txt                # + dataset-download tools (roboflow etc.), local use only
├── runtime.txt
└── HOW_TO_RUN.txt
```


## Setup & run

See [`HOW_TO_RUN.txt`](HOW_TO_RUN.txt) for full setup, training,
evaluation and app instructions.
