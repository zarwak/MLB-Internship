# Day 24 - Document OCR Web Application: three new features

Day 23 finished as a working, deployed four-engine OCR app. Day 24 does
not rebuild any of that - it is the same app, copied over, with three
things added on top:

1. Every extraction now shows how long it took.
2. You can upload several documents at once, and they get processed
   together using a small thread pool instead of one at a time.
3. There is a fifth engine to choose from - **RapidOCR**.

Everything about the preprocessing pipeline, the first four engines'
individual quirks, and the Streamlit Cloud deployment saga is still
exactly as Day 23's README describes it - this file only covers what is
actually new in Day 24, and the real problems that came with it.

---

## What the app does now

1. Pick an **OCR Engine** in the sidebar: EasyOCR, PaddleOCR, Tesseract,
   docTR, or **RapidOCR**.
2. Upload **one or more** document images.
3. Every document gets cleaned up and read - up to 4 at a time, in
   parallel, using the same engine you picked.
4. A results table appears: filename, time taken, confidence, lines,
   characters, status - one row per document.
5. Each row expands into the same image-plus-text view Day 23 always
   had, with a **Time taken** metric added alongside Confidence, Lines
   found and Characters.

---

## Feature 1: extraction time

Timed with `time.perf_counter()` around preprocessing and the OCR call,
stored on `OcrResult.elapsed_seconds`. Two small decisions worth
explaining:

**It does not include the one-time model load.** The very first time an
engine is used in a session, `get_reader()` also downloads/loads its
model - several seconds that have nothing to do with how long *this*
document took to read. Timing starts after the reader is already in
hand, so the first document of a session does not look artificially slow
next to every one after it. The app already has its own "Loading the OCR
model" spinner for that separate cost.

**Auto mode reports its real total, not the winning preset's own time.**
Auto mode tries two or three presets and keeps the best one. If
`elapsed_seconds` only reflected the winning preset's own run, a person
watching the spinner for 6 seconds would see "2.1s" on screen, which is
not a lie exactly, but is not what they experienced either. The number
shown is the total across every preset Auto mode actually tried.

---

## Feature 2: multiple documents at once, with threading

`st.file_uploader(..., accept_multiple_files=True)`, and a
`ThreadPoolExecutor(max_workers=4)` - a capped pool, not one thread per
document, so uploading twenty documents does not mean twenty threads all
fighting over the CPU at once.

### The real risk: is it safe to call one shared model from four threads?

Every worker thread that picks the same engine shares the exact same
cached reader object - the one `st.cache_resource` already builds once
per engine. None of the five engines document being safe to call from
multiple threads simultaneously at once. Two research findings settled
what to do about that:

- **onnxruntime's CPU execution provider** (what RapidOCR runs on) is
  explicitly documented by Microsoft as safe for concurrent `Run()`
  calls on one session.
- **Tesseract shells out to a brand new OS process per call** - there is
  no shared in-process state to race on in the first place, so it was
  never actually a question for that one.

EasyOCR, PaddleOCR and docTR have no such guarantee published anywhere.
`ocr_pipeline.run_on_reader_thread(engine, fn, ...)` routes their
inference call onto one dedicated single-worker thread per engine
instead - not a lock (see challenge 6 below for why a lock alone turned
out not to be enough) - while RapidOCR and Tesseract just run `fn`
directly, on whichever pool worker picked them up. Preprocessing always
runs fully in parallel either way, since it works on each thread's own
private numpy array with nothing shared; only the inference call itself
gets routed.

### Measuring it, instead of assuming it

These numbers were taken with the first version of the design, which
did use a plain `threading.Lock` per engine rather than the dedicated
thread it has now - see challenge 6 for why that changed. The
lock-vs-no-lock comparison below is still the reason RapidOCR and
Tesseract skip serialization at all; only *how* EasyOCR, PaddleOCR and
docTR get serialized changed afterward, not whether wall time behaves
this way. The first version locked every engine by default. Testing
that against 8 real images on 4 threads showed it barely helped at all:

| Engine | Lock | Wall time, 8 images / 4 threads |
|---|---|---|
| EasyOCR | locked (correct - no safety guarantee) | 73.7s |
| RapidOCR | locked (overly cautious) | 19.3s |
| RapidOCR | unlocked (after the fix) | **13.3s** |

Locking is not free even when it is the right call: inference is the
part that actually dominates wall time for these models, so serializing
it serializes almost the whole pipeline. RapidOCR's own case makes the
point cleanly - removing an unnecessary lock cut its 8-image batch time
by nearly a third, with identical confidence scores before and after
(0.98-0.99 both times - the numbers did not change, only the wall clock
did).

The end-to-end app confirms the same story with Auto mode running.
EasyOCR (locked) processing 3 documents together still finished in
90.7s total - close to its single slowest document's own 90.7s, not the
250s a naive sum of all three would suggest, because Auto mode's
lock-per-preset-attempt (not lock-per-document) still lets other
documents' preprocessing and their own inference attempts interleave in
the gaps. RapidOCR (unlocked) processing 4 documents finished in 17.4s -
almost exactly its slowest document's own 17.4s, essentially perfect
4-way overlap.

### Keeping one bad file from ruining the batch

Each document's whole pipeline call is wrapped in a `try`/`except`
inside `process_one()` - the function that actually runs on a worker
thread. A corrupt or unreadable file returns an error dict instead of
raising, so that document's row just shows "Error: ..." in the table and
every other document still gets processed and shown normally.

### Why no `st.*` call ever happens inside a worker thread

Streamlit only tracks UI state for the main script thread. A widget call
from a worker thread either does nothing or raises a "missing
ScriptRunContext" warning, depending on the version. `process_one()`
only ever returns a plain dictionary - every bit of rendering (the
table, the expanders, the metrics) happens back on the main thread, once
every worker has already finished.

---

## Feature 3: RapidOCR, the fifth engine

RapidOCR runs PP-OCR-family models through **ONNXRuntime** instead of a
full deep-learning framework. Unlike the other four engines, it adds no
torch/paddle equivalent of its own - just `onnxruntime` itself, a CPU
wheel of about 15-30MB with no bundled CUDA.

```python
from rapidocr import RapidOCR
engine = RapidOCR(params={
    "Global.model_root_dir": str(MODEL_CACHE_DIR / "rapidocr"),
    "Det.lang_type": "en",
    "Rec.lang_type": "en",
})
result = engine(image_array)          # takes a numpy array directly
lines = list(zip(result.txts, result.scores))
```

Two things made this easier than every other engine in this app:

- **It takes a numpy array directly.** No BGR conversion like PaddleOCR
  needs, no PIL conversion like Tesseract needs.
- **It already returns line-level text with parallel confidence
  scores** (`.txts`, `.scores`) - no manual word-to-line grouping like
  Tesseract needs, no manual block/line flattening like docTR needs.

One thing needed explicit fixing: `Det.lang_type`/`Rec.lang_type`
default to `"ch"` (Chinese) - the model still reads English text under
that setting, but explicit English is safer than accidentally shipping a
model tuned for a language this app never asks for.

Total model download: about 30MB (9.5MB detection + 0.6MB angle
classification + 20MB recognition) - noticeably lighter than any of the
other four engines' first-run download.

---

## Challenges I faced

**1. The venv silently picked up the wrong Python version.** Day 23's
venv is Python 3.11 (matches `runtime.txt`'s pin for Streamlit Cloud).
Day 24's fresh venv was created with a plain `python -m venv .venv` -
and on this machine, that resolved to Python 3.13, a newer version than
whatever `python` meant when Day 23's venv was made. Nothing announced
this; the venv was created without error. It only surfaced once
`numpy==1.26.4` (pinned for `paddleocr==2.9.1`'s sake) tried to install:
numpy 1.26.4 has no prebuilt wheel for Python 3.13, so pip fell back to
building it from source, which needs a C compiler this machine does not
have, and the install failed with a wall of meson/compiler errors that
had nothing to do with numpy itself on the surface. The fix was
recreating the venv from Day 23's exact known-good interpreter path
(`C:\Users\...\Python\Python311\python.exe`) instead of trusting
whatever `python` happens to resolve to. **Lesson: pin the interpreter
explicitly, the same way `runtime.txt` already pins it for the deployed
app - a working local venv today does not guarantee `python -m venv`
finds the same version tomorrow, even on the same machine.**

**2. Two copies of `requirements.txt` had quietly drifted apart.** While
fixing Day 23's real Streamlit Cloud deployment (a separate, earlier
session), the root `requirements.txt` got replaced with a simpler,
unpinned version - closer to a working reference build found online -
while `mini_project/requirements.txt` (the copy Streamlit Cloud actually
reads, confirmed from real deploy logs) still had the old, heavily
pinned version. Both looked plausible on their own; only comparing them
side by side showed they had stopped matching. Copying Day 23 to Day 24
carried both stale copies forward unnoticed.

**3. That drift caused a second, much worse problem.** The very first
`pip install -r requirements.txt` for Day 24 was pointed at the stale
root copy before the drift above was even discovered - installing
`paddleocr==3.7.0` and `paddlepaddle==3.3.1` instead of the pinned
`2.9.1`/`3.0.0`. `3.3.1` is *exactly* the version Day 23's README already
documents as breaking `paddleocr`'s CPU inference outright. Running
`pip install` again from the correct file fixed it - pip's resolver
noticed the pin violation and reinstalled the right versions - but it
was a reminder that a `pip install -r requirements.txt` succeeding with
exit code 0 only proves *a* requirements file installed cleanly, not
which one, and not that it was the one actually intended.

**4. `HF_HOME` was the other half of the docTR cache-redirect fix.**
`DOCTR_CACHE_DIR` (set in Day 23) was not the whole story - docTR's own
dependency on `huggingface_hub` downloads some files through Hugging
Face's own cache, which ignores `DOCTR_CACHE_DIR` completely and
defaults to `~/.cache/huggingface` on C: regardless. This was not
theoretical: C: had actually filled back up to 0 bytes free again by the
time this was noticed, months after Day 22's original discovery of the
exact same class of problem, this time via a completely different
environment variable. `HF_HOME` is the one that actually redirects it,
and it now gets set alongside `DOCTR_CACHE_DIR` in the same place.

**5. Verifying a multi-file upload needed a different tool than the
browser.** The browser automation available for this project cannot
drive a native OS file-picker dialog, so injecting files by hand -
first as base64 data, then by running a small local HTTP server for the
browser to `fetch()` from - kept hitting either message-size limits or
the browser's own cross-origin/private-network security policies.
Streamlit ships its own official testing framework
(`streamlit.testing.v1.AppTest`) specifically for this: it runs the real
`app.py` script in-process, and `file_uploader.upload(name, bytes,
mime_type)` simulates a real upload without any browser involved at
all. It turned out to be a better verification tool than the browser
would have been anyway - it directly confirms zero exceptions across the
whole script, not just that something rendered on screen, and it is what
produced the real timing numbers quoted above.

**6. A lock was not actually enough to make PaddleOCR safe - real usage
found what my own testing missed.** After all of the above shipped, a
real batch of 7 documents through PaddleOCR came back with 2-3 of them
failing: `Could not process this file: could not execute a primitive`.
My own earlier stress-testing of the locking design (challenges above,
and the table in "Measuring it") had used EasyOCR and RapidOCR, both of
which came back clean - PaddleOCR specifically was never put through the
same real multi-file, multi-thread batch before this. Two separate bugs
turned out to be layered on top of each other:

- **The lock itself had a race in how it was built.** `_reader_locks`
  was populated lazily, on first use: `if engine not in _reader_locks:
  _reader_locks[engine] = threading.Lock()`. That check-then-create is
  not atomic - with 4 worker threads all reaching PaddleOCR for the
  first time in the same instant, two of them can both see "no lock yet"
  before either has written one, so each creates its *own* separate
  `Lock()`. From then on those two threads are each waiting on a
  *different* lock object, which provides no mutual exclusion between
  them at all, despite the code looking correct on a read-through. This
  alone was a real bug, fixed by building every engine's lock once at
  import time - `{engine: threading.Lock() for engine in ENGINES if
  ...}` - before any worker thread exists, which makes the race
  structurally impossible rather than just less likely.
- **Fixing that race did not fully fix the crash.** Reproducing the
  exact scenario (7 real images, one shared PaddleOCR reader, 4 threads,
  the corrected lock) still failed 2 of 7 documents with the same error.
  Pinning all 7 calls to run on a single dedicated worker thread instead
  - `ThreadPoolExecutor(max_workers=1)`, so every call is guaranteed to
  execute on the exact same OS thread every time - made all 7 succeed,
  three separate trials in a row, zero failures. That comparison is the
  real diagnosis: a lock stops two calls from overlapping *in time*, but
  it does nothing to stop *different* threads from taking turns calling
  in, one after another. PaddlePaddle's CPU backend (Intel oneDNN)
  caches compiled primitives keyed to whichever thread first built them,
  and errors when a different thread's call later touches that same
  cached primitive - which a lock, whose entire job is letting different
  threads take turns, cannot prevent by design.

The fix: `ocr_pipeline.run_on_reader_thread(engine, fn, *args, **kwargs)`
replaced the lock entirely. For EasyOCR, PaddleOCR and docTR it submits
the call to that engine's own dedicated single-worker
`ThreadPoolExecutor`, built once at import time, and blocks for the
result - true thread affinity, not just mutual exclusion. RapidOCR and
Tesseract still run `fn` directly on whichever pool worker picked them
up, since neither one was ever the risk. Re-running the original failing
scenario through the real app (`AppTest`, PaddleOCR, the same 7 files)
afterward: 7 of 7 succeeded, confidence scores 97-100%, no exceptions.
**Lesson: a lock guarantees calls don't overlap; it does not guarantee
they all happen on the same thread. For a native library whose
thread-safety story is undocumented, those are different guarantees,
and only real multi-engine, multi-file, multi-trial testing surfaced
that PaddleOCR specifically needed the stronger one - EasyOCR and
RapidOCR being fine under a lock was not evidence that a lock was
sufficient in general, only that it was sufficient for those two.**

---

## Possible improvements

- **Show a live per-document progress state**, not just an overall
  progress bar - which document is currently running versus queued
  versus done.
- **Let a document be re-run with a different engine** from its own
  result row, without re-uploading everything.
- **A "download all as .zip" button**, next to the existing per-document
  download, for a batch that came back clean.
- **Actually load-test the free-tier RAM ceiling** once this is
  deployed - Day 23's README already flagged this as open, and adding a
  fifth engine (even a light one) does not make that question easier.

---

## What is in this folder

```
Day24/
  .venv/                   <- this app's own environment, Python 3.11
                             explicitly (see challenge 1) - all five
                             engines' worth of dependencies
  mini_project/            <- the deliverable: the Streamlit web app
    app.py                 <- start here - multi-upload, the results
                             table, and the per-document detail view
    requirements.txt       <- copy of the root one, kept next to app.py
    runtime.txt
    packages.txt
    .streamlit/config.toml
  coding_practice/          <- the Day 23 practice script, unchanged
  ocr_pipeline.py          <- all the shared OCR logic - now also holds
                             get_reader_lock() and THREAD_SAFE_ENGINES
                             for the threading feature, and the
                             RapidOCR branch in get_reader()/extract_text()
  sample_images/           <- the same 18 images from Day 23
  requirements.txt / runtime.txt / packages.txt
  HOW_TO_RUN.txt
  README.md               <- this file
```

---

## How to run it

```bash
cd D:\GitHub\ML-Bench\Day24
.venv\Scripts\activate
streamlit run mini_project\app.py
```

Starting fresh instead - **use Python 3.11 explicitly**, not whatever
`python` resolves to on the machine (see challenge 1 above):

```bash
C:\Users\<you>\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Requirements met

- [x] Shows how long each extraction took
- [x] Upload and process multiple documents at once, using a thread pool
- [x] A fifth OCR engine (RapidOCR) added and selectable
- [x] One failed document does not stop the rest of the batch from
      processing
- [x] Verified end-to-end with real documents (via Streamlit's own
      `AppTest` framework, including a run using RapidOCR specifically)

