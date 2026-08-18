# Day 22 - Optical Character Recognition (OCR)

After a week of OpenCV (filters, contours, video, and Day 21's full CV
Studio app), today moved from "understanding an image" to "reading an
image" - turning pixels into actual text a program can search, store, or
feed into another system. This is the field that makes things like
scanned-invoice processing, license-plate readers, and searchable PDF
archives possible.

## What's in this folder

- `images/` - `make_sample_images.py` generates the 15+ test images
  (5 categories x 3, plus 2 extra "hard" noisy/low-light variants).
  Synthetic on purpose - see [Why synthetic images](#why-synthetic-images-instead-of-real-photos) below.
- `ocr_practice/run_easyocr_batch.py` - runs EasyOCR on every sample
  image, saves each result to `extracted_text/`, and runs a before/after
  preprocessing comparison on the 3 hardest images.
- `extracted_text/` - one `.txt` per image, plus 3
  `*_preprocessing_comparison.txt` reports.
- `mini_project/app.py` - the Streamlit "Simple OCR Document Reader":
  upload an image, see the original + extracted text side by side,
  download or save the result.
- `requirements.txt` - dependencies for running/deploying the app.
- `.venv/` - local virtual environment (see [Why a venv this time](#why-a-venv-this-time) below).

## What is OCR?

OCR (Optical Character Recognition) is the process of converting an
image containing text - a scanned page, a photo of a receipt, a
screenshot - into machine-readable, editable text. The input is pixels;
the output is a string.

## How OCR works (conceptually)

Under the hood, OCR is really **two separate ML problems chained
together**, and understanding that split explains almost every design
choice in this folder:

1. **Text detection** - "where in this image is there text?" This is an
   object-detection problem (the same family as detecting faces or cars
   in a photo), except the "object" being detected is a line or word of
   text. The output is a set of bounding boxes.
2. **Text recognition** - "given this cropped box, what characters does
   it say?" This is a sequence-recognition problem: an image of shapes
   goes in, a string of characters comes out. Modern engines (EasyOCR,
   PaddleOCR, DocTR) solve this with a CNN (extracts visual features)
   feeding into an RNN or Transformer (reads the features left-to-right
   as a sequence) - the same detect-then-recognize pattern used in
   things like license-plate readers.

This is exactly why `easyocr.Reader.readtext()` returns a list of
`(box, text, confidence)` triples per line - it's handing back the raw
output of both stages, not just the final text.

## OCR applications

- Document digitization (scanned contracts, forms, books -> searchable text)
- Invoice / receipt processing (auto-extracting vendor, date, total)
- ID card / passport reading (KYC and check-in kiosks)
- License plate recognition
- Real-time translation apps (photograph a sign, get translated text)
- Accessibility tools (screen readers for scanned documents)

## Challenges in OCR

- **Lighting and image quality** - shadows, glare, low resolution, or
  noisy phone-camera sensors all degrade the text-detection stage first.
- **Fonts and layouts** - stylised fonts, dense multi-column layouts, or
  tables confuse both detection (where does one text region end and the
  next begin) and recognition.
- **Handwriting** - far higher variance than print; every person's pen
  strokes are a different "font," so recognition accuracy drops sharply.
- **Skew and rotation** - a tilted photo of a signboard shifts every
  character's baseline, which most recognition models assume is roughly
  horizontal.
- **Background clutter** - busy backgrounds behind text (a sign against
  foliage, a receipt with a printed logo) create false detections.
- **Language/script mixing** - documents with two languages or scripts
  in the same line need a model trained on both.

## Why image preprocessing matters before OCR

Preprocessing mostly helps the **detection** stage, not recognition -
because a detector that's hunting for "dark strokes on a light
background" gets thrown off by exactly the things preprocessing removes:

- **Grayscale conversion** collapses 3 color channels into 1. OCR only
  cares about stroke shape (dark vs. light pixels), not hue, so color is
  pure noise for this task - dropping it removes a source of confusion
  and shrinks the data the model processes.
- **Contrast enhancement (CLAHE)** stretches *local* contrast in small
  tiles across the image, instead of one global stretch. That matters
  for photos with uneven lighting (e.g. a receipt half in shadow), where
  a single global contrast boost would blow out the bright half while
  trying to fix the dark half.
- **Thresholding (Otsu)** picks the brightness cutoff that best splits
  the image's own histogram into two peaks (text vs. background)
  automatically, instead of a fixed number that only works under one
  lighting condition.

See [Preprocessing results](#preprocessing-results-raw-vs-grayscale-vs-grayscalecontrast) below for what this actually did to the noisy test images.

## Which OCR libraries support multithreading?

This isn't the same question for every library - "multithreading" here
covers a few different things: multiple CPU threads working on *one*
image, vs. multiple *processes* working on a *batch* of images.

| Library | Multithreading / parallelism support |
|---|---|
| **Tesseract** | The engine itself (compiled with OpenMP) uses up to 4 threads internally *per image* by default - controllable via the `OMP_THREAD_LIMIT` environment variable. That helps one image finish faster but doesn't parallelize a batch. For processing many images, the common pattern is the opposite: set `OMP_THREAD_LIMIT=1` and run one Tesseract process per CPU core instead. |
| **PaddleOCR** | Has explicit, documented **multi-process** batch support built in: `use_mp=True` plus `total_process_num=N` on its inference scripts, which splits a folder of images across N worker processes. This is the most out-of-the-box parallel option of the four. |
| **EasyOCR** | No dedicated multithreading/multiprocessing flag. It exposes GPU acceleration and a `batch_size` parameter for batching multiple detected text regions into one recognition-model forward pass, but parallelizing across many separate *images* on CPU is left to the caller (e.g. Python's own `multiprocessing.Pool`). |
| **DocTR** | Same story as EasyOCR - built on TensorFlow/PyTorch, so it benefits from batched tensor inference and GPU parallelism, but has no library-level multi-process flag of its own. |

**Practical takeaway:** if the goal is "process a folder of 10,000 scanned
pages as fast as possible on CPU," PaddleOCR's built-in `use_mp` is the
least amount of extra code to write. For everything else, either lean on
GPU batching or wrap the calls in your own `multiprocessing.Pool`.

## OCR libraries compared

| Library | Advantages | Limitations | Commonly used for |
|---|---|---|---|
| **Tesseract** | Small install (~10 MB), no GPU needed, very fast on clean scans, mature/stable, huge community | Accuracy drops fast on noisy/handwritten/skewed input; per-language data files needed; weaker on complex layouts | Clean scanned documents, quick CLI scripting, environments where install size matters |
| **EasyOCR** | Easiest Python API of the four (`reader.readtext(path)` and you're done); handles 80+ languages out of the box; decent on handwriting and mixed scripts | ~3x slower than Tesseract on CPU; ~500 MB+ of model weights (PyTorch); heavier dependency footprint (torch) | Quick prototyping, multilingual documents, projects where dev speed matters more than raw throughput - **this is what today's mini project uses**, exactly for that reason |
| **PaddleOCR** | Fastest of the four on GPU; only one with built-in table/layout detection; built-in multiprocessing (`use_mp`) | Baidu's PaddlePaddle framework is a second deep-learning ecosystem to install alongside/instead of PyTorch/TF, which adds setup friction | High-throughput document pipelines (invoices, forms) where structure (tables) matters, not just raw text |
| **DocTR** | Clean, modern API; supports both TensorFlow and PyTorch backends; strong on structured documents and layout-aware extraction | Smaller community than the other three; fewer pretrained language options out of the box | Document-layout-aware pipelines, research/production hybrid use cases |

## Why this OCR library: EasyOCR

EasyOCR was used for both the practice scripts and the mini project. The
reason is almost entirely the API simplicity/dev-speed tradeoff from the
table above: `easyocr.Reader(["en"]).readtext(image)` returns
ready-to-use `(box, text, confidence)` triples with no separate
detection/recognition wiring to set up, and it runs fine on CPU (slower
than PaddleOCR on GPU, but there's no GPU here anyway - see below). For
a same-day internship task testing 17 images and a Streamlit app, that
setup speed mattered more than PaddleOCR's raw throughput advantage.

## Why a venv this time

Every earlier Day in this repo installed packages into the shared global
Python (`C:\...\Python313\...`). Today that broke: the C: drive had
**0 bytes free**, so `pip install easyocr` failed mid-install with
`OSError: [Errno 28] No space left on device` (it even had to roll back
a numpy upgrade it had already started). Rather than touching the
already-full C: drive, Day22 uses its own `.venv/`, created straight on
the D: drive (same fix Day17 already used) - `python -m venv .venv`
puts all of EasyOCR's ~1.5 GB of dependencies (torch, opencv, scikit-image,
etc.) on D: instead. To run anything in this folder:

```
cd D:\GitHub\ML-Bench\Day22
.venv\Scripts\activate
```

## Why synthetic images instead of real photos

All 17 sample images are generated by `images/make_sample_images.py`
with PIL rather than sourced from real photos or Kaggle. Two reasons:

1. **Ground truth.** Because the text is rendered by code, the exact
   correct answer for every image is known upfront - so when EasyOCR
   gets something wrong, that's a real, checkable error, not a guess.
2. **Controlled variation.** Lighting, font size, and noise are
   deliberately varied per image (see the category table below) to make
   the preprocessing comparison mean something, instead of relying on
   whatever conditions a handful of found photos happened to have.

This mirrors the same pattern Day 21 used for `cv_image_studio` -
synthetic sample images so the app doesn't need real photos to
demonstrate on.

**The one caveat:** the "handwritten" category uses a script font
(Segoe Script) with small per-character jitter, which *looks*
handwritten but is not a substitute for real handwriting when it comes
to OCR difficulty - real handwriting varies in ways a single font never
will (pen pressure, connected letters, individual style). It's a
reasonable stand-in for practicing the workflow, not a rigorous
handwriting-OCR benchmark.

## Sample images generated

| Category | Files | What's varied |
|---|---|---|
| Printed document | `printed_doc_normal`, `printed_doc_small_text`, `printed_doc_dim_lighting`, `printed_doc_noisy_lowlight` | Font size, brightness, sensor noise |
| Receipt | `receipt_normal`, `receipt_faded`, `receipt_small`, `receipt_noisy_lowlight` | Brightness (faded = overexposed), font size, noise |
| Signboard | `signboard_stop`, `signboard_open`, `signboard_exit` | Rotation angle, color scheme, font size |
| Book page | `book_page_clean`, `book_page_aged`, `book_page_large_print` | Paper tint, font size |
| Handwritten note | `handwritten_note1`, `handwritten_note2`, `handwritten_note3` | Text length, font size |

## OCR results across images

Full text for every image is in `extracted_text/`; here's the average
per-line confidence EasyOCR reported for each (1.00 = fully certain):

| Image | Confidence | Notes |
|---|---|---|
| signboard_stop | 1.00 | Big bold black-on-yellow text - the easiest possible case |
| signboard_exit | 0.96 | Same reason |
| handwritten_note3 | 0.95 | Short, well-spaced script text |
| handwritten_note2 | 0.87 | |
| printed_doc_small_text | 0.88 | Small font size handled fine here - EasyOCR's detector coped |
| printed_doc_noisy_lowlight | 0.87 | Noise/dim lighting barely dented it - see why below |
| printed_doc_dim_lighting | 0.84 | |
| handwritten_note1 | 0.84 | Misread "Buy" as "Bwy" - the two letters are genuinely similar in this script font |
| receipt_noisy_lowlight | 0.91 | |
| receipt_normal | 0.85 | |
| book_page_large_print | 0.82 | |
| book_page_clean | 0.81 | |
| book_page_aged | 0.81 | Aged paper tint barely mattered |
| printed_doc_normal | 0.80 | |
| receipt_faded | 0.76 | Overexposed (washed out) - text starts to disappear into the background |
| **signboard_open** | **0.56** | White text on dark red - EasyOCR's detector treats it as lower-priority than black-on-light text |
| **receipt_small** | **0.49** | Smallest font (13px) - characters started blurring together ("FRESH" -> "FRES:", "GROCERY" -> "GROCERZ") |

**Takeaway:** the two weakest results (`signboard_open`, `receipt_small`)
both fail for reasons that make sense once you know how the detector
works - white-on-dark inverts the usual "dark strokes on light
background" assumption most of these models are tuned for, and text
below a certain pixel height simply doesn't have enough resolution left
for the recognition model to tell similar letter shapes apart.

## Preprocessing results: raw vs. grayscale vs. grayscale+contrast

This is where the results actually surprised me. The plan was: grayscale
+ CLAHE contrast + Otsu threshold should help the noisy/dim images. It
helped exactly **one** of the three test images, and badly hurt the
other two:

| Image | raw | grayscale | +CLAHE+Otsu | +denoise+CLAHE+Otsu |
|---|---|---|---|---|
| printed_doc_dim_lighting (dim, no noise) | 0.84 | 0.84 | **0.91** | 0.10 |
| printed_doc_noisy_lowlight (dim + sensor noise) | 0.89 | 0.83 | 0.03 | 0.09 |
| receipt_noisy_lowlight (dim + sensor noise) | 0.92 | 0.94 | 0.00 (nothing detected) | 0.00 (nothing detected) |

Full reports: `extracted_text/*_preprocessing_comparison.txt`.

**What's going on:** CLAHE+Otsu *helped* `printed_doc_dim_lighting`
(0.84 -> 0.91) because that image's only problem is uniformly low
brightness - Otsu's histogram-based threshold cleanly separates dark
text from light background once contrast is stretched back out. But on
the two *noisy* images, that same hard black/white cutoff turns every
noisy pixel near the threshold boundary into random black or white
speckle sitting on top of the letters - visually it looks like the text
got shredded, and EasyOCR's confidence collapsed to near zero (0.03,
and total failure - 0 characters detected - on the receipt).

My first fix attempt made it worse, not better. I added a median blur
*before* CLAHE+Otsu on the theory that removing the noise first would
let the threshold do its job cleanly. Confidence barely moved on the
noisy image (0.03 -> 0.09, still unusable) and it **tanked the one
case that had been working** (0.91 -> 0.10) - a 5x5 median blur is
roughly the same size as the text strokes themselves at this font size,
so it smeared adjacent strokes together and destroyed letter shapes
that Otsu's threshold had been handling just fine on its own.

**The actual lesson here:** classical binarization (threshold to pure
black/white) is a technique built for classical, non-neural OCR engines
like Tesseract, which genuinely need a clean bimodal image to work well.
EasyOCR's detector is a trained neural network that already learned what
text looks like under a wide range of lighting/noise conditions during
training - feeding it a raw or lightly-grayscaled image let it use all
that learned robustness, while forcing a hard binary threshold *first*
actively threw information away before the model ever saw it. For every
image in this test, **raw or plain grayscale beat every thresholded
variant** - the "improve OCR with preprocessing" advice from the
assignment brief is real, but it's engine-dependent, and blindly adding
more preprocessing steps is not automatically better.

## Challenges faced

- **The C: drive was completely full (0 bytes free)** when `pip install
  easyocr` was first run, and it failed mid-install with `OSError:
  [Errno 28] No space left on device`, having already started rolling
  back an unrelated numpy upgrade. Fixed by building a dedicated
  `.venv/` on the D: drive instead of the shared global Python - see
  "Why a venv this time" above.
- **My first two synthetic-image scripts clipped text at the canvas
  edge** (the signboard and handwritten-note generators) because I
  measured the *combined string's* width with `textbbox` but then drew
  character-by-character with per-character advance widths, and for
  these fonts the two measurements don't agree - summing individual
  glyph widths for the script font actually produces *more* total
  width than the kerned full-string measurement, so my "generously
  sized" canvas was still too narrow. Fixed by measuring width the same
  way I draw: walking the actual per-character advance loop once ahead
  of time (without drawing) to get the real pixel width needed, then
  sizing the canvas from that.
- **Preprocessing that should have helped, didn't** - see the whole
  section above. This was the most useful mistake of the day: my
  assumption that "more classical image cleanup = better OCR" turned
  out to be backwards for a neural-network-based engine, and the failed
  denoise fix made that even clearer.
- **`opencv-python-headless` failed to install on Streamlit Cloud's
  build**, even though it installed fine locally - the deployed app
  crashed at runtime with `ModuleNotFoundError: No module named 'cv2'`
  while everything else worked. Since EasyOCR already depends on
  `scikit-image` internally (proven to install cleanly in that same
  environment - EasyOCR itself needs it to even boot), `mini_project/app.py`
  was rewritten to do its grayscale + CLAHE + Otsu-threshold
  preprocessing with `skimage.color`/`exposure`/`filters` instead of
  `cv2`, and `opencv-python-headless` was dropped from `requirements.txt`
  entirely. One fewer heavy binary dependency that a platform we don't
  control could fail to build.

## Requirements met

- [x] Introduction to OCR (what/how/applications/challenges/preprocessing/multithreading) - this README
- [x] OCR libraries explored and compared (Tesseract, EasyOCR, PaddleOCR, DocTR)
- [x] EasyOCR installed and set up (in a dedicated `.venv/`)
- [x] Text read from 17 images (>= 10 required), across 5 categories
- [x] Preprocessing comparison (raw vs. grayscale vs. grayscale+CLAHE+Otsu) on 3 hard images
- [x] Mini project: Simple OCR Document Reader (`mini_project/app.py`)
- [ ] Deployed to Hugging Face Spaces / Streamlit Cloud - **pending, needs Zarwa's own account** (see below)
- [ ] Screen recording - **pending, needs Zarwa's own desktop capture**

## Deployment (needs your own account - see chat handoff)

The app is deployment-ready (`mini_project/app.py` + `requirements.txt`),
but pushing to Hugging Face Spaces / Streamlit Cloud and recording a demo
both need to happen under your own account, same as the ngrok step on
earlier Days. Exact copy-paste steps are in the chat response.
